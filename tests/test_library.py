from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from astro.lib import library

DARK = {"camera": "Canon EOS 600D", "iso": 800, "shutter": "5"}
FLAT = {"camera": "Canon EOS 600D", "scope": "Tamron 28-75mm f/4", "focal": 28, "aperture": 5}


class Tag:
    """An EXIF tag as exifread gives it, which reads as its printable text."""

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


def tags(exposure="1/500", temperature="18 C", **extra):
    values = {"Image Model": "Canon EOS 600D", "EXIF LensModel": "Tamron 28-75mm f/4", "EXIF FocalLength": "28",
              "EXIF ExposureTime": exposure, "EXIF FNumber": "5", "EXIF ISOSpeedRatings": "800",
              "MakerNote CameraTemperature": temperature, **extra}
    return {key: Tag(value) for key, value in values.items() if value is not None}


class LibraryCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def series(self, name, raws=2, master=False):
        directory = self.root / name
        directory.mkdir()
        for index in range(raws):
            (directory / f"IMG_{index:04d}.CR2").write_text("raw")
        if master:
            (directory / library.MASTERS[library.kind(name)]).write_text("master")
        return directory


class DescribeTests(LibraryCase):
    def test_a_series_is_described_from_its_raw_files_exif(self):
        directory = self.series("20261127-2350-dark", raws=3)
        readings = iter([tags(temperature="17 C"), tags(temperature="19 C"), tags(temperature="18 C")])
        with patch.object(library, "read_tags", side_effect=lambda path: next(readings)):
            section = library.describe(directory)
        self.assertEqual(section, {"camera": "Canon EOS 600D", "scope": "Tamron 28-75mm f/4", "focal": 28,
                                   "shutter": "1/500", "aperture": 5, "iso": 800, "temperature": 18})

    def test_exposures_and_apertures_read_as_a_camera_names_them(self):
        directory = self.series("20261127-2350-light", raws=1)
        for exposure, label in [("5", "5"), ("30", "30"), ("1/4", "1/4"), ("3/10", "0.3"), ("13/10", "1.3")]:
            with self.subTest(exposure=exposure):
                with patch.object(library, "read_tags", return_value=tags(exposure, None, **{"EXIF FNumber": "28/5"})):
                    section = library.describe(directory)
                self.assertEqual((section["shutter"], section["aperture"]), (label, 5.6))
                self.assertNotIn("temperature", section)

    def test_a_lens_without_contacts_leaves_out_what_it_does_not_record(self):
        directory = self.series("20261127-2350-light", raws=1)
        missing = {"EXIF LensModel": None, "EXIF FNumber": "0", "EXIF FocalLength": None}
        with patch.object(library, "read_tags", return_value=tags(**missing)):
            section = library.describe(directory)
        self.assertNotIn("scope", section)
        self.assertNotIn("focal", section)

    def test_without_raw_files_the_fits_frames_describe_the_series(self):
        directory = self.root / "20261127-2350-dark"
        directory.mkdir()
        for index, temperature in enumerate((16, 20)):
            header = fits.Header({"INSTRUME": "Canon EOS 600D", "ISO": 800, "EXPTIME": 0.002, "FNUMBER": 5.6,
                                  "FOCALLEN": 28.0, "CCD-TEMP": temperature})
            fits.writeto(directory / f"frame{index}.fit", np.zeros((2, 2)), header)
        # Neither the master nor a stack is a frame
        fits.writeto(directory / "dark.fit", np.zeros((2, 2)), fits.Header({"ISO": 100}))
        self.assertEqual(library.describe(directory), {"camera": "Canon EOS 600D", "focal": 28, "shutter": "1/500",
                                                       "aperture": 5.6, "iso": 800, "temperature": 18})

    def test_a_series_without_frames_is_not_described(self):
        self.assertIsNone(library.describe(self.series("20261127-2350-light", raws=0)))


class FileTests(LibraryCase):
    def test_sections_are_written_in_order_as_toml(self):
        entries = {"20261127-2350-dark": {"camera": "Canon EOS 600D", "focal": 28, "aperture": 5.6, "shutter": "1/500"},
                   "20261127-2130-flat": {"scope": 'Lens "with" quotes'}}
        library.save(self.root, entries)
        text = (self.root / library.NAME).read_text()
        self.assertTrue(text.startswith("[20261127-2130-flat]\nscope = \"Lens \\\"with\\\" quotes\"\n\n"
                                        "[20261127-2350-dark]\ncamera = \"Canon EOS 600D\"\nfocal = 28\n"))
        self.assertEqual(tomllib.loads(text), entries)
        self.assertEqual(library.load(self.root), entries)

    def test_missing_series_are_added_and_others_kept(self):
        self.series("20261127-2130-flat")
        self.series("20261127-2200-light")
        self.series("20261127-2350-dark", raws=0)
        (self.root / "notes").mkdir()
        library.save(self.root, {"20261127-2130-flat": {"camera": "edited by hand"}})
        with patch.object(library, "read_tags", return_value=tags()):
            entries = library.update(self.root, skip=("20261127-2200-light",))
        self.assertEqual(list(entries), ["20261127-2130-flat"])
        with patch.object(library, "read_tags", return_value=tags()):
            entries = library.update(self.root)
        # A series without frames yet waits for them
        self.assertEqual(sorted(entries), ["20261127-2130-flat", "20261127-2200-light"])
        self.assertEqual(entries["20261127-2130-flat"], {"camera": "edited by hand"})
        self.assertEqual(library.load(self.root), entries)

    def test_a_series_that_ends_is_described_again(self):
        self.series("20261127-2350-dark")
        library.save(self.root, {"20261127-2350-dark": {"temperature": 30}})
        with patch.object(library, "read_tags", return_value=tags()):
            library.record(self.root, "20261127-2350-dark")
        self.assertEqual(library.load(self.root)["20261127-2350-dark"]["temperature"], 18)

    def test_an_unreadable_library_is_reported_rather_than_overwritten(self):
        (self.root / library.NAME).write_text("[20261127-2350-dark\n")
        self.series("20261127-2200-light")
        with self.assertRaisesRegex(library.LibraryError, "Could not read .*library.toml"):
            library.update(self.root)
        self.assertEqual((self.root / library.NAME).read_text(), "[20261127-2350-dark\n")


class FindTests(LibraryCase):
    def add(self, entries, name, section, master=True):
        self.series(name, raws=0, master=master)
        entries[name] = section

    def test_darks_match_the_sensor_settings_then_the_temperature_then_the_newest(self):
        entries = {}
        self.add(entries, "20261001-2200-dark", {**DARK, "temperature": 12})
        self.add(entries, "20261120-2200-dark", {**DARK, "temperature": 25})
        self.add(entries, "20261125-2200-dark", {**DARK, "shutter": "5.0"})
        self.add(entries, "20261126-2200-dark", {**DARK, "iso": 1600, "temperature": 10})
        self.add(entries, "20261127-2200-dark", {**DARK, "temperature": 10}, master=False)
        find = lambda light: library.find(self.root, entries, "dark", light)
        self.assertEqual(find({**DARK, "temperature": 10}), "20261001-2200-dark")
        self.assertEqual(find({**DARK, "temperature": 22}), "20261120-2200-dark")
        # Without a temperature to match, the newest wins
        self.assertEqual(find(DARK), "20261125-2200-dark")
        self.assertIsNone(find({**DARK, "shutter": "30"}))
        self.assertIsNone(find({**DARK, "camera": "Canon EOS 20D"}))

    def test_flats_match_the_optics_whatever_the_iso_and_shutter_and_the_newest_wins(self):
        entries = {}
        self.add(entries, "20261101-1800-flat", {**FLAT, "iso": 100, "shutter": "1/50"})
        self.add(entries, "20261120-1800-flat", {**FLAT, "iso": 200})
        self.add(entries, "20261125-1800-flat", {**FLAT, "focal": 75})
        light = {**FLAT, "iso": 800, "shutter": "5"}
        self.assertEqual(library.find(self.root, entries, "flat", light), "20261120-1800-flat")
        self.assertEqual(library.find(self.root, entries, "flat", {**light, "focal": 75}), "20261125-1800-flat")
        self.assertIsNone(library.find(self.root, entries, "flat", {**light, "aperture": 8}))
        self.assertIsNone(library.find(self.root, entries, "dark", light))


if __name__ == "__main__":
    unittest.main()
