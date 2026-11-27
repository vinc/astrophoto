import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from astrophoto import stack
from astrophoto.lib import alignment, darks
from astrophoto.lib.livestack import LiveStack
from tests.test_livestack import sky

HEADER = {"INSTRUME": "Canon EOS 600D", "ISO": 800, "EXPTIME": 5.0, "BLACKLVL": 100.0, "WHITELEV": 16000,
          "BAYERPAT": "RGGB"}


def raw_frame(directory, name, dx=0.0, dy=0.0, seed=0, iso=800, dark=False, temperature=None):
    """Write a RAW-like FITS frame: a star field at twice the binned size, or a dark frame."""
    if dark:
        image = np.random.default_rng(seed).normal(150, 3, (240, 320))
    else:
        binned = sky(dx, dy, seed=seed, shape=(120, 160), stars=40)
        image = np.kron(binned, np.ones((2, 2)))
    header = fits.Header({**HEADER, "ISO": iso, "RAWFILE": name.replace(".fit", ".CR2"),
                          "DATE-OBS": f"2026-11-27T21:00:{seed:02d}.000", "DATE-AVG": f"2026-11-27T21:00:{seed:02d}.500"})
    if temperature is not None:
        header["CCD-TEMP"] = temperature
    path = Path(directory) / name
    fits.writeto(path, image.astype(np.float32), header)
    return path


def series(directory, count=5, iso=800, temperature=None):
    shifts = [(0, 0), (2, -1), (-3, 2), (4, 3), (1, -4), (-2, -2)][:count]
    return [raw_frame(directory, f"frame{index}.fit", dx, dy, seed=index, iso=iso, temperature=temperature)
            for index, (dx, dy) in enumerate(shifts)]


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        function(*args, **kwargs)
    return output.getvalue()


class SavedAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory, ignore_errors=True))
        self.frames = series(self.directory)
        live = LiveStack()
        for path in self.frames:
            live.add(stack.load(path), path)
        live.write(self.directory / "live-stack.fit", self.directory / "live-stack.png")

    def test_the_live_stack_saves_each_frame_relative_to_the_first(self):
        saved = alignment.load(self.directory)
        self.assertEqual(saved.reference, "frame0.fit")
        self.assertEqual(sorted(saved.matrices), [path.name for path in self.frames])
        np.testing.assert_allclose(saved.matrices["frame0.fit"], np.eye(3))
        # Frame 1 drifted by 2 and -1 binned pixels along x and y, so it maps back by the opposite shift
        self.assertAlmostEqual(saved.transform("frame1.fit", "frame0.fit").translation[0], -2, delta=0.2)
        # Between two frames, through the reference: frame 3 lies 2 and 4 pixels further along x and y than frame 1
        between = saved.transform("frame3.fit", "frame1.fit").translation
        self.assertAlmostEqual(between[0], -2, delta=0.3)
        self.assertAlmostEqual(between[1], -4, delta=0.3)
        self.assertIsNone(saved.transform("frame3.fit", "missing.fit"))

    def test_stack_reuses_the_saved_alignment_instead_of_matching_stars(self):
        with patch.object(stack.astroalign, "register", side_effect=AssertionError("matched stars")):
            output = quietly(stack.stack_directory, self.directory, output_prefix="cached")
        self.assertIn("Reused the saved alignment of 4 frames", output)
        expected_output = quietly(stack.stack_directory, self.directory, output_prefix="matched", realign=True)
        self.assertNotIn("Reused", expected_output)
        cached = fits.getdata(self.directory / "cached-stack.fit")
        matched = fits.getdata(self.directory / "matched-stack.fit")
        inner = (slice(10, -10), slice(10, -10))
        self.assertLess(np.nanmax(np.abs(cached[inner] - matched[inner])), 5)
        history = str(fits.getheader(self.directory / "cached-stack.fit")["HISTORY"])
        self.assertIn("Saved alignment reused for 4 frames", history)

    def test_frames_missing_from_the_saved_alignment_have_their_stars_matched(self):
        extra = raw_frame(self.directory, "frame9.fit", 3, 1, seed=9)
        with patch.object(stack.astroalign, "register", wraps=stack.astroalign.register) as register:
            output = quietly(stack.stack_directory, self.directory)
        self.assertEqual(register.call_count, 1)
        self.assertIn(f"Aligned {extra.name}", output)
        self.assertIn("Reused the saved alignment of 4 frames", output)

    def test_an_unreadable_alignment_is_ignored(self):
        (self.directory / alignment.FILENAME).write_text("{not json")
        self.assertIsNone(alignment.load(self.directory))
        output = quietly(stack.stack_directory, self.directory)
        self.assertNotIn("Reused", output)


class DarkLibraryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        patcher = patch.object(darks, "library_path", return_value=self.root / "library")
        patcher.start()
        self.addCleanup(patcher.stop)

    def session(self, name, iso=800, temperature=None, dark_temperature=None):
        directory = self.root / name
        directory.mkdir()
        series(directory, count=3, iso=iso, temperature=temperature)
        dark_directory = self.root / f"{name}-dark"
        dark_directory.mkdir()
        dark_frames = [raw_frame(dark_directory, f"dark{index}.fit", seed=index, dark=True,
                                 temperature=dark_temperature) for index in range(3)]
        return directory, dark_frames

    def test_stacking_with_darks_keeps_one_master_per_set_of_frames(self):
        lights, dark_frames = self.session("first")
        output = quietly(stack.stack_directory, lights, dark_files=dark_frames)
        masters = list((self.root / "library").glob("*.fit"))
        self.assertEqual(len(masters), 1)
        self.assertEqual(masters[0].name, "CanonEOS600D-iso800-5s-20261127210000.fit")
        self.assertIn(f"kept in the library as {masters[0].name}", output)
        header = fits.getheader(masters[0])
        self.assertEqual((header["ISO"], header["EXPTIME"], header["NDARKS"]), (800, 5.0, 3))
        quietly(stack.stack_directory, lights, dark_files=dark_frames)
        self.assertEqual(len(list((self.root / "library").glob("*.fit"))), 1)

    def test_a_later_session_without_darks_uses_the_library(self):
        lights, dark_frames = self.session("first")
        quietly(stack.stack_directory, lights, dark_files=dark_frames, output_prefix="with-darks")
        later, _ = self.session("later")
        output = quietly(stack.stack_directory, later, dark_library=True, output_prefix="library")
        self.assertIn("Using the library's master dark CanonEOS600D-iso800-5s-20261127210000.fit", output)
        header = fits.getheader(later / "library-stack.fit")
        self.assertTrue(header["DARKCOR"])
        self.assertEqual(header["NDARKS"], 3)
        self.assertIn("Library dark CanonEOS600D-iso800-5s", str(header["HISTORY"]))
        # The same master gives the same result as the frames it came from
        np.testing.assert_allclose(fits.getdata(later / "library-stack.fit"),
                                   fits.getdata(lights / "with-darks-stack.fit"), atol=1e-3)

    def test_the_newest_matching_master_is_chosen(self):
        for stamp in ("2026-10-01T20:00:00", "2026-11-20T20:00:00"):
            with patch.object(darks, "sources_digest", return_value=stamp):
                frame = self.root / f"{stamp[:10]}.fit"
                fits.writeto(frame, np.zeros((4, 4)), fits.Header({**HEADER, "DATE-OBS": stamp}))
                darks.save(np.zeros((4, 4)), {key: HEADER[key] for key in darks.KEYS}, [frame])
        self.assertIn("20261120", darks.find(fits.Header(HEADER)).name)
        self.assertIsNone(darks.find(fits.Header({**HEADER, "EXPTIME": 30.0})))

    def test_no_matching_master_stops_with_a_hint(self):
        lights, dark_frames = self.session("first")
        quietly(stack.stack_directory, lights, dark_files=dark_frames)
        later, _ = self.session("later", iso=1600)
        with self.assertRaisesRegex(ValueError, "No master dark for Canon EOS 600D at ISO 1600 and 5 s in the library; "
                                                "stack once with --dark"):
            quietly(stack.stack_directory, later, dark_library=True)

    def test_the_master_closest_in_temperature_beats_a_newer_one(self):
        for stamp, temperature in [("2026-10-01T20:00:00", 12), ("2026-11-20T20:00:00", 25), ("2026-11-25T20:00:00", None)]:
            with patch.object(darks, "sources_digest", return_value=stamp):
                frame = self.root / f"{stamp[:10]}.fit"
                extra = {} if temperature is None else {"CCD-TEMP": temperature}
                fits.writeto(frame, np.zeros((4, 4)), fits.Header({**HEADER, "DATE-OBS": stamp, **extra}))
                saved = darks.save(np.zeros((4, 4)), {key: HEADER[key] for key in darks.KEYS}, [frame])
                self.assertEqual(fits.getheader(saved).get("CCD-TEMP"), temperature)
        self.assertIn("20261001", darks.find(fits.Header(HEADER), temperature=10).name)
        self.assertIn("20261120", darks.find(fits.Header(HEADER), temperature=22).name)
        # Without a temperature to match, the newest wins
        self.assertIn("20261125", darks.find(fits.Header(HEADER)).name)

    def test_darks_far_from_the_lights_temperature_are_flagged(self):
        lights, dark_frames = self.session("first", temperature=8, dark_temperature=21)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            quietly(stack.stack_directory, lights, dark_files=dark_frames, output_prefix="warm")
        self.assertIn("the darks were taken at 21 °C and the lights at 8 °C", errors.getvalue())
        self.assertEqual(fits.getheader(lights / "warm-stack.fit")["DARKTEMP"], 21)
        later, _ = self.session("later", temperature=19)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            output = quietly(stack.stack_directory, later, dark_library=True)
        self.assertIn("taken at 21 °C", output)
        self.assertEqual(errors.getvalue(), "")

    def test_darks_and_the_library_cannot_be_combined(self):
        lights, dark_frames = self.session("first")
        with self.assertRaisesRegex(ValueError, "either dark frames or the dark library"):
            quietly(stack.stack_directory, lights, dark_files=dark_frames, dark_library=True)


if __name__ == "__main__":
    unittest.main()
