import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from astro import stack
from astro.lib import alignment
from astro.lib.livestack import LiveStack
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


class MasterTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))

    def session(self, name, iso=800, temperature=None, dark_temperature=None):
        directory = self.root / name
        directory.mkdir()
        series(directory, count=3, iso=iso, temperature=temperature)
        dark_directory = self.root / f"{name}-dark"
        dark_directory.mkdir()
        dark_frames = [raw_frame(dark_directory, f"dark{index}.fit", seed=index, dark=True,
                                 temperature=dark_temperature) for index in range(3)]
        return directory, dark_frames

    def flat_frames(self, iso=800):
        flats = self.root / "flats"
        flats.mkdir()
        frames = []
        for index in range(2):
            frames.append(flats / f"flat{index}.fit")
            image = np.random.default_rng(index).normal(8000, 20, (240, 320)).astype(np.float32)
            fits.writeto(frames[-1], image, fits.Header({**HEADER, "ISO": iso, "EXPTIME": 0.02}))
        return frames

    def test_a_master_stacks_like_the_frames_it_was_combined_from(self):
        lights, dark_frames = self.session("first", dark_temperature=12)
        quietly(stack.stack_directory, lights, dark_files=dark_frames, output_prefix="frames")
        data, metadata = stack.master_dark(dark_frames)
        master = stack.write_master(dark_frames[0].parent / "dark.fit", data, metadata, dark_frames)
        header = fits.getheader(master)
        self.assertEqual((header["ISO"], header["EXPTIME"], header["NDARKS"], header["CCD-TEMP"]), (800, 5.0, 3, 12))
        output = quietly(stack.stack_directory, lights, dark_master=master, output_prefix="master")
        self.assertIn("Using the master dark of first-dark, taken at 12 °C", output)
        stacked = fits.getheader(lights / "master-stack.fit")
        self.assertTrue(stacked["DARKCOR"])
        self.assertEqual((stacked["NDARKS"], stacked["DARKTEMP"]), (3, 12))
        self.assertIn("Master dark first-dark/dark.fit", str(stacked["HISTORY"]))
        np.testing.assert_allclose(fits.getdata(lights / "master-stack.fit"),
                                   fits.getdata(lights / "frames-stack.fit"), atol=1e-3)

    def test_masters_are_not_taken_for_frames(self):
        lights, dark_frames = self.session("first")
        data, metadata = stack.master_dark(dark_frames)
        for name in ("dark.fit", "flat.fit"):
            stack.write_master(lights / name, data, metadata, dark_frames)
        quietly(stack.stack_directory, lights)
        self.assertEqual(fits.getheader(lights / "first-stack.fit")["NCOMBINE"], 3)

    def test_flats_need_not_share_the_lights_iso(self):
        lights, _ = self.session("first")
        frames = self.flat_frames(iso=100)
        data, metadata = stack.master_flat(frames)
        self.assertNotIn("ISO", metadata)
        master = stack.write_master(frames[0].parent / "flat.fit", data, metadata, frames)
        self.assertEqual(fits.getheader(master)["NFLATS"], 2)
        output = quietly(stack.stack_directory, lights, flat_master=master)
        self.assertIn("Using the master flat of flats", output)
        header = fits.getheader(lights / "first-stack.fit")
        self.assertEqual((header["FLATCOR"], header["NFLATS"]), (True, 2))

    def test_darks_far_from_the_lights_temperature_are_flagged(self):
        lights, dark_frames = self.session("first", temperature=8, dark_temperature=21)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            quietly(stack.stack_directory, lights, dark_files=dark_frames, output_prefix="warm")
        self.assertIn("the darks were taken at 21 °C and the lights at 8 °C", errors.getvalue())
        self.assertEqual(fits.getheader(lights / "warm-stack.fit")["DARKTEMP"], 21)
        data, metadata = stack.master_dark(dark_frames)
        master = stack.write_master(dark_frames[0].parent / "dark.fit", data, metadata, dark_frames)
        later, _ = self.session("later", temperature=19)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            output = quietly(stack.stack_directory, later, dark_master=master)
        self.assertIn("taken at 21 °C", output)
        self.assertEqual(errors.getvalue(), "")

    def test_each_step_reports_its_progress_in_order(self):
        lights, dark_frames = self.session("first")
        reports = []
        quietly(stack.stack_directory, lights, dark_files=dark_frames, flat_files=self.flat_frames(),
                progress=lambda *report: reports.append(report))
        stages = list(dict.fromkeys(stage for stage, _, _ in reports))
        self.assertEqual(stages, ["Combining darks", "Combining flats", "Aligning lights", "Stacking lights", "Saving"])
        for name in stages:
            steps = [(done, total) for stage, done, total in reports if stage == name]
            # Each step counts up to its end without going back
            self.assertEqual([done for done, _ in steps], sorted(done for done, _ in steps))
            self.assertEqual(steps[-1][0], steps[-1][1])
        self.assertEqual([done for stage, done, _ in reports if stage == "Aligning lights"], [0, 1, 2, 3])

    def test_frames_and_a_master_cannot_be_combined(self):
        lights, dark_frames = self.session("first")
        with self.assertRaisesRegex(ValueError, "either calibration frames or a master"):
            quietly(stack.stack_directory, lights, dark_files=dark_frames, dark_master=dark_frames[0])


if __name__ == "__main__":
    unittest.main()
