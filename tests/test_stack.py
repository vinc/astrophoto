import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

import numpy as np
from astropy.io import fits

import stack

class DarkTests(unittest.TestCase):
    def test_master_dark_preserves_detector_pattern_and_rejects_a_transient(self):
        y, x = np.mgrid[:8, :10]
        dark = (100 + y * 3 + x % 2 * 20).astype(np.uint16)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for i in range(3):
                data = dark.copy()
                if i == 0:
                    data[4, 5] += 1000
                path = Path(temporary) / f"dark-{i}.fit"
                fits.writeto(path, data, fits.Header({"ISO": 800, "EXPTIME": 5}))
                paths.append(path)
            result, metadata = stack.master_dark(paths)
        np.testing.assert_array_equal(result, dark)
        self.assertEqual(metadata, {"ISO": 800, "EXPTIME": 5})

    def test_subtraction_precedes_alignment_and_keeps_negative_signal_and_metadata(self):
        import camera

        y, x = np.mgrid[:8, :10]
        dark = (100 + y * 3 + x % 2 * 20).astype(np.uint16)
        signal = np.full(dark.shape, 10, dtype=int)
        signal[:2, :2] = -3
        signal[4:6, 6:8] = 100
        expected = signal.reshape(4, 2, 5, 2).mean(axis=(1, 3))
        header = fits.Header({
            "ISO": 800, "EXPTIME": 5, "INSTRUME": "test camera",
            "BLACKLVL": 100, "WHITELEV": 4095, "RAWFILE": "test.CR2",
        })
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dark_paths = []
            for i in range(3):
                path = directory / f"dark-{i}.fit"
                fits.writeto(path, dark, header)
                dark_paths.append(path)
            for i in range(2):
                fits.writeto(directory / f"light-{i}.fit",
                             (dark + signal).astype(np.uint16), header)

            def register(image, reference):
                np.testing.assert_array_equal(image, expected)
                np.testing.assert_array_equal(reference, expected)
                return image.copy(), np.zeros(image.shape, dtype=bool)

            with patch.object(stack.astroalign, "register", side_effect=register) as align, \
                    contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory, normalize=False,
                                      dark_files=[*dark_paths, dark_paths[0]])
            align.assert_called_once()
            output = directory / f"{directory.name}-stack.fit"
            data, result = fits.getdata(output, header=True)
            np.testing.assert_array_equal(data, expected)
            self.assertTrue(result["DARKCOR"])
            self.assertEqual(result["NDARKS"], 3)
            self.assertEqual(result["NCOMBINE"], 2)
            self.assertEqual(result["BLACKLVL"], 0)
            self.assertNotIn("WHITELEV", result)
            self.assertIn("Dark dark-0.fit", str(result["HISTORY"]))
            np.testing.assert_array_equal(camera.read_image(output)[0], expected)

    def test_mismatched_darks_are_rejected(self):
        cases = [
            ("shape", (6, 8), {}, "shape"),
            ("camera", (8, 8), {"INSTRUME": "other camera"}, "INSTRUME"),
            ("iso", (8, 8), {"ISO": 1600}, "ISO"),
            ("exposure", (8, 8), {"EXPTIME": 10}, "EXPTIME"),
            ("stack", (8, 8), {"NCOMBINE": 1}, "unstacked"),
            ("corrected", (8, 8), {"DARKCOR": True}, "already"),
        ]
        for name, shape, changes, message in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                header = fits.Header({"ISO": 800, "EXPTIME": 5, "INSTRUME": "test camera"})
                first, second = directory / "first.fit", directory / "second.fit"
                fits.writeto(first, np.full((8, 8), 100), header)
                header.update(changes)
                fits.writeto(second, np.full(shape, 100), header)
                with self.assertRaisesRegex(ValueError, message):
                    stack.master_dark([first, second])

    def test_nonfinite_dark_and_incompatible_light_fail_before_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "dark.fit"
            dark = np.full((8, 8), 100.0)
            dark[0, 0] = np.nan
            fits.writeto(path, dark)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                stack.master_dark([path])
            fits.writeto(path, np.full((8, 8), 100),
                         fits.Header({"ISO": 800, "EXPTIME": 5}), overwrite=True)
            fits.writeto(directory / "light.fit", np.full((8, 8), 150),
                         fits.Header({"ISO": 800, "EXPTIME": 10}))
            with self.assertRaisesRegex(ValueError, "EXPTIME"), \
                    contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory, dark_files=[path])
            self.assertFalse(list(directory.glob("*-stack.fit")))

    def test_cli_accepts_multiple_dark_paths(self):
        argv = ["stack.py", ".", "--dark", "dark-1.fit", "dark-2.fit"]
        with patch.object(sys, "argv", argv), \
                patch.object(stack, "stack_directory") as run:
            stack.main()
        self.assertEqual(run.call_args.kwargs["dark_files"],
                         [Path("dark-1.fit"), Path("dark-2.fit")])

class CombinationTests(unittest.TestCase):
    def test_clipped_mean_rejects_transients_and_preserves_stars(self):
        rng = np.random.default_rng(12)
        y, x = np.mgrid[:48, :64]
        sky = 100 + 200 * np.exp(-((x - 32)**2 + (y - 24)**2) / 8)
        images = [sky + rng.normal(0, 2, sky.shape) for _ in range(40)]
        images[0][8, 9] += 1000
        images[1][12, 13] -= 1000

        result, coverage, accepted = stack.combine_frames(images)
        median, _, _ = stack.combine_frames(images, method="median")
        np.testing.assert_array_equal(coverage, 40)
        self.assertLess(accepted[8, 9], 40)
        self.assertLess(accepted[12, 13], 40)
        self.assertAlmostEqual(result[8, 9], sky[8, 9], delta=1)
        self.assertAlmostEqual(result[12, 13], sky[12, 13], delta=1)
        self.assertAlmostEqual(result[24, 32], sky[24, 32], delta=1)
        self.assertLess(np.std(result - sky), 0.95 * np.std(median - sky))

    def test_sparse_and_empty_pixels_keep_correct_counts(self):
        images = [np.full((2, 3), 10.0) for _ in range(10)]
        for image in images:
            image[0, 0] = np.nan
            image[0, 1] = np.inf
        for image in images[2:]:
            image[1, 0] = np.nan
        images[0][1, 0] = 2
        images[1][1, 0] = 18
        images[0][1, 1] = 100

        result, coverage, accepted = stack.combine_frames(images)
        self.assertTrue(np.isnan(result[0, :2]).all())
        np.testing.assert_array_equal(coverage[0, :2], 0)
        np.testing.assert_array_equal(accepted[0, :2], 0)
        self.assertEqual(coverage[1, 0], 2)
        self.assertEqual(accepted[1, 0], 2)
        self.assertEqual(result[1, 0], 10)
        self.assertEqual(coverage[1, 1], 10)
        self.assertEqual(accepted[1, 1], 9)
        self.assertEqual(result[1, 1], 10)

    def test_normalization_uses_shared_sky_without_scaling_star_flux(self):
        rng = np.random.default_rng(5)
        y, x = np.mgrid[:64, :80]
        reference = 100 + 0.2 * x + rng.normal(0, 0.5, x.shape)
        reference += 1000 * np.exp(-((x - 40)**2 + (y - 30)**2) / 6)
        image = reference + 75
        image[:, :20] = np.nan
        normalized, offset = stack.normalize_background(image, reference)
        self.assertAlmostEqual(offset, -75, places=5)
        np.testing.assert_allclose(normalized[:, 20:], reference[:, 20:])
        self.assertTrue(np.isnan(normalized[:, :20]).all())
        with self.assertRaisesRegex(ValueError, "No finite overlap"):
            stack.normalize_background(np.full_like(image, np.nan), reference)

    def test_crop_is_the_largest_rectangle_even_with_a_hole(self):
        coverage = np.full((5, 7), 10)
        coverage[0, :] = 1
        coverage[:, 0] = 1
        coverage[2, 3] = 5
        rows, columns = stack.coverage_crop(coverage, 10)
        self.assertTrue((coverage[rows, columns] >= 10).all())
        self.assertEqual(coverage[rows, columns].size, 12)
        with self.assertRaisesRegex(ValueError, "No pixels"):
            stack.coverage_crop(coverage, 11)

class StackOutputTests(unittest.TestCase):
    def test_outputs_crop_maps_metadata_and_repeated_comparisons(self):
        rng = np.random.default_rng(19)
        sky = rng.normal(100, 1, (12, 16))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for i in range(6):
                raw = np.repeat(np.repeat(sky + i * 20, 2, axis=0), 2, axis=1)
                header = fits.Header({
                    "DATE-OBS": f"2026-01-01T00:00:{i * 5:02}",
                    "DATE-AVG": f"2026-01-01T00:00:{i * 5 + 2:02}",
                    "EXPTIME": 4,
                    "RAWFILE": f"frame-{i}.CR2",
                })
                fits.writeto(directory / f"frame-{i}.fit", raw, header)

            def register(image, reference):
                footprint = np.zeros(image.shape, dtype=bool)
                footprint[:, :2] = True
                footprint[-1, :] = True
                return image.copy(), footprint

            with patch.object(stack.astroalign, "register", side_effect=register) as align, \
                    contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory, min_coverage=1)
                output = directory / f"{directory.name}-stack.fit"
                with fits.open(output) as hdus:
                    first = hdus[0].data.copy()
                    self.assertEqual(first.shape, (11, 14))
                    np.testing.assert_allclose(first, (sky + 60)[:-1, 2:], atol=1e-4)
                    np.testing.assert_array_equal(hdus["COVERAGE"].data, 6)
                    np.testing.assert_array_equal(hdus["ACCEPTED"].data, 6)
                    header = hdus[0].header
                    self.assertEqual(header["NCOMBINE"], 6)
                    self.assertEqual(header["STACKMTH"], "SIGMEAN")
                    self.assertEqual(header["CROPX"], 2)
                    self.assertEqual(header["CROPY"], 0)
                    self.assertNotIn("RAWFILE", header)
                    self.assertEqual(header["DATE-AVG"], "2026-01-01T00:00:14.500")
                stack.stack_directory(directory, method="median", normalize=False,
                                      output_prefix="median")
                stack.stack_directory(directory, min_coverage=1)
                np.testing.assert_array_equal(fits.getdata(output), first)
                self.assertEqual(align.call_count, 15)
            self.assertTrue((directory / f"{directory.name}-coverage.png").is_file())
            self.assertEqual(fits.getheader(directory / "median-stack.fit")["STACKMTH"], "MEDIAN")
            self.assertFalse(fits.getheader(directory / "median-stack.fit")["BGNORM"])

    def test_invalid_cli_options_fail_before_stacking(self):
        for option, value in [
            ("--sigma", "nan"), ("--sigma", "0"),
            ("--min-coverage", "1.1"), ("--min-coverage", "nan"),
            ("--output-prefix", "../outside"),
        ]:
            with self.subTest(option=option, value=value), \
                    patch.object(sys, "argv", ["stack.py", ".", option, value]), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(stack, "stack_directory") as run, \
                    self.assertRaises(SystemExit):
                stack.main()
            run.assert_not_called()

if __name__ == "__main__":
    unittest.main()
