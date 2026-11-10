import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.ndimage import shift

import camera
import color
import locate
import stack

def mosaic(rgb, pattern="RGGB"):
    height, width = rgb.shape[:2]
    raw = np.empty((2 * height, 2 * width), dtype=np.float32)
    for index, colour in enumerate(pattern):
        raw[index // 2::2, index % 2::2] = rgb[..., "RGB".index(colour)]
    return raw

def header():
    return fits.Header({
        "BAYERPAT": "RGGB", "BLACKLVL": 128, "WHITELEV": 4095,
        "WBR": 2.0, "WBG": 1.0, "WBB": 1.5,
        "DATE-OBS": "2026-01-01T00:00:00", "DATE-AVG": "2026-01-01T00:00:02.500",
        "EXPTIME": 5, "ISO": 800,
    })

def star_field():
    rng = np.random.default_rng(46)
    y, x = np.mgrid[:96, :128]
    image = np.full((96, 128, 3), [80.0, 120.0, 60.0])
    for sx, sy in rng.uniform([10, 10], [118, 86], (35, 2)):
        flux = rng.uniform(100, 1000, 3)
        image += np.exp(-((x - sx)**2 + (y - sy)**2) / 3)[..., None] * flux
    return image.astype(np.float32)

class BayerTests(unittest.TestCase):
    def test_all_bayer_layouts_keep_red_blue_and_average_both_greens(self):
        expected = np.full((3, 4, 3), [10.0, 25.0, 50.0])
        for pattern in color.BAYER_PATTERNS:
            with self.subTest(pattern=pattern):
                raw = mosaic(expected, pattern)
                greens = [index for index, value in enumerate(pattern) if value == "G"]
                for index, value in zip(greens, [20, 30]):
                    raw[index // 2::2, index % 2::2] = value
                raw = np.pad(raw, ((0, 1), (0, 1)), constant_values=9999)
                rgb = color.bin_rgb(raw, pattern)
                np.testing.assert_array_equal(rgb, expected)
                np.testing.assert_allclose(color.luminance(rgb), (10 + 20 + 30 + 50) / 4)

    def test_legacy_20d_metadata_and_explicit_override(self):
        legacy = fits.Header({"INSTRUME": "Canon EOS 20D"})
        metadata = color.color_metadata(Path("old.fit"), legacy)
        self.assertEqual(metadata["BAYERPAT"], "RGGB")
        self.assertGreater(metadata["WBR"], 2)
        self.assertEqual(metadata["WBG"], 1)
        with self.assertRaisesRegex(ValueError, "BAYERPAT"):
            color.color_metadata(Path("old.fit"), fits.Header())
        self.assertEqual(color.color_metadata(Path("old.fit"), fits.Header(), "BGGR")["BAYERPAT"], "BGGR")

    def test_calibration_precedes_rgb_extraction_and_debanding(self):
        rgb = np.full((20, 30, 3), [-5.0, 30.0, 80.0])
        dark = np.full((40, 60), 128.0)
        flat = np.linspace(0.7, 1.3, 2400).reshape(40, 60)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "light.fit"
            fits.writeto(path, dark + mosaic(rgb) * flat, header())
            with patch.object(stack, "reduce_banding", side_effect=lambda image, amount: image) as deband:
                result = stack.load(path, dark=dark, flat=flat, rgb=True, deband=0.5)
            np.testing.assert_allclose(result, rgb, atol=1e-5)
            np.testing.assert_allclose(deband.call_args.args[0], rgb, atol=1e-5)
            self.assertEqual(deband.call_args.args[1], 0.5)
            fits.writeto(path, 128 + mosaic(rgb), header(), overwrite=True)
            np.testing.assert_allclose(stack.load(path, rgb=True), rgb)
            np.testing.assert_allclose(stack.load(path, flat=np.ones_like(flat), rgb=True), rgb)

    def test_rgb_debanding_preserves_each_channels_stars_and_smooth_background(self):
        y, x = np.mgrid[:128, :160]
        base = 100 + 0.01 * y**2 + np.zeros_like(x)
        base += 1000 * np.exp(-((x - 80)**2 + (y - 65)**2) / 8)
        truth = base[..., None] * np.array([0.5, 1, 1.5])
        bands = (5 * (-1.0)**y)[..., None] * np.array([1, 2, 3])
        corrected = stack.reduce_banding(truth + bands, 1)
        for channel in range(3):
            residual = np.std(np.median(corrected[..., channel] - truth[..., channel], axis=1))
            self.assertLess(residual, 0.05 * np.std(bands[..., channel]))

class RGBStackTests(unittest.TestCase):
    def test_real_registration_coloured_stars_coverage_and_linear_fits(self):
        truth = star_field()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for i, offset in enumerate([-2, -1, 0, 1, 2]):
                image = shift(truth, (offset, 2 * offset, 0), order=1, mode="nearest")
                image += np.array([i * 4, i * 8, i * 2])
                fits.writeto(directory / f"light-{i}.fit", mosaic(image) + 128, header())
            with contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory, rgb=True, min_coverage=1)
            output = directory / f"{directory.name}-rgb-stack.fit"
            with fits.open(output) as hdus:
                rgb = np.moveaxis(hdus["RGB"].data, 0, -1)
                height, width = rgb.shape[:2]
                self.assertLess(height * width, truth.shape[0] * truth.shape[1])
                self.assertEqual(hdus[0].header["NCOMBINE"], 5)
                self.assertEqual(hdus[0].header["BLACKLVL"], 0)
                self.assertTrue(hdus[0].header["RGB"])
                self.assertNotIn("BAYERPAT", hdus[0].header)
                np.testing.assert_allclose(hdus[0].data, color.luminance(rgb), rtol=1e-6)
                np.testing.assert_array_equal(hdus["COVERAGE"].data, 5)
                self.assertEqual(hdus["ACCEPTED"].data.shape, (3, height, width))
                rows = slice(hdus[0].header["CROPY"], hdus[0].header["CROPY"] + height)
                columns = slice(hdus[0].header["CROPX"], hdus[0].header["CROPX"] + width)
                expected = truth[rows, columns] + np.array([8, 16, 4])
                np.testing.assert_allclose(rgb, expected, rtol=0.02, atol=0.2)
                expected_png = color.render_rgb(rgb, hdus["RGB"].header)
                self.assertGreater(float(rgb.max()), 255)
            actual_png = plt.imread(output.with_suffix(".png"))
            np.testing.assert_allclose(actual_png, expected_png, atol=1 / 255)
            np.testing.assert_allclose(camera.read_image(output)[0], color.luminance(rgb), rtol=1e-6)
            original = fits.getdata(output).copy()
            with contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory, rgb=True, min_coverage=1)
            np.testing.assert_array_equal(fits.getdata(output), original)

    def test_background_normalization_and_clipping_are_per_channel(self):
        reference = star_field()
        image = reference + [10, -20, 30]
        normalized, offsets = stack.normalize_background(image, reference)
        np.testing.assert_allclose(normalized, reference, atol=1e-4)
        np.testing.assert_allclose(offsets, [-10, 20, -30], atol=1e-4)
        images = [reference.copy() for _ in range(7)]
        images[0][40, 50, 0] += 10000
        result, coverage, accepted = stack.combine_frames(images)
        np.testing.assert_allclose(result, reference, rtol=1e-6)
        self.assertEqual(coverage[40, 50].tolist(), [7, 7, 7])
        self.assertEqual(accepted[40, 50].tolist(), [6, 7, 7])

    def test_cli_accepts_rgb_and_validates_pattern_override(self):
        with patch.object(sys, "argv", ["stack.py", ".", "--rgb", "--bayer-pattern", "BGGR"]), \
                patch.object(stack, "stack_directory") as run:
            stack.main()
        self.assertTrue(run.call_args.kwargs["rgb"])
        self.assertEqual(run.call_args.kwargs["bayer_pattern"], "BGGR")
        with patch.object(sys, "argv", ["stack.py", ".", "--bayer-pattern", "BGGR"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            stack.main()

class RGBPreviewTests(unittest.TestCase):
    def test_noisy_sky_estimates_each_background_and_limits_noise_amplification(self):
        rng = np.random.default_rng(123)
        background = np.array([80, 120, 60])
        rgb = rng.normal(background, [4, 2, 3], (200, 300, 3)).astype(np.float32)
        rgb[90:94, 110:114] += [1000, 500, 250]
        display = color.rgb_display_header(rgb, {"WBR": 2, "WBG": 1, "WBB": 1.5})
        estimated = [display[f"RGBLO{index}"] for index in range(1, 4)]
        np.testing.assert_allclose(estimated, background, atol=0.1)
        preview = color.render_rgb(rgb, display)
        self.assertLess(float(np.median(color.luminance(preview[..., :3]))), 0.15)

    def test_shared_stretch_preserves_ratios_and_masks_missing_pixels(self):
        rgb = np.zeros((20, 30, 3), dtype=np.float32)
        rgb[10, 10] = [50, 100, 200]
        rgb[10, 11] = [10000, 20000, 40000]
        rgb[0, 0] = np.nan
        metadata = {"WBR": 2, "WBG": 1, "WBB": 1}
        display = color.rgb_display_header(rgb, metadata)
        result = color.render_rgb(rgb, display)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[0, 0], 0)
        self.assertAlmostEqual(result[10, 10, 0] / result[10, 10, 1], 1)
        self.assertAlmostEqual(result[10, 10, 2] / result[10, 10, 1], 2)
        self.assertAlmostEqual(result[10, 11, 2] / result[10, 11, 1], 2)

    def test_locate_and_crop_reuse_the_full_stacks_colour_rendering(self):
        rgb = star_field()
        metadata = color.rgb_display_header(rgb, {"WBR": 2, "WBG": 1, "WBB": 1.5})
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [120, 20]
        wcs.wcs.crpix = [64.5, 48.5]
        wcs.wcs.cdelt = [-0.02, 0.02]
        source_header = header()
        del source_header["BAYERPAT"]
        source_header.update({"NCOMBINE": 5, "RGB": True})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, cached = directory / "rgb.fit", directory / "rgb.new"
            fits.HDUList([fits.PrimaryHDU(color.luminance(rgb), source_header),
                          fits.ImageHDU(np.moveaxis(rgb, -1, 0), metadata, name="RGB")]).writeto(source)
            solved_header = source_header.copy()
            solved_header.update(wcs.to_header())
            fits.writeto(cached, color.luminance(rgb), solved_header)
            original = source.read_bytes()
            for radius in [None, 8]:
                figure, axes = plt.subplots()
                with patch.object(locate.plt, "subplots", return_value=(figure, axes)), \
                        patch("matplotlib.figure.Figure.savefig"), contextlib.redirect_stdout(io.StringIO()):
                    locate.save_image(cached, 120, 20, Time("2026-01-01"), "test",
                                      radius=radius, source_path=source)
                image = axes.images[0]
                x0, x1, y0, y1 = [int(value + 0.5) for value in image.get_extent()]
                expected = color.render_rgb(rgb, metadata)[y0:y1, x0:x1]
                np.testing.assert_allclose(image.get_array(), expected)
            self.assertEqual(source.read_bytes(), original)

    def test_solver_keeps_rgb_when_solve_field_only_copies_primary(self):
        rgb = star_field()
        source_header = header()
        source_header["RGB"] = True
        del source_header["BAYERPAT"]
        metadata = color.rgb_display_header(rgb, {"WBR": 2, "WBG": 1, "WBB": 1.5})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "rgb.fit"
            fits.HDUList([fits.PrimaryHDU(color.luminance(rgb), source_header),
                          fits.ImageHDU(np.moveaxis(rgb, -1, 0), metadata, name="RGB")]).writeto(path)
            def solve(command, **kwargs):
                wcs = WCS(naxis=2)
                wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                wcs.wcs.crval = [120, 20]
                wcs.wcs.crpix = [64.5, 48.5]
                wcs.wcs.cdelt = [-0.02, 0.02]
                solved_header = source_header.copy()
                solved_header.update(wcs.to_header())
                output = Path(command[command.index("--new-fits") + 1])
                fits.writeto(output, color.luminance(rgb), solved_header, overwrite=True)
                Path(command[command.index("--solved") + 1]).write_bytes(b"\x01")
            with patch.object(locate.subprocess, "run", side_effect=solve):
                output = locate.solve_image(path, directory, None, 3, 20, keep_solution=True)
            np.testing.assert_array_equal(color.read_rgb(output)[0], rgb)

if __name__ == "__main__":
    unittest.main()
