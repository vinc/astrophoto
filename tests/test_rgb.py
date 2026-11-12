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
from skimage.transform import SimilarityTransform

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
    def test_saturation_flags_handle_every_layout_and_either_green(self):
        for pattern in color.BAYER_PATTERNS:
            with self.subTest(pattern=pattern):
                data = np.full((7, 9), 128.0)
                data[:2, :2] = 4095
                greens = [i for i, value in enumerate(pattern) if value == "G"]
                for column, index in enumerate(greens, 1):
                    data[2 + index // 2, 2 * column + index % 2] = 4095
                flags = color.saturation_flags(data, header(), pattern)
                self.assertEqual(flags.shape, (3, 4))
                self.assertEqual(flags.dtype, np.uint8)
                self.assertEqual(flags[0, 0], 7)
                self.assertEqual(flags[1, 1], 2)
                self.assertEqual(flags[1, 2], 2)
                self.assertEqual(np.count_nonzero(flags), 3)
        np.testing.assert_array_equal(color.saturation_flags(data, {}, "RGGB"), 0)

    def test_saturation_is_detected_before_dark_flat_and_debanding(self):
        rgb = np.full((20, 30, 3), [100.0, 200.0, 300.0])
        rgb[8:12, 12:16] = 3967
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "light.fit"
            fits.writeto(path, mosaic(rgb) + 128, header())
            original = path.read_bytes()
            dark = np.full((40, 60), 160.0)
            for response in [0.8, 1.2]:
                with self.subTest(response=response):
                    flat = np.full(dark.shape, response)
                    result, flags = stack.load(path, dark=dark, flat=flat, rgb=True,
                                               deband=0.5, return_saturation=True)
                    np.testing.assert_array_equal(result, stack.load(path, dark=dark, flat=flat, rgb=True, deband=0.5))
                    np.testing.assert_array_equal(flags[8:12, 12:16], 7)
                    self.assertEqual(np.count_nonzero(flags), 16)
                    display = color.rgb_display_header(result, {"WBR": 2.256, "WBG": 1, "WBB": 1.269})
                    preview = color.render_rgb(result, display, (flags != 0).astype(float))
                    self.assertGreater(preview[9, 13, :3].min(), 0.99)
                    np.testing.assert_array_equal(preview[9, 13, :3], preview[9, 13, 0])
            self.assertEqual(path.read_bytes(), original)

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
    def test_rejected_saturated_samples_do_not_desaturate_the_stack(self):
        images = [np.full((10, 12, 3), [100, 200, 300], dtype=np.float32) for _ in range(7)]
        flags = [np.zeros((10, 12), dtype=np.uint8) for _ in images]
        images[0][4, 5] = 4000
        flags[0][4, 5] = 7
        result, _, accepted, fraction = stack.combine_frames(images, saturation=flags)
        np.testing.assert_array_equal(result[4, 5], [100, 200, 300])
        np.testing.assert_array_equal(accepted[4, 5], 6)
        np.testing.assert_array_equal(fraction, 0)
        images = images[:4]
        result, _, accepted, fraction = stack.combine_frames(images, saturation=flags[:4])
        self.assertEqual(fraction[4, 5], 0.25)
        np.testing.assert_array_equal(accepted[4, 5], 4)

    def test_registered_saturation_covers_interpolation_and_excludes_footprints(self):
        flags = np.zeros((80, 100), dtype=np.uint8)
        flags[40, 50] = 1
        flags[10, 20] = 4
        transform = SimilarityTransform(translation=(6.5, -3.5), rotation=0.03)
        footprint = np.zeros(flags.shape, dtype=bool)
        footprint[:20] = True
        result = stack.align_saturation(flags, transform, flags.shape, footprint)
        x, y = transform([[50, 40]])[0]
        self.assertEqual(result[round(y), round(x)], 1)
        np.testing.assert_array_equal(result[:20], 0)
        np.testing.assert_array_equal(result[60:], 0)

    def test_saturated_star_masks_follow_real_registration_and_coverage_crop(self):
        truth = star_field()
        truth[44:50, 60:66] = 5000
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for i, offset in enumerate([-2, -1, 0, 1, 2]):
                image = shift(truth, (offset, 2 * offset, 0), order=1, mode="nearest")
                raw = np.minimum(mosaic(image) + 128, 4095)
                fits.writeto(directory / f"light-{i}.fit", raw, header())
            with contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory, rgb=True, min_coverage=1)
            output = directory / f"{directory.name}-rgb-stack.fit"
            with fits.open(output) as hdus:
                rgb = np.moveaxis(hdus["RGB"].data, 0, -1)
                saturation = hdus["SATURATED"].data
                self.assertEqual(saturation.shape, hdus[0].data.shape)
                y = 47 - hdus[0].header["CROPY"]
                x = 63 - hdus[0].header["CROPX"]
                self.assertEqual(saturation[y, x], 1)
                self.assertEqual(saturation[0, 0], 0)
                expected = color.render_rgb(rgb, hdus["RGB"].header, saturation)
                np.testing.assert_allclose(expected[y, x, :3], 1, atol=0.01)
                actual = plt.imread(output.with_suffix(".png"))
                np.testing.assert_allclose(actual, expected, atol=1 / 255)
                np.testing.assert_array_equal(color.read_saturation(output), saturation)

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
    def test_highlight_feather_removes_box_edges_without_blurring_rgb(self):
        y, x = np.mgrid[:64, :64]
        profile = np.exp(-((x - 31.5)**2 + (y - 31.5)**2) / 50)
        rgb = (profile[..., None] * [5000, 3000, 2500]).astype(np.float32)
        original = rgb.copy()
        saturation = np.zeros((64, 64), dtype=np.float32)
        saturation[28:36, 28:36] = 1
        display = fits.Header({"WBR": 2.256, "WBG": 1, "WBB": 1.269,
                               "RGBLO1": 0, "RGBLO2": 0, "RGBLO3": 0,
                               "RGBHIGH": 5000, "RGBSOFT": 3})
        hard = color.render_rgb(rgb, {**display, "SATFEATH": 0}, saturation)
        smooth = color.render_rgb(rgb, display, saturation)
        hard_step = np.linalg.norm(hard[32, 28, :3] - hard[32, 27, :3])
        smooth_step = np.linalg.norm(smooth[32, 28, :3] - smooth[32, 27, :3])
        self.assertLess(smooth_step, hard_step / 2)
        hard_white = np.count_nonzero(np.ptp(hard[28:36, 28:36, :3], axis=-1) < 1e-6)
        smooth_white = np.count_nonzero(np.ptp(smooth[28:36, 28:36, :3], axis=-1) < 1e-6)
        self.assertLess(smooth_white, hard_white / 4)
        np.testing.assert_array_equal(smooth[:20], hard[:20])
        np.testing.assert_array_equal(rgb, original)

    def test_bright_halo_approaches_white_without_a_saturation_mask_boundary(self):
        y, x = np.mgrid[:64, :64]
        profile = np.exp(-((x - 31.5)**2 + (y - 31.5)**2) / 50)
        rgb = (profile[..., None] * [5000, 3000, 2500]).astype(np.float32)
        display = fits.Header({"WBR": 2.256, "WBG": 1, "WBB": 1.269,
                               "RGBLO1": 0, "RGBLO2": 0, "RGBLO3": 0,
                               "RGBHIGH": 50, "RGBSOFT": 3})
        saturation = np.zeros((64, 64), dtype=np.float32)
        saturation[28:36, 28:36] = 1
        unmasked = color.render_rgb(rgb, display)
        masked = color.render_rgb(rgb, display, saturation)
        # The bright halo outside the square mask must also be near white
        np.testing.assert_allclose(unmasked[32, 27, :3], 1, atol=0.001)
        np.testing.assert_allclose(masked[26:38, 26:38, :3],
                                   unmasked[26:38, 26:38, :3], atol=0.001)
        # Old RGB FITS files use the same shoulder without requiring restacking
        explicit = color.render_rgb(rgb, {**display, "RGBKNEE": 0.8}, saturation)
        np.testing.assert_array_equal(masked, explicit)

    def test_feather_retains_independent_highlight_strengths_and_handles_empty_maps(self):
        saturation = np.zeros((40, 60), dtype=np.float32)
        saturation[10, 10] = 1
        saturation[30, 40] = 0.25
        original = saturation.copy()
        smooth = color.feather_saturation(saturation)
        self.assertEqual(smooth[10, 10], 1)
        self.assertEqual(smooth[30, 40], 0.25)
        self.assertGreater(smooth[10, 11], 0)
        self.assertGreater(smooth[31, 40], 0)
        self.assertEqual(smooth[20, 25], 0)
        np.testing.assert_array_equal(saturation, original)
        np.testing.assert_array_equal(color.feather_saturation(np.zeros((4, 6))), 0)

    def test_saturated_highlights_are_neutral_while_unsaturated_colours_and_linear_data_remain(self):
        rgb = np.zeros((20, 30, 3), dtype=np.float32)
        rgb[8, 10] = [3000, 3000, 3000]
        rgb[8, 20] = [1500, 400, 200]
        rgb[12, 20] = [100, 200, 800]
        original = rgb.copy()
        display = color.rgb_display_header(rgb, {"WBR": 2.256, "WBG": 1, "WBB": 1.269})
        display["RGBHIGH"] = 5000
        display["SATFEATH"] = 0
        previous = color.render_rgb(rgb, display)
        self.assertLess(previous[8, 10, 1], 0.8)
        saturation = np.zeros(rgb.shape[:2], dtype=np.float32)
        saturation[8, 10] = 1
        preview = color.render_rgb(rgb, display, saturation)
        np.testing.assert_array_equal(preview[8, 10, :3], previous[8, 10, :3].max())
        np.testing.assert_array_equal(preview[saturation == 0], previous[saturation == 0])
        np.testing.assert_array_equal(rgb, original)
        saturation[8, 10] = 0.5
        halfway = color.render_rgb(rgb, display, saturation)
        np.testing.assert_allclose(halfway[8, 10, :3],
                                   (previous[8, 10, :3] + previous[8, 10, :3].max()) / 2)

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

    def test_red_heavy_sky_noise_renders_neutral_instead_of_half_clipped(self):
        rng = np.random.default_rng(7)
        # Like a 20D stack: single red and blue photosites, two greens averaged, then daylight gains
        rgb = rng.normal([60, 90, 50], [4, 4 / np.sqrt(2), 4], (300, 400, 3)).astype(np.float32)
        display = color.rgb_display_header(rgb, {"WBR": 2.256, "WBG": 1, "WBB": 1.269})
        self.assertGreater(display["RGBNOISE"], 0)
        preview = color.render_rgb(rgb, display)[..., :3].reshape(-1, 3)
        mean = preview.mean(axis=0)
        np.testing.assert_allclose(mean / mean[1], 1, atol=0.1)
        self.assertLess(float((preview == 0).mean()), 0.3)
        spread = preview.std(axis=0)
        self.assertLess(spread[0] / spread[1], 1.6)

    def test_colour_smoothing_keeps_star_colours_without_leaking_them_into_the_sky(self):
        rng = np.random.default_rng(8)
        sky = rng.normal([60, 90, 50], [4, 2.8, 4], (100, 100, 3)).astype(np.float32)
        rgb = sky.copy()
        rgb[50, 50] += [900, 300, 60]
        display = color.rgb_display_header(sky, {"WBR": 2, "WBG": 1, "WBB": 1.5})
        display["RGBHIGH"] = 20000
        preview = color.render_rgb(rgb, display)[..., :3]
        balanced = (rgb[50, 50] - [display[f"RGBLO{index}"] for index in range(1, 4)]) * [2, 1, 1.5] + display["RGBNOISE"]
        self.assertAlmostEqual(preview[50, 50, 0] / preview[50, 50, 1], balanced[0] / balanced[1], places=4)
        without = color.render_rgb(sky, display)[..., :3]
        ring = np.ones((100, 100), dtype=bool)
        ring[50, 50] = False
        self.assertLess(float(np.abs(preview - without)[ring].max()), 0.01 * float(preview[50, 50].max()))

    def test_invalid_sky_noise_is_rejected(self):
        rgb = star_field()
        display = color.rgb_display_header(rgb, {"WBR": 2, "WBG": 1, "WBB": 1.5})
        display["RGBNOISE"] = -1
        with self.assertRaisesRegex(ValueError, "sky noise"):
            color.render_rgb(rgb, display)

    def test_shared_stretch_preserves_faint_colours_rolls_off_highlights_and_masks_missing_pixels(self):
        rgb = np.zeros((20, 30, 3), dtype=np.float32)
        rgb[10, 10] = [50, 100, 200]
        rgb[10, 11] = [10000, 20000, 40000]
        rgb[0, 0] = np.nan
        metadata = {"WBR": 2, "WBG": 1, "WBB": 1}
        display = color.rgb_display_header(rgb, metadata)
        display["RGBHIGH"] = 1000
        result = color.render_rgb(rgb, display)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[0, 0], 0)
        self.assertAlmostEqual(result[10, 10, 0] / result[10, 10, 1], 1)
        self.assertAlmostEqual(result[10, 10, 2] / result[10, 10, 1], 2)
        np.testing.assert_allclose(result[10, 11, :3], 1, atol=0.001)

    def test_locate_and_crop_reuse_the_full_stacks_colour_rendering(self):
        rgb = star_field()
        rgb[48, 64] = 3967
        saturation = np.zeros(rgb.shape[:2], dtype=np.float32)
        saturation[45:50, 61:70] = 1
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
                          fits.ImageHDU(np.moveaxis(rgb, -1, 0), metadata, name="RGB"),
                          fits.ImageHDU(saturation, name="SATURATED")]).writeto(source)
            solved_header = source_header.copy()
            solved_header.update(wcs.to_header())
            fits.writeto(cached, color.luminance(rgb), solved_header)
            original = source.read_bytes()
            for radius in [None, 1, 3, 8]:
                figure, axes = plt.subplots()
                with patch.object(locate.plt, "subplots", return_value=(figure, axes)), \
                        patch("matplotlib.figure.Figure.savefig"), contextlib.redirect_stdout(io.StringIO()):
                    locate.save_image(cached, 120, 20, Time("2026-01-01"), "test",
                                      radius=radius, source_path=source)
                image = axes.images[0]
                x0, x1, y0, y1 = [int(value + 0.5) for value in image.get_extent()]
                expected = color.render_rgb(rgb, metadata, saturation)[y0:y1, x0:x1]
                np.testing.assert_allclose(image.get_array(), expected)
            self.assertEqual(source.read_bytes(), original)

    def test_solver_keeps_rgb_when_solve_field_only_copies_primary(self):
        rgb = star_field()
        saturation = np.zeros(rgb.shape[:2], dtype=np.float32)
        saturation[48, 64] = 1
        source_header = header()
        source_header["RGB"] = True
        del source_header["BAYERPAT"]
        metadata = color.rgb_display_header(rgb, {"WBR": 2, "WBG": 1, "WBB": 1.5})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "rgb.fit"
            fits.HDUList([fits.PrimaryHDU(color.luminance(rgb), source_header),
                          fits.ImageHDU(np.moveaxis(rgb, -1, 0), metadata, name="RGB"),
                          fits.ImageHDU(saturation, name="SATURATED")]).writeto(path)
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
            np.testing.assert_array_equal(color.read_saturation(output), saturation)

if __name__ == "__main__":
    unittest.main()
