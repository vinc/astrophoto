import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers
from astropy.wcs import WCS

from lib import camera
import center
import convert as conversion
import focus
import locate

def star_field(sigma):
    rng = np.random.default_rng(42)
    y, x = np.mgrid[:192, :256]
    image = rng.normal(100, 0.2, x.shape)
    for sx, sy in [(40.2, 40.3), (100.1, 50.4), (180.3, 120.2), (60.4, 140.1)]:
        image += 1000 * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * sigma**2))
    return image

def solved_header():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [150.5, 100.5]
    wcs.wcs.cd = [[-0.01732, -0.01], [-0.01, 0.01732]]
    wcs.wcs.crval = [120, 20]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["DATE-AVG"] = "2025-01-01T00:00:00.000"
    return header

class PreviewTests(unittest.TestCase):
    def test_noninteractive_backend_has_a_clear_error_and_closes_figure(self):
        figures = camera.plt.get_fignums()
        with self.assertRaisesRegex(SystemExit, "cannot open a preview window") as error:
            camera.preview_figure()
        self.assertIn("python -m pip install PyQt6", str(error.exception))
        self.assertIn("--no-plot", str(error.exception))
        self.assertEqual(camera.plt.get_fignums(), figures)

    def test_both_helpers_check_window_support_before_capturing(self):
        commands = [
            (focus, ["focus.py"]),
            (center, ["center.py", "--query", "vesta", "--latitude", "0", "--longitude", "0", "--elevation", "0"]),
        ]
        for module, argv in commands:
            with self.subTest(module=module.__name__), \
                    patch.object(sys, "argv", argv), patch.object(module, "capture") as capture, \
                    self.assertRaisesRegex(SystemExit, "cannot open a preview window"):
                module.main()
            capture.assert_not_called()

class FocusTests(unittest.TestCase):
    def test_close_pairs_do_not_discard_other_in_focus_stars(self):
        y, x = np.mgrid[:256, :320]
        for radius in [12, 24]:
            for separation, brightness in [(10, 0.5), (12, 0.2)]:
                with self.subTest(radius=radius, separation=separation):
                    data = np.random.default_rng(42).normal(100, 0.2, x.shape)
                    for sx, sy in [(40.2, 40.3), (100.1, 50.4), (180.3, 120.2), (60.4, 140.1)]:
                        data += 1000 * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * 1.5**2))
                    for sx, peak in [(260.2, 5000), (260.2 + separation, 5000 * brightness)]:
                        data += peak * np.exp(-((x - sx)**2 + (y - 200.3)**2) / (2 * 1.5**2))
                    stars = focus.measure_stars(data, radius=radius)
                    self.assertEqual(len(stars), 4)
                    self.assertTrue(all(sx < 220 for sx, _, _ in stars))
                    self.assertAlmostEqual(np.median([star[2] for star in stars]), 1.1774 * 1.5, delta=0.2)

    def test_crowded_fields_stay_measurable_with_a_larger_aperture(self):
        y, x = np.mgrid[:512, :512]
        for seed in range(3):
            rng = np.random.default_rng(seed)
            data = rng.normal(100, 0.2, x.shape)
            for sx, sy, peak in rng.uniform([30, 30, 100], [482, 482, 1000], (150, 3)):
                data += peak * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * 1.5**2))
            for radius in [12, 24]:
                with self.subTest(seed=seed, radius=radius):
                    stars = focus.measure_stars(data, radius=radius)
                    self.assertEqual(len(stars), 20)
                    self.assertAlmostEqual(np.median([star[2] for star in stars]), 1.1774 * 1.5, delta=0.2)

    def test_one_broad_candidate_does_not_discard_a_sharp_star(self):
        y, x = np.mgrid[:256, :256]
        data = np.random.default_rng(42).normal(100, 0.2, x.shape)
        data += 5000 * np.exp(-((x - 80.2)**2 + (y - 80.3)**2) / (2 * 8**2))
        data += 1000 * np.exp(-((x - 180.2)**2 + (y - 180.3)**2) / (2 * 1.5**2))
        stars = focus.measure_stars(data)
        self.assertEqual(len(stars), 1)
        self.assertAlmostEqual(stars[0][0], 180.2, delta=0.2)
        self.assertAlmostEqual(stars[0][2], 1.1774 * 1.5, delta=0.2)

    def test_broad_gaussians_and_rings_need_a_larger_aperture(self):
        y, x = np.mgrid[:256, :256]
        distance = np.hypot(x - 128.2, y - 128.3)
        for kind, widths in [("gaussian", [5, 8]), ("ring", [6, 10])]:
            measured = []
            for width in widths:
                data = np.random.default_rng(42).normal(100, 0.2, distance.shape)
                if kind == "gaussian":
                    data += 1000 * np.exp(-distance**2 / (2 * width**2))
                    expected = 1.1774 * width
                else:
                    data += 100 * np.exp(-(distance - width)**2 / 2)
                    expected = width
                with self.subTest(kind=kind, width=width), \
                        self.assertRaisesRegex(focus.FocusRangeError, "out of range"):
                    focus.measure_stars(data)
                stars = focus.measure_stars(data, radius=32)
                self.assertEqual(len(stars), 1)
                measured.append(stars[0][2])
                self.assertAlmostEqual(measured[-1], expected, delta=0.4)
            self.assertGreater(measured[1], measured[0] * 1.4)

    def test_out_of_range_frame_does_not_replace_best_focus(self):
        argv = ["focus.py", "--no-plot", "--frames", "3", "--interval", "0"]
        with patch.object(sys, "argv", argv), \
                patch.object(focus, "capture", return_value=Path("capture.fit")), \
                patch.object(focus, "read_image", side_effect=[
                    (star_field(1.5), None, 1), (star_field(8), None, 1), (star_field(3), None, 1),
                ]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            focus.main()
        readings = [line for line in output.getvalue().splitlines() if line.startswith("capture.fit:")]
        self.assertEqual(len(readings), 3)
        self.assertIn("out of range", readings[1])
        self.assertNotIn("best", readings[1])
        self.assertEqual(readings[0].split("best ")[1], readings[2].split("best ")[1])

    def test_live_capture_failure_is_retried_without_counting_a_frame(self):
        argv = ["focus.py", "--no-plot", "--frames", "1", "--interval", "0", "--radius", "32"]
        with patch.object(sys, "argv", argv), \
                patch.object(focus, "capture", side_effect=[
                    subprocess.CalledProcessError(1, "gphoto2"), Path("capture.fit"),
                ]) as capture, \
                patch.object(focus, "read_image", return_value=(star_field(3), None, 1)), \
                patch.object(focus.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            focus.main()
        self.assertEqual(capture.call_count, 2)
        self.assertIn("retrying with the next capture", output.getvalue())
        self.assertIn("HFR", output.getvalue())

    def test_saved_image_defaults_to_text_measurements(self):
        argv = ["focus.py", "saved.fit"]
        with patch.object(sys, "argv", argv), \
                patch.object(focus, "read_image", return_value=(star_field(1.5), None, 1)), \
                patch.object(focus, "preview_figure") as preview, \
                patch.object(focus.plt, "show") as show, \
                patch.object(focus.plt, "pause") as pause, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            focus.main()
        preview.assert_not_called()
        show.assert_not_called()
        pause.assert_not_called()
        self.assertIn("HFR", output.getvalue())
        self.assertNotIn("best", output.getvalue())

    def test_saved_image_can_write_a_preview_without_a_gui(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "focus.png"
            argv = ["focus.py", "saved.fit", "--save-preview", str(output)]
            with patch.object(sys, "argv", argv), \
                    patch.object(focus, "read_image", return_value=(star_field(1.5), None, 1)), \
                    patch.object(focus, "preview_figure") as preview, \
                    patch.object(focus.plt, "show") as show, \
                    patch.object(focus.plt, "pause") as pause, \
                    contextlib.redirect_stdout(io.StringIO()):
                focus.main()
            self.assertGreater(output.stat().st_size, 1000)
        preview.assert_not_called()
        show.assert_not_called()
        pause.assert_not_called()

    def test_defocus_does_not_replace_stars_with_unsaturated_warm_pixels(self):
        rng = np.random.default_rng(17)
        y, x = np.mgrid[:512, :512]
        measured = []
        for sigma in [1, 2, 3, 4]:
            data = rng.normal(100, 1, x.shape)
            for j in range(4):
                for i in range(5):
                    sx, sy = 40.2 + i * 55, 40.3 + j * 60
                    data += 2000 / sigma**2 * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * sigma**2))
            for j in range(4):
                for i in range(10):
                    data[310 + j * 45, 35 + i * 45] += 200
            stars = focus.measure_stars(data)
            self.assertEqual(len(stars), 20)
            self.assertTrue(all(sy < 280 for _, sy, _ in stars))
            measured.append(np.median([hfr for _, _, hfr in stars]))
        self.assertTrue(np.all(np.diff(measured) > 0), measured)
        self.assertGreater(measured[-1], 3 * measured[0])

    def test_background_noise_does_not_become_extra_star_flux(self):
        rng = np.random.default_rng(23)
        y, x = np.mgrid[:192, :256]
        measured = []
        for peak in [1000, 30]:
            image = rng.normal(100, 1, x.shape)
            for sx, sy in [(40.2, 40.3), (100.1, 50.4), (180.3, 120.2), (60.4, 140.1)]:
                image += peak * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * 1.5**2))
            stars = focus.measure_stars(image)
            self.assertEqual(len(stars), 4)
            measured.append(np.median([hfr for _, _, hfr in stars]))
        self.assertLess(abs(measured[0] - measured[1]), 0.4)

    def test_hfr_increases_with_defocus(self):
        sharp = focus.measure_stars(star_field(1.5))
        blurred = focus.measure_stars(star_field(3))
        self.assertEqual(len(sharp), 4)
        self.assertEqual(len(blurred), 4)
        sharp_hfr = np.median([star[2] for star in sharp])
        blurred_hfr = np.median([star[2] for star in blurred])
        self.assertAlmostEqual(sharp_hfr, 1.1774 * 1.5, delta=0.3)
        self.assertGreater(blurred_hfr, sharp_hfr * 1.7)

    def test_saturated_stars_and_hot_pixels_are_excluded(self):
        image = star_field(1.5)
        image[80, 220] = 10000
        saturated = np.zeros(image.shape, dtype=bool)
        saturated[40, 40] = True
        stars = focus.measure_stars(image, saturated)
        self.assertEqual(len(stars), 3)
        self.assertTrue(all(np.hypot(x - 40, y - 40) > 12 for x, y, _ in stars))

    def test_raw_binning_preserves_saturation_from_any_bayer_pixel(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.fit"
            data = np.full((20, 20), 100, dtype=np.uint16)
            data[4, 6] = 4095
            header = fits.Header({"RAWFILE": "test.CR2", "BLACKLVL": 100, "WHITELEV": 4095})
            fits.writeto(path, data, header)
            image, saturated, scale = camera.read_image(path)
            self.assertEqual(image.shape, (10, 10))
            self.assertEqual(scale, 2)
            self.assertTrue(saturated[2, 3])
            self.assertEqual(image[0, 0], 0)

class CenterTests(unittest.TestCase):
    def test_live_correction_keeps_camera_fixed_and_advances_target_time(self):
        exposure = Time("2025-01-01T00:00:00")
        current = Time("2025-01-01T00:10:00")
        observer = {"lat": 0, "lon": 0, "elevation": 0}
        frame = AltAz(obstime=exposure, location=EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg))
        with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
            target = SkyCoord(az=90 * u.deg, alt=40 * u.deg, frame=frame).icrs
            future = target.transform_to(AltAz(obstime=current, location=frame.location))
        pan, tilt = center.pointing_offset(target, target, exposure, observer, target_time=current)
        self.assertAlmostEqual(pan, (future.az.deg - 90 + 180) % 360 - 180, places=5)
        self.assertAlmostEqual(tilt, future.alt.deg - 40, places=5)
        self.assertGreater(abs(tilt), 2)
        old_pan, old_tilt = center.pointing_offset(target, target, exposure, observer)
        self.assertAlmostEqual(old_pan, 0, places=5)
        self.assertAlmostEqual(old_tilt, 0, places=5)

    def test_live_lookup_updates_target_but_marker_uses_exposure_position(self):
        header = solved_header()
        exposure = Time(header["DATE-AVG"])
        current = exposure + 60 * u.s
        original = WCS(header).pixel_to_world(100, 80)
        from astropy.table import Table
        positions = Table({
            "RA": [original.ra.deg, original.ra.deg + 0.01],
            "DEC": [original.dec.deg, original.dec.deg + 0.01],
            "targetname": ["test", "test"],
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.new"
            fits.writeto(path, np.zeros((200, 300)), header)
            with patch.object(center, "get_ephemerides", return_value=positions) as lookup, \
                    patch.object(center, "pointing_offset", return_value=(1, 2)) as offset, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = center.inspect_image(path, "test", {"lat": 0, "lon": 0, "elevation": 0}, correction_time=current)
        self.assertEqual(lookup.call_args.args[2], [exposure.jd, current.jd])
        self.assertAlmostEqual(offset.call_args.args[1].ra.deg, original.ra.deg + 0.01)
        self.assertEqual(offset.call_args.kwargs["target_time"], current)
        self.assertAlmostEqual(result["x"], 100, places=5)
        self.assertAlmostEqual(result["y"], 80, places=5)

    def test_live_capture_failure_is_retried_and_saved_images_keep_exposure_time(self):
        options = ["--query", "vesta", "--latitude", "0", "--longitude", "0", "--elevation", "0", "--no-plot"]
        now = Time("2025-01-01T00:10:00")
        with patch.object(sys, "argv", ["center.py", "--loop", *options]), \
                patch.object(center, "capture", side_effect=[
                    subprocess.CalledProcessError(1, "gphoto2"), Path("capture.fit"), KeyboardInterrupt(),
                ]) as capture, \
                patch.object(center, "solve_image", return_value=Path("solved.new")) as solve, \
                patch.object(center, "inspect_image") as inspect, \
                patch.object(center.Time, "now", return_value=now), \
                patch.object(center.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            center.main()
        self.assertEqual(capture.call_count, 3)
        inspect.assert_called_once()
        self.assertEqual(inspect.call_args.kwargs["correction_time"], now)
        self.assertFalse(solve.call_args.kwargs["keep_solution"])
        self.assertIn("retrying with the next capture", output.getvalue())

        with patch.object(sys, "argv", ["center.py", "saved.fit", *options]), \
                patch.object(center, "capture") as capture, \
                patch.object(center, "solve_image", return_value=Path("saved.new")) as solve, \
                patch.object(center, "inspect_image") as inspect:
            center.main()
        capture.assert_not_called()
        self.assertIsNone(inspect.call_args.kwargs["correction_time"])
        self.assertTrue(solve.call_args.kwargs["keep_solution"])

    def test_pan_wraps_across_north_and_tilt_has_correct_sign(self):
        time = Time("2025-01-01T00:00:00")
        observer = {"lat": 0, "lon": 0, "elevation": 0}
        frame = AltAz(obstime=time, location=EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg))
        with iers.conf.set_temp("auto_download", False):
            start = SkyCoord(az=359 * u.deg, alt=40 * u.deg, frame=frame).icrs
            target = SkyCoord(az=1 * u.deg, alt=43 * u.deg, frame=frame).icrs
        pan, tilt = center.pointing_offset(start, target, time, observer)
        self.assertAlmostEqual(pan, 2, places=5)
        self.assertAlmostEqual(tilt, 3, places=5)

    def test_outside_target_projection_on_a_rotated_frame(self):
        header = solved_header()
        target = WCS(header).pixel_to_world(350, 110)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.new"
            fits.writeto(path, np.zeros((200, 300)), header)
            with contextlib.redirect_stdout(io.StringIO()):
                result = center.inspect_image(
                    path, "test", {"lat": 0, "lon": 0, "elevation": 0},
                    coordinates=(target.ra.deg, target.dec.deg),
                )
        self.assertFalse(result["inside"])
        self.assertAlmostEqual(result["x"], 350, places=5)
        self.assertAlmostEqual(result["y"], 110, places=5)

    def test_solver_checks_success_marker_and_uses_repository_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "frame.fit"
            fits.writeto(path, np.zeros((50, 50)), fits.Header({"DATE-AVG": "2025-01-01T00:00:00"}))
            (directory / "solved.solved").touch()
            (directory / "solved.new").touch()
            with patch.object(locate.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "Plate solving failed"):
                    center.solve_image(path, directory, None, 3, 20)
            self.assertEqual(run.call_args.kwargs["cwd"], locate.ROOT)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--backend-config") + 1], str(locate.ROOT / "astrometry.cfg"))

    def test_solved_input_does_not_invoke_solver(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.new"
            fits.writeto(path, np.zeros((200, 300)), solved_header())
            with patch.object(locate.subprocess, "run") as run:
                result = center.solve_image(path, temporary, None, 3, 20)
            self.assertEqual(result, path)
            run.assert_not_called()

    def test_capture_loop_retries_after_an_unsolved_exposure(self):
        argv = [
            "center.py", "--query", "vesta", "--latitude", "0",
            "--longitude", "0", "--elevation", "0", "--loop", "--no-plot",
        ]
        with patch.object(sys, "argv", argv), \
                patch.object(center, "capture", return_value=Path("capture.fit")), \
                patch.object(center, "solve_image", side_effect=[
                    center.PlateSolveError("No solution"), Path("solved.new"), KeyboardInterrupt(),
                ]) as solve, \
                patch.object(center, "inspect_image") as inspect, \
                patch.object(center.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            center.main()
        self.assertEqual(solve.call_count, 3)
        inspect.assert_called_once()
        self.assertIn("retrying with the next capture", output.getvalue())

class CaptureTests(unittest.TestCase):
    def test_capture_configures_raw_and_returns_downloaded_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            def download(command, **kwargs):
                pattern = command[command.index("--filename") + 1]
                Path(pattern.replace("%C", "CR2")).touch()
            with patch.object(camera.subprocess, "run", side_effect=download) as run:
                path = camera.capture(directory, "1/4")
            self.assertEqual(path, directory / "capture.CR2")
            self.assertIn("imageformat=RAW", run.call_args.args[0])
            self.assertIn("shutterspeed=1/4", run.call_args.args[0])

    def test_conversion_can_keep_intermediate_fits_in_a_temporary_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "original.CR2"
            source.touch()
            output = directory / "scratch" / "capture.fit"
            output.parent.mkdir()
            tags = {"EXIF DateTimeOriginal": "2025:01:01 00:00:00", "EXIF ExposureTime": "1/4"}
            with patch.object(conversion.rawpy, "imread") as imread, \
                    patch.object(conversion.exifread, "process_file", return_value=tags), \
                    contextlib.redirect_stdout(io.StringIO()):
                raw = imread.return_value.__enter__.return_value
                raw.raw_image_visible = np.full((20, 20), 123, dtype=np.uint16)
                raw.black_level_per_channel = [100] * 4
                raw.white_level = 4095
                raw.color_desc = b"RGBG"
                raw.raw_colors_visible = np.tile([[0, 1], [3, 2]], (10, 10))
                raw.daylight_whitebalance = [2, 1, 1.5, 0]
                result = conversion.convert(source, ZoneInfo("UTC"), output=output)
            self.assertEqual(result, output)
            self.assertFalse(source.with_suffix(".fit").exists())
            data, header = fits.getdata(output, header=True)
            self.assertTrue((data == 123).all())
            self.assertEqual(header["RAWFILE"], "original.CR2")
            self.assertEqual(header["DATE-AVG"], "2025-01-01T00:00:00.125")
            self.assertEqual(header["BAYERPAT"], "RGGB")
            self.assertEqual(header["WBR"], 2)
            self.assertEqual(header["WBG"], 1)
            self.assertEqual(header["WBB"], 1.5)

if __name__ == "__main__":
    unittest.main()
