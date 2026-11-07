import contextlib
import io
from pathlib import Path
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

import camera
import center
import convert as conversion
import focus

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
            with patch.object(center.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "Plate solving failed"):
                    center.solve_image(path, directory, None, 3, 20)
            self.assertEqual(run.call_args.kwargs["cwd"], center.ROOT)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--backend-config") + 1], str(center.ROOT / "astrometry.cfg"))

    def test_solved_input_does_not_invoke_solver(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.new"
            fits.writeto(path, np.zeros((200, 300)), solved_header())
            with patch.object(center.subprocess, "run") as run:
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
                result = conversion.convert(source, ZoneInfo("UTC"), output=output)
            self.assertEqual(result, output)
            self.assertFalse(source.with_suffix(".fit").exists())
            data, header = fits.getdata(output, header=True)
            self.assertTrue((data == 123).all())
            self.assertEqual(header["RAWFILE"], "original.CR2")
            self.assertEqual(header["DATE-AVG"], "2025-01-01T00:00:00.125")

if __name__ == "__main__":
    unittest.main()
