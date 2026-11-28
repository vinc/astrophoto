import argparse
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from astro import capture as capture_command, center, focus, locate
from astro.lib import camera, config, gphoto

OBSERVER = ["--latitude", "0", "--longitude", "0", "--elevation", "0"]


class ConfiguredTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.dict(os.environ, {"XDG_CONFIG_HOME": directory.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.path = Path(directory.name) / "astro" / "config.toml"
        self.path.parent.mkdir(parents=True)

    def run_helper(self, helper, *args):
        parser = argparse.ArgumentParser()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                return helper(parser, *args)
            except SystemExit:
                raise AssertionError(err.getvalue())


class CameraConfigTests(ConfiguredTests):
    def test_camera_is_a_top_level_name_of_a_known_profile(self):
        self.assertIsNone(self.run_helper(config.camera_profile))
        self.path.write_text('camera = "Canon EOS 600D"\n')
        self.assertEqual(self.run_helper(config.camera_profile).pixel_size, 4.3)
        for text, message in [
            ('camera = "Canon EOS 5D"', "Unknown camera Canon EOS 5D .* Canon EOS 20D, Canon EOS 600D"),
            ("camera = 600", "camera in .* must be a string"),
        ]:
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaisesRegex(AssertionError, message):
                    self.run_helper(config.camera_profile)

    def test_command_tables_layer_over_defaults(self):
        defaults = {"shutterspeed": "1/4", "iso": None}
        self.assertEqual(self.run_helper(config.command_settings, "focus", defaults), defaults)
        self.path.write_text("[focus]\niso = 3200\n")
        self.assertEqual(self.run_helper(config.command_settings, "focus", defaults), {"shutterspeed": "1/4", "iso": "3200"})
        for text, message in [
            ("focus = 1", r"\[focus\] in .* must be a table"),
            ("[focus]\naperture = 4", r"Unknown setting aperture in \[focus\]"),
            ("[focus]\niso = true", r"iso in \[focus\] of .* must be a number or a string"),
        ]:
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaisesRegex(AssertionError, message):
                    self.run_helper(config.command_settings, "focus", defaults)


class ScaleTests(unittest.TestCase):
    def test_single_raw_frames_get_a_tight_pixel_scale(self):
        eos600d = camera.CAMERAS["Canon EOS 600D"]
        header = fits.Header({"RAWFILE": "IMG_0001.CR2", "FOCALLEN": 300.0})
        options = locate.scale_options(header, 3, 20, eos600d)
        self.assertEqual(options[:2], ["--scale-units", "arcsecperpix"])
        # 206.265 * 4.3 / 300 is 2.957 arcseconds per pixel
        self.assertEqual(options[3], "2.513")
        self.assertEqual(options[5], "3.400")

    def test_recorded_camera_wins_over_the_configured_one(self):
        header = fits.Header({"RAWFILE": "IMG_0001.CR2", "FOCALLEN": 300.0, "INSTRUME": "Canon EOS 20D"})
        options = locate.scale_options(header, 3, 20, camera.CAMERAS["Canon EOS 600D"])
        self.assertEqual(options[3], f"{0.85 * 206.265 * 6.4 / 300:.3f}")

    def test_stacks_use_their_binned_pixels(self):
        header = fits.Header({"INSTRUME": "Canon EOS 600D", "FOCALLEN": 28.0, "NCOMBINE": 24, "XBINNING": 2})
        options = locate.scale_options(header, 3, 20)
        # A 2x2-binned 600D frame at 28 mm has 63 arcseconds per pixel
        self.assertEqual(options[:2], ["--scale-units", "arcsecperpix"])
        self.assertAlmostEqual(float(options[3]) / 0.85, 206.265 * 4.3 * 2 / 28, places=2)

    def test_unknown_cameras_use_the_recorded_pixel_size(self):
        header = fits.Header({"INSTRUME": "Canon EOS 1100D", "FOCALLEN": 50.0, "XPIXSZ": 10.4, "XBINNING": 2})
        options = locate.scale_options(header, 3, 20)
        self.assertAlmostEqual(float(options[5]) / 1.15, 206.265 * 10.4 / 50, places=2)

    def test_stack_header_records_its_binning(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.fit"
            fits.writeto(path, np.zeros((4, 4)), fits.Header({"RAWFILE": "IMG_1.CR2", "XPIXSZ": 4.3, "XBINNING": 1}))
            from astro import stack
            header = stack.stack_header(path, [path])
        self.assertEqual((header["XBINNING"], header["XPIXSZ"]), (2, 8.6))
        self.assertNotIn("RAWFILE", header)

    def test_pixel_size_is_read_from_the_exif_focal_plane_resolution(self):
        from astro import convert

        class Tag:
            def __init__(self, text, values):
                self.text, self.values = text, values

            def __str__(self):
                return self.text

        # 5184 pixels across a 22.3 mm wide sensor, in pixels per inch
        tags = {"EXIF FocalPlaneXResolution": Tag("5904.5", None), "EXIF FocalPlaneResolutionUnit": Tag("Pixels/Inch", [2])}
        self.assertAlmostEqual(convert.exif_pixel_size(tags), 4.3, places=2)
        tags["EXIF FocalPlaneResolutionUnit"] = Tag("3", [3])
        tags["EXIF FocalPlaneXResolution"] = Tag("2325", None)
        self.assertAlmostEqual(convert.exif_pixel_size(tags), 4.3, places=2)
        self.assertIsNone(convert.exif_pixel_size({}))
        # A resolution taken from the JPEG size rather than the sensor gives an implausible pixel
        tags["EXIF FocalPlaneXResolution"] = Tag("72", None)
        self.assertIsNone(convert.exif_pixel_size(tags))

    def test_stacks_unknown_cameras_and_missing_focal_lengths_keep_the_width_range(self):
        eos600d = camera.CAMERAS["Canon EOS 600D"]
        width = ["--scale-units", "degwidth", "--scale-low", "3", "--scale-high", "20"]
        for header, profile in [
            (fits.Header({"FOCALLEN": 300.0, "NCOMBINE": 20}), eos600d),
            (fits.Header({"RAWFILE": "IMG_0001.CR2", "FOCALLEN": 300.0}), None),
            (fits.Header({"RAWFILE": "IMG_0001.CR2"}), eos600d),
            (fits.Header({"RAWFILE": "IMG_0001.CR2", "FOCALLEN": 0.0}), eos600d),
        ]:
            with self.subTest(header=dict(header)):
                self.assertEqual(locate.scale_options(header, 3, 20, profile), width)


class CenterTests(ConfiguredTests):
    def test_loop_searches_near_the_last_solution_until_a_solve_fails(self):
        self.path.write_text('camera = "Canon EOS 600D"\n[center]\nshutterspeed = 2\niso = 3200\n')
        argv = ["center.py", "--query", "vesta", *OBSERVER, "--loop", "--no-plot"]
        results = [{"center": (10.0, 20.0)}, {"center": (30.0, 40.0)}]
        with patch.object(sys, "argv", argv), patch.object(center, "resolve", return_value=None), \
                patch.object(center, "capture", return_value=Path("capture.CR2")) as capture, \
                patch.object(center, "solve_image", side_effect=[
                    Path("a.new"), center.PlateSolveError("No solution"), Path("b.new"), Path("c.new"), KeyboardInterrupt(),
                ]) as solve, \
                patch.object(center, "inspect_image", side_effect=results * 2), \
                patch.object(center.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            center.main()
        hints = [call.kwargs["near"] for call in solve.call_args_list]
        self.assertEqual(hints, [None, (10.0, 20.0, 10), None, (30.0, 40.0, 10), (10.0, 20.0, 10)])
        self.assertEqual(solve.call_args.kwargs["camera"].name, "Canon EOS 600D")
        capture.assert_called_with(unittest.mock.ANY, "2", "3200")

    def test_options_override_the_center_table(self):
        self.path.write_text("[center]\nshutterspeed = 2\n")
        argv = ["center.py", "--query", "vesta", *OBSERVER, "--no-plot", "--shutterspeed", "1", "--iso", "6400"]
        with patch.object(sys, "argv", argv), patch.object(center, "resolve", return_value=None), \
                patch.object(center, "capture", return_value=Path("capture.CR2")) as capture, \
                patch.object(center, "solve_image", return_value=Path("a.new")) as solve, \
                patch.object(center, "inspect_image", return_value={"center": (1.0, 2.0)}):
            center.main()
        capture.assert_called_once_with(unittest.mock.ANY, "1", "6400")
        self.assertIsNone(solve.call_args.kwargs["camera"])


class FocusTests(ConfiguredTests):
    def test_radius_scales_with_pixel_size(self):
        self.assertEqual(focus.default_radius(None), 12)
        self.assertEqual(focus.default_radius(camera.CAMERAS["Canon EOS 20D"]), 12)
        self.assertEqual(focus.default_radius(camera.CAMERAS["Canon EOS 600D"]), 18)

    def test_hfr_is_given_in_arcseconds_when_the_optics_are_known(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.fit"
            fits.writeto(path, np.zeros((4, 4)), fits.Header({"INSTRUME": "Canon EOS 600D", "FOCALLEN": 300.0}))
            self.assertEqual(focus.hfr_text(2.0, path, None), 'HFR 2.00 px (5.9")')
            fits.writeto(path, np.zeros((4, 4)), fits.Header({"FOCALLEN": 300.0}), overwrite=True)
            self.assertEqual(focus.hfr_text(2.0, path, None), "HFR 2.00 px")
            self.assertEqual(focus.hfr_text(2.0, path, camera.CAMERAS["Canon EOS 20D"]), 'HFR 2.00 px (8.8")')

    def test_focus_table_sets_captures_and_radius_follows_the_camera(self):
        self.path.write_text('camera = "Canon EOS 600D"\n[focus]\niso = 6400\n')
        with patch.object(sys, "argv", ["focus.py", "--frames", "1", "--no-plot"]), \
                patch.object(focus, "capture", return_value=Path("capture.CR2")) as capture, \
                patch.object(focus, "read_image", return_value=(np.zeros((64, 64)), None, 2)), \
                patch.object(focus, "measure_stars", return_value=[]) as measure, \
                contextlib.redirect_stdout(io.StringIO()):
            focus.main()
        capture.assert_called_once_with(unittest.mock.ANY, "1/4", "6400")
        self.assertEqual(measure.call_args.kwargs["radius"], 18)


class CaptureTests(unittest.TestCase):
    def test_iso_is_only_set_when_given(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def download(command, **kwargs):
                Path(command[command.index("--filename") + 1].replace("%C", "CR2")).touch()

            with patch.object(camera.subprocess, "run", side_effect=download) as run:
                camera.capture(directory, "1/4", "3200")
            self.assertIn("iso=3200", run.call_args.args[0])
            (directory / "capture.CR2").unlink()
            with patch.object(camera.subprocess, "run", side_effect=download) as run:
                camera.capture(directory)
            self.assertFalse(any(arg.startswith("iso=") for arg in run.call_args.args[0]))


class FakeCamera:
    """Stand in for gphoto2: track the lens position and save live-view frames blurred by defocus."""

    # The EOS 600D custom functions as gphoto2 shows them, with C.Fn IV-1 (701) left to fill in
    FUNCTIONS = "c8,4,1,2c,3,101,1,0,103,1,0,10f,1,0,2,2c,3,201,1,0,202,1,0,203,1,0,3,20,2,50e,1,0,60f,1,0,4,38,4,701,1,{:x},704,1,0,811,1,0,80f,1,0,"

    def __init__(self, best=3, backlash=0, model="Canon EOS 600D", shutter=0):
        self.position, self.best, self.backlash = 0, best, backlash
        self.model, self.functions = model, self.FUNCTIONS.format(shutter)
        self.direction = 0
        self.commands = []
        rng = np.random.default_rng(1)
        self.stars = rng.uniform(40, 360, size=(25, 2))

    def frame(self):
        sigma = 1.2 + 0.8 * abs(self.position - self.best)
        yy, xx = np.indices((400, 600), dtype=float)
        image = np.full(xx.shape, 20.0)
        for x, y in self.stars:
            x += 100
            image += 200 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)) * (1.2 / sigma) ** 2
        image += np.random.default_rng(len(self.commands)).normal(0, 1, image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)

    def __call__(self, command, cwd=None, **kwargs):
        self.commands.append(command)
        output = ""
        for name in [command[i + 1] for i, option in enumerate(command) if option == "--get-config"]:
            output += f"Label: {name}\nCurrent: {self.model if name == 'cameramodel' else self.functions}\nEND\n"
        for option in command:
            if option.startswith("customfuncex="):
                self.functions = option.split("=", 1)[1]
            if option.startswith("manualfocusdrive="):
                direction, size = option.split("=")[1].split()
                sign = 1 if direction == "Far" else -1
                # A reversal first takes up the slack in the focus mechanism without moving the focus
                if sign != self.direction and self.direction:
                    slack = min(self.backlash, int(size))
                    self.position += sign * (int(size) - slack)
                else:
                    self.position += sign * int(size)
                self.direction = sign
        if "--capture-preview" in command:
            name = command[command.index("--filename") + 1]
            import matplotlib.image
            matplotlib.image.imsave(Path(cwd) / f"thumb_{name}", self.frame(), cmap="gray", vmin=0, vmax=255)
        return subprocess.CompletedProcess(command, 0, output, "")


class AutofocusTests(ConfiguredTests):
    def test_v_curve_is_fitted_from_the_near_side_and_the_lens_left_at_its_minimum(self):
        for best, step in [(3, 1), (-2, 2), (0, 1)]:
            with self.subTest(best=best, step=step):
                fake = FakeCamera(best=best * step)
                argv = ["focus.py", "--auto", "--no-plot", "--step", str(step), "--span", "6"]
                with patch.object(sys, "argv", argv), patch.object(gphoto.subprocess, "run", side_effect=fake), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    focus.main()
                self.assertEqual(fake.position, best * step)
                self.assertIn(f"Focused at step {best:+d}", output.getvalue())
                # The lens arrives at the minimum moving toward infinity, as it did for every measurement
                self.assertEqual(fake.direction, 1)
                drives = {option for command in fake.commands for option in command if option.startswith("manualfocusdrive=")}
                self.assertEqual(drives, {f"manualfocusdrive=Near {step}", f"manualfocusdrive=Far {step}"})

    def test_backlash_is_taken_up_before_measuring(self):
        fake = FakeCamera(best=2, backlash=1)
        argv = ["focus.py", "--auto", "--no-plot", "--step", "1", "--span", "6"]
        with patch.object(sys, "argv", argv), patch.object(gphoto.subprocess, "run", side_effect=fake), \
                contextlib.redirect_stdout(io.StringIO()):
            focus.main()
        self.assertEqual(fake.position, 2)

    def test_a_minimum_outside_the_scan_returns_the_lens_and_stops(self):
        fake = FakeCamera(best=20)
        argv = ["focus.py", "--auto", "--no-plot", "--step", "1", "--span", "4"]
        with patch.object(sys, "argv", argv), patch.object(gphoto.subprocess, "run", side_effect=fake), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "No focus minimum"):
            focus.main()
        self.assertEqual(fake.position, 0)

    def test_fit_finds_a_minimum_between_steps_and_rejects_edges(self):
        positions = list(range(-4, 5))
        hfrs = [abs(p - 0.4) + 1 for p in positions]
        self.assertAlmostEqual(focus.fit_minimum(positions, hfrs), 0.4, delta=0.5)
        self.assertIsNone(focus.fit_minimum(positions, [5 - p for p in positions]))
        self.assertIsNone(focus.fit_minimum([0, 1], [2, 1]))
        # Frames without stars are skipped rather than ending the curve
        self.assertAlmostEqual(focus.fit_minimum([0, 1, 2, 3, 4], [3, None, 1, None, 3]), 2, delta=0.1)

    def test_stray_readings_do_not_move_the_fitted_minimum(self):
        positions = list(range(-8, 9))
        hfrs = [float(np.hypot(1.5, 0.6 * (p + 1))) for p in positions]
        # Noise taken for a sharp star, including at the edge of the scan
        hfrs[12], hfrs[16] = 1.0, 0.9
        self.assertAlmostEqual(focus.fit_minimum(positions, hfrs), -1, delta=0.5)

    def test_a_noisy_night_scan_still_finds_the_dip(self):
        # Readings from a real scan with one to three stars per frame and two stray readings
        hfrs = [5.74, 5.29, 5.01, 5.82, None, 5.83, 3.06, 2.12, 3.63, 4.10, 3.94, 4.53, 1.65, 4.85, 4.85, 5.30, 1.32]
        self.assertTrue(-2 <= focus.fit_minimum(list(range(-8, 9)), hfrs) <= 1)

    def test_each_step_is_the_median_of_several_frames(self):
        fake = FakeCamera(best=1)
        argv = ["focus.py", "--auto", "--no-plot", "--step", "1", "--span", "3", "--samples", "3"]
        with patch.object(sys, "argv", argv), patch.object(gphoto.subprocess, "run", side_effect=fake), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            focus.main()
        previews = [command for command in fake.commands if "--capture-preview" in command]
        # Seven scanned positions and the final check, three frames each
        self.assertEqual(len(previews), 8 * 3)
        # Only the first frame at each position moves the lens
        moving = [command for command in previews if any(o.startswith("manualfocusdrive=") for o in command)]
        self.assertEqual(len(moving), 6)
        self.assertEqual(fake.position, 1)
        self.assertIn("3 of 3 frames", output.getvalue())

    def test_live_mode_measures_zoomed_previews(self):
        fake = FakeCamera(best=0)
        argv = ["focus.py", "--live", "--no-plot", "--frames", "2", "--interval", "0"]
        with patch.object(sys, "argv", argv), patch.object(gphoto.subprocess, "run", side_effect=fake), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            focus.main()
        self.assertEqual(len(fake.commands), 2)
        self.assertIn("eoszoom=5", fake.commands[0])
        self.assertNotIn("manualfocusdrive", " ".join(fake.commands[0]))
        readings = [line for line in output.getvalue().splitlines() if line.startswith("thumb_preview.jpg:")]
        self.assertEqual(len(readings), 2)
        self.assertRegex(readings[0], r"HFR \d+\.\d+ px \| \d+ stars")
        self.assertNotIn('"', readings[0])

    def test_live_and_auto_reject_saved_files(self):
        for option in ["--live", "--auto"]:
            with self.subTest(option=option), patch.object(sys, "argv", ["focus.py", option, "saved.fit"]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                focus.main()
            self.assertEqual(raised.exception.code, 2)


class BackButtonFocusTests(ConfiguredTests):
    def run_auto(self, fake):
        argv = ["focus.py", "--auto", "--no-plot", "--step", "1", "--span", "4"]
        with patch.object(sys, "argv", argv), patch.object(gphoto.subprocess, "run", side_effect=fake), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            focus.main()
        return output.getvalue()

    def test_auto_moves_autofocus_off_the_shutter_button(self):
        for shutter, expected in [(0, 1), (2, 3)]:
            with self.subTest(shutter=shutter):
                fake = FakeCamera(best=1, shutter=shutter)
                output = self.run_auto(fake)
                self.assertEqual(fake.functions, FakeCamera.FUNCTIONS.format(expected))
                self.assertIn("Moved autofocus off the shutter button of the Canon EOS 600D", output)

    def test_settings_already_using_the_rear_button_or_unknown_cameras_are_left_alone(self):
        for model, shutter in [("Canon EOS 600D", 1), ("Canon EOS 600D", 3), ("Canon EOS 5D Mark II", 0)]:
            with self.subTest(model=model, shutter=shutter):
                fake = FakeCamera(best=1, model=model, shutter=shutter)
                output = self.run_auto(fake)
                self.assertEqual(fake.functions, FakeCamera.FUNCTIONS.format(shutter))
                self.assertFalse(any(o.startswith("customfuncex=") for c in fake.commands for o in c))
                self.assertNotIn("Moved autofocus", output)

    def test_a_failed_check_does_not_stop_focusing(self):
        fake = FakeCamera(best=1)
        original = fake.__call__

        def refuse(command, **kwargs):
            if "--get-config" in command:
                raise subprocess.CalledProcessError(1, command, stderr="")
            return original(command, **kwargs)

        with contextlib.redirect_stderr(io.StringIO()):
            output = self.run_auto(refuse)
        self.assertIn("Could not check whether the shutter button autofocuses", output)
        self.assertEqual(fake.position, 1)

    def test_custom_functions_are_parsed_by_structure(self):
        text = FakeCamera.FUNCTIONS.format(0)
        self.assertEqual(gphoto.custom_function(text, 0x701), 0)
        self.assertIsNone(gphoto.custom_function(text, 0x999))
        self.assertEqual(gphoto.set_custom_function(text, 0x701, 1), FakeCamera.FUNCTIONS.format(1))
        with self.assertRaises(ValueError):
            gphoto.set_custom_function(text, 0x999, 1)


class PreviewWindowTests(unittest.TestCase):
    def test_an_unreachable_display_suggests_running_without_a_window(self):
        class TclError(Exception):
            """Stands in for tkinter.TclError, which a Python without Tk cannot import."""

        error = TclError('couldn\'t connect to display "localhost:10.0"')
        with patch.object(camera.plt, "figure", side_effect=error), \
                self.assertRaisesRegex(SystemExit, r'(?s)couldn\'t connect to display.*--no-plot'):
            camera.preview_figure()


class ReviewFixTests(ConfiguredTests):
    def test_gphoto2_always_runs_untranslated(self):
        result = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(gphoto.subprocess, "run", return_value=result) as run:
            gphoto.run_gphoto2(["gphoto2", "--set-config", "manualfocusdrive=Near 1"])
        self.assertEqual(run.call_args.kwargs["env"]["LC_ALL"], "C")
        self.assertEqual(run.call_args.kwargs["env"]["LANGUAGE"], "C")
        with patch.object(capture_command.subprocess, "run", return_value=result) as run:
            capture_command.files_on_card()
        self.assertEqual(run.call_args.kwargs["env"]["LC_ALL"], "C")

    def test_a_wrong_metadata_scale_falls_back_to_the_configured_width(self):
        from tests.test_solver import solve
        calls = []

        def solver(command, **kwargs):
            calls.append(command[command.index("--scale-units") + 1])
            # The first, metadata-based attempt finds nothing
            return None if len(calls) == 1 else solve(command, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "frame.fit"
            header = fits.Header({"DATE-AVG": "2025-01-01T00:00:00", "RAWFILE": "IMG_1.CR2",
                                  "INSTRUME": "Canon EOS 600D", "FOCALLEN": 300.0})
            fits.writeto(source, np.zeros((20, 30)), header)
            with patch.object(locate.subprocess, "run", side_effect=solver):
                locate.solve_image(source, temporary, None, 3, 50)
        self.assertEqual(calls, ["arcsecperpix", "degwidth"])

    def test_a_solver_error_with_the_metadata_scale_still_tries_the_configured_width(self):
        from tests.test_solver import solve
        calls = []

        def solver(command, **kwargs):
            calls.append(command[command.index("--scale-units") + 1])
            # A hinted scale between the installed indexes makes the solver abort
            if len(calls) == 1:
                raise subprocess.CalledProcessError(255, command)
            return solve(command, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "frame.fit"
            header = fits.Header({"DATE-AVG": "2025-01-01T00:00:00", "RAWFILE": "IMG_1.CR2",
                                  "INSTRUME": "Canon EOS 600D", "FOCALLEN": 300.0})
            fits.writeto(source, np.zeros((20, 30)), header)
            with patch.object(locate.subprocess, "run", side_effect=solver):
                locate.solve_image(source, temporary, None, 3, 50)
            self.assertEqual(calls, ["arcsecperpix", "degwidth"])
            # When the fallback fails as well, the solver's error is reported
            calls.clear()
            with patch.object(locate.subprocess, "run", side_effect=subprocess.CalledProcessError(255, "solve-field")), \
                    self.assertRaisesRegex(locate.PlateSolveError, "exit status 255"):
                locate.solve_image(source, temporary, None, 3, 50)

    def test_unusable_fits_binning_falls_back_for_the_solver(self):
        for extra, units in [({}, "degwidth"), ({"XPIXSZ": 8.6}, "arcsecperpix")]:
            header = fits.Header({"INSTRUME": "Canon EOS 600D", "FOCALLEN": 300.0, "RAWFILE": "IMG_1.CR2",
                                  "XBINNING": "unknown", **extra})
            with self.subTest(extra=extra):
                options = locate.scale_options(header, 3, 50)
                self.assertEqual(options[1], units)
                if extra:
                    self.assertAlmostEqual(float(options[3]) / 0.85, 206.265 * 8.6 / 300, places=2)

    def test_unusable_fits_binning_omits_the_arcseconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stack.fit"
            for binning in ["2", 0, -2, True]:
                header = fits.Header({"INSTRUME": "Canon EOS 600D", "FOCALLEN": 300.0, "NCOMBINE": 24, "XBINNING": binning})
                fits.writeto(path, np.zeros((4, 4)), header, overwrite=True)
                with self.subTest(binning=binning):
                    self.assertEqual(focus.hfr_text(2.36, path, None), "HFR 2.36 px")

    def test_autofocus_fails_when_the_final_position_cannot_be_measured(self):
        class Clouded(FakeCamera):
            """Shows stars during the scan, then only empty sky."""

            def frame(self):
                previews = sum("--capture-preview" in command for command in self.commands)
                return super().frame() if previews <= 9 else np.full((400, 600), 20, dtype=np.uint8)

        fake = Clouded(best=1)
        argv = ["focus.py", "--auto", "--no-plot", "--step", "1", "--span", "4", "--samples", "1"]
        with patch.object(sys, "argv", argv), patch.object(gphoto.subprocess, "run", side_effect=fake), \
                contextlib.redirect_stdout(io.StringIO()) as output, \
                self.assertRaisesRegex(SystemExit, "could not measure the focus there"):
            focus.main()
        self.assertNotIn("Focused at", output.getvalue())

    def test_stack_hfr_in_arcseconds_uses_the_binned_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stack.fit"
            for extra in [{"NCOMBINE": 24}, {"NCOMBINE": 24, "XBINNING": 2}]:
                fits.writeto(path, np.zeros((4, 4)), fits.Header({"INSTRUME": "Canon EOS 600D", "FOCALLEN": 300.0, **extra}),
                             overwrite=True)
                # 2.36 binned pixels of 8.6 micrometers at 300 mm
                self.assertEqual(focus.hfr_text(2.36, path, None), 'HFR 2.36 px (14.0")')

    def test_normalized_fits_keeps_its_stars(self):
        from tests.test_helpers import star_field
        image = (star_field(1.5) - 100) / 1000
        self.assertEqual(focus.noise_floor(Path("frame.fit")), 0.0)
        self.assertEqual(focus.noise_floor(Path("thumb_preview.jpg")), 0.5)
        self.assertGreater(len(focus.measure_stars(image)), 0)
        self.assertEqual(focus.measure_stars(image, min_noise=0.5), [])

    def test_live_view_is_refused_on_a_camera_without_it(self):
        self.path.write_text('camera = "Canon EOS 20D"\n')
        for option in ["--live", "--auto"]:
            with self.subTest(option=option), patch.object(sys, "argv", ["focus.py", option, "--no-plot"]), \
                    patch.object(gphoto.subprocess, "run") as run, \
                    contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as raised:
                focus.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Canon EOS 20D does not have", err.getvalue())
            run.assert_not_called()


class NoFocusTests(unittest.TestCase):
    def test_failed_autofocus_explains_the_fix(self):
        error = subprocess.CalledProcessError(1, "gphoto2", stderr="Canon EOS Capture failed to release: Perhaps no focus?\n")
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(gphoto.subprocess, "run", side_effect=error), \
                contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(subprocess.CalledProcessError):
            camera.capture(temporary)
        self.assertIn("Perhaps no focus?", err.getvalue())
        self.assertIn("C.Fn IV-1", err.getvalue())

    def test_capture_command_explains_the_fix_too(self):
        def fake(command, **kwargs):
            if "--capture-image" in command:
                raise subprocess.CalledProcessError(1, command, stderr="Perhaps no focus?\n")
            return subprocess.CompletedProcess(command, 0, "", "")

        settings = {"frames": 2, "iso": "800", "aperture": "4.0", "shutterspeed": "5"}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(capture_command.subprocess, "run", side_effect=fake), \
                contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(subprocess.CalledProcessError):
            capture_command.capture_series(Path(temporary) / "light", settings, log=lambda text: None)
        self.assertIn("C.Fn IV-1", err.getvalue())


class LibraryTests(unittest.TestCase):
    """The commands' building blocks report through return values and exceptions, without exiting."""

    def test_gphoto2_calls_hold_the_camera_lock(self):
        import threading
        held = []

        def fake(command, **kwargs):
            # Another thread cannot take the camera while gphoto2 runs
            result = []
            thread = threading.Thread(target=lambda: result.append(gphoto.camera_lock.acquire(blocking=False)))
            thread.start()
            thread.join()
            held.append(not result[0])
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(gphoto.subprocess, "run", side_effect=fake):
            gphoto.run_gphoto2(["gphoto2", "--summary"])
        self.assertEqual(held, [True])

    def test_capture_series_refuses_a_card_with_files(self):
        listing = subprocess.CompletedProcess([], 0, "/DCIM/100CANON/IMG_0001.CR2\n", "")
        settings = {"frames": 2, "iso": "800", "aperture": "4.0", "shutterspeed": "5"}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(gphoto.subprocess, "run", return_value=listing) as run, \
                self.assertRaisesRegex(capture_command.CaptureError, "already holds 1 files"):
            capture_command.capture_series(Path(temporary) / "light", settings, log=lambda text: None)
        run.assert_called_once()

    def test_a_ram_series_reports_each_frame_and_stops_between_frames(self):
        import threading
        calls = []

        def fake(command, **kwargs):
            calls.append(kwargs)
            pattern = command[command.index("--filename") + 1]
            Path(pattern.replace("%Y%m%d-%H%M%S", "20261127-210000").replace("%C", "cr2")).write_text("raw")
            # Meanwhile the server converts the previous frame into the series directory
            (Path(temporary) / "light" / f"converted{len(calls)}.fit").write_text("fits")
            return subprocess.CompletedProcess(command, 0, "", "")

        stop = threading.Event()
        frames = []

        def on_frame(path):
            frames.append(path.name)
            if len(frames) == 2:
                stop.set()

        settings = {"frames": 5, "iso": "800", "aperture": "4.0", "shutterspeed": "5", "target": "ram"}
        with tempfile.TemporaryDirectory() as temporary, patch.object(gphoto.subprocess, "run", side_effect=fake):
            paths = capture_command.capture_series(Path(temporary) / "light", settings, log=lambda text: None,
                                                   stop=stop, on_frame=on_frame)
            left = sorted(path.name for path in (Path(temporary) / "light").iterdir())
        self.assertEqual(frames, ["20261127-210000-001.CR2", "20261127-210000-002.CR2"])
        self.assertEqual([path.name for path in paths], frames)
        self.assertTrue(all(call["start_new_session"] for call in calls))
        # The staging directory is gone, and only the frames and the converted files are left
        self.assertEqual(left, [*frames, "converted1.fit", "converted2.fit"])

    def test_a_frame_saved_before_gphoto2_fails_is_kept(self):
        def fake(command, **kwargs):
            pattern = command[command.index("--filename") + 1]
            Path(pattern.replace("%Y%m%d-%H%M%S", "20261127-210000").replace("%C", "cr2")).write_text("raw")
            # The frame is downloaded, then deleting it from the camera's memory fails
            raise subprocess.CalledProcessError(1, command, stderr="*** Error: Could not delete file ***\n")

        logged = []
        settings = {"frames": 3, "iso": "800", "aperture": "4.0", "shutterspeed": "5", "target": "ram"}
        with tempfile.TemporaryDirectory() as temporary, patch.object(gphoto.subprocess, "run", side_effect=fake), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                capture_command.capture_series(Path(temporary) / "light", settings, log=logged.append)
            left = sorted(path.name for path in (Path(temporary) / "light").iterdir())
        self.assertEqual(left, ["20261127-210000-001.CR2"])
        self.assertIn("Kept 20261127-210000-001.CR2 from a frame that did not finish", logged)

    def test_series_directory_names_the_start_time_and_kind(self):
        from datetime import datetime
        self.assertEqual(capture_command.series_directory("dark", datetime(2026, 11, 27, 21, 5)),
                         Path("20261127-2105-dark"))

    def test_autofocus_reports_through_callbacks_and_raises_without_a_minimum(self):
        fake = FakeCamera(best=20)
        shown, logged = [], []
        with patch.object(gphoto.subprocess, "run", side_effect=fake), \
                self.assertRaisesRegex(focus.AutofocusError, "No focus minimum"):
            focus.autofocus(4, 1, 5, 12, samples=1, show=lambda *args: shown.append(args), log=logged.append)
        self.assertEqual(len(shown), 9)
        self.assertEqual([position for position, _ in shown[-1][3]], list(range(-4, 5)))
        self.assertTrue(any(text.startswith("Step +4:") for text in logged))
        self.assertEqual(fake.position, 0)

    def test_measure_frame_returns_stars_and_hfr(self):
        from tests.test_helpers import star_field
        with patch.object(focus, "read_image", return_value=(star_field(1.5), None, 2)):
            data, stars, hfr, problem = focus.measure_frame(Path("frame.fit"), 12)
        self.assertGreater(len(stars), 0)
        self.assertGreater(hfr, 0)
        self.assertIsNone(problem)

    def test_resolve_target_from_coordinates_names_or_bodies(self):
        self.assertEqual(center.resolve_target(ra=10.5, dec=-2), center.Target("RA 10.5, Dec -2", (10.5, -2)))
        with patch.object(center, "resolve", return_value=None):
            self.assertEqual(center.resolve_target("vesta"), center.Target("vesta"))
        from types import SimpleNamespace
        fixed = SimpleNamespace(label="M31", ra=10.68, dec=41.27, size=3.2)
        with patch.object(center, "resolve", return_value=fixed):
            self.assertEqual(center.resolve_target("andromeda"), center.Target("M31", (10.68, 41.27), 3.2))

    def test_point_returns_the_solved_image_and_correction(self):
        target = center.Target("M31", (10.68, 41.27), 3.2)
        settings = {"shutterspeed": "2", "iso": "3200"}
        with patch.object(center, "capture", return_value=Path("capture.CR2")) as capture, \
                patch.object(center, "solve_image", return_value=Path("capture.new")) as solve, \
                patch.object(center, "inspect_image", return_value={"center": (1.0, 2.0)}) as inspect:
            solved, result = center.point("work", target, {}, settings, None, (3, 20), near=(1, 2, 10))
        capture.assert_called_once_with("work", "2", "3200")
        self.assertEqual(solve.call_args.kwargs["near"], (1, 2, 10))
        self.assertFalse(solve.call_args.kwargs["keep_solution"])
        self.assertEqual(inspect.call_args.args[3], (10.68, 41.27))
        self.assertEqual(inspect.call_args.kwargs["size"], 3.2)
        self.assertIsNotNone(inspect.call_args.kwargs["correction_time"])
        self.assertEqual((solved, result), (Path("capture.new"), {"center": (1.0, 2.0)}))


if __name__ == "__main__":
    unittest.main()
