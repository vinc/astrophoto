import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from astrophoto import center, focus, locate
from astrophoto.lib import camera, config

OBSERVER = ["--latitude", "0", "--longitude", "0", "--elevation", "0"]


class ConfiguredTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.dict(os.environ, {"XDG_CONFIG_HOME": directory.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.path = Path(directory.name) / "astrophoto" / "config.toml"
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


if __name__ == "__main__":
    unittest.main()
