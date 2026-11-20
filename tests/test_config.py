import argparse
import contextlib
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from astrophoto import config as config_command
from astrophoto.lib import config

COMMANDS = ["center", "chart", "crop", "locate", "visible", "weather"]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.dict(os.environ, {"XDG_CONFIG_HOME": directory.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.path = Path(directory.name) / "astrophoto" / "config.toml"

    def write(self, text):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text)

    def parse(self, *arguments, required=True):
        parser = argparse.ArgumentParser()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                config.add_observer_arguments(parser, required=required)
                return parser.parse_args(arguments)
            except SystemExit:
                raise AssertionError(err.getvalue())

    def test_path_follows_xdg_config_home(self):
        self.assertEqual(config.config_path(), self.path)
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": ""}):
            self.assertEqual(config.config_path(), Path.home() / ".config" / "astrophoto" / "config.toml")

    def test_data_path_follows_xdg_data_home(self):
        with patch.dict(os.environ, {"XDG_DATA_HOME": "/srv/data"}):
            self.assertEqual(config.data_path(), Path("/srv/data/astrophoto"))
        with patch.dict(os.environ, {"XDG_DATA_HOME": ""}):
            self.assertEqual(config.data_path(), Path.home() / ".local" / "share" / "astrophoto")

    def test_missing_file_keeps_observer_required(self):
        self.assertEqual(config.load_config(), {})
        with self.assertRaisesRegex(AssertionError, "required: --latitude, --longitude, --elevation"):
            self.parse()
        args = self.parse(required=False)
        self.assertEqual((args.latitude, args.longitude, args.elevation), (None, None, None))

    def test_config_provides_defaults_that_options_override(self):
        self.write("latitude = 51.18\nlongitude = -1.83\nelevation = 0\n")
        args = self.parse()
        self.assertEqual((args.latitude, args.longitude, args.elevation), (51.18, -1.83, 0.0))
        self.assertIsInstance(args.elevation, float)
        args = self.parse("--latitude", "-33.9")
        self.assertEqual((args.latitude, args.longitude), (-33.9, -1.83))

    def test_partial_config_only_requires_the_missing_values(self):
        self.write("latitude = 51.18\nlongitude = -1.83\n")
        with self.assertRaisesRegex(AssertionError, "required: --elevation$"):
            self.parse()
        self.assertEqual(self.parse("--elevation", "100").elevation, 100.0)

    def test_horizons_location_takes_the_elevation_in_km(self):
        args = argparse.Namespace(latitude=51.18, longitude=-1.83, elevation=100.0)
        self.assertEqual(config.observer_location(args), {"lon": -1.83, "lat": 51.18, "elevation": 0.1})

    def run_helper(self, helper):
        parser = argparse.ArgumentParser()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                return helper(parser)
            except SystemExit:
                raise AssertionError(err.getvalue())

    def test_camera_timezone_defaults_to_the_system_timezone(self):
        with patch.object(config, "local_timezone", return_value=ZoneInfo("Pacific/Auckland")):
            self.assertEqual(self.run_helper(config.camera_timezone), ZoneInfo("Pacific/Auckland"))
            self.write('timezone = "Europe/Paris"\n')
            self.assertEqual(self.run_helper(config.camera_timezone), ZoneInfo("Europe/Paris"))
        for text, message in [('timezone = "Mars/Olympus"', "Unknown timezone Mars/Olympus"), ("timezone = 1", "must be a string")]:
            with self.subTest(text=text):
                self.write(text)
                with self.assertRaisesRegex(AssertionError, message):
                    self.run_helper(config.camera_timezone)

    def test_solve_scale_defaults_and_config(self):
        self.assertEqual(self.run_helper(config.solve_scale), (3.0, 20.0))
        self.write("[solve]\nscale_low = 16\n")
        self.assertEqual(self.run_helper(config.solve_scale), (16.0, 20.0))
        for text, message in [
            ("solve = 3", r"\[solve\] in .* must be a table"),
            ("[solve]\nscale_min = 3", r"Unknown setting scale_min in \[solve\]"),
            ('[solve]\nscale_low = "3"', r"scale_low in \[solve\] of .* must be a number"),
            ("[solve]\nscale_low = 20\nscale_high = 16", "0 < scale_low < scale_high"),
        ]:
            with self.subTest(text=text):
                self.write(text)
                with self.assertRaisesRegex(AssertionError, message):
                    self.run_helper(config.solve_scale)

    def test_invalid_config_fails_clearly(self):
        for text, message in [
            ("latitude = ", "Could not read"),
            ('latitude = "51.18"', "latitude in .* must be a number"),
            ("elevation = true", "elevation in .* must be a number"),
        ]:
            with self.subTest(text=text):
                self.write(text)
                with self.assertRaisesRegex(AssertionError, message):
                    self.parse()

    def test_commands_take_the_observer_from_the_config(self):
        self.write("latitude = 51.18\nlongitude = -1.83\nelevation = 100\n")
        for name in COMMANDS:
            with self.subTest(command=name):
                module = importlib.import_module(f"astrophoto.{name}")
                out = io.StringIO()
                with patch.object(sys, "argv", [name, "--help"]), contextlib.redirect_stdout(out), \
                        self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 0)
                self.assertIn("(default: 51.18 from config)", " ".join(out.getvalue().split()))


class UpdateConfigTests(unittest.TestCase):
    def test_empty_document_gets_top_level_keys_and_tables(self):
        text = config.update_config("", {None: {"latitude": 51.18, "elevation": 100}, "solve": {"scale_low": 16}})
        self.assertEqual(text, "latitude = 51.18\nelevation = 100\n\n[solve]\nscale_low = 16\n")

    def test_existing_keys_are_replaced_and_comments_kept(self):
        text = (
            "# Stonehenge\nlatitude = 1\nlongitude = -1.83 # west\n\n"
            "[capture]\niso = 800\n\n[capture.flat]\nshutterspeed = 50\n"
        )
        updated = config.update_config(text, {None: {"latitude": 51.18, "elevation": 0}, "capture": {"iso": 400, "frames": 24}})
        self.assertEqual(updated, (
            "# Stonehenge\nlatitude = 51.18\nlongitude = -1.83 # west\nelevation = 0\n\n"
            "[capture]\niso = 400\nframes = 24\n\n[capture.flat]\nshutterspeed = 50\n"
        ))

    def test_top_level_keys_go_before_the_first_table(self):
        text = config.update_config("[solve]\nscale_low = 3\n", {None: {"timezone": "Europe/Paris"}})
        self.assertEqual(text, 'timezone = "Europe/Paris"\n\n[solve]\nscale_low = 3\n')
        self.assertEqual(tomllib.loads(text), {"timezone": "Europe/Paris", "solve": {"scale_low": 3}})

    def test_keys_set_in_an_unexpected_way_are_left_alone(self):
        for text in ("solve.scale_low = 3\n", "solve = { scale_low = 3 }\n"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, r"scale_low in \[solve\]"):
                config.update_config(text, {"solve": {"scale_low": 16}})


class ConfigCommandTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.dict(os.environ, {"XDG_CONFIG_HOME": directory.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.path = Path(directory.name) / "astrophoto" / "config.toml"

    def run_config(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        status = 0
        with patch.object(sys, "argv", ["astrophoto config", *arguments]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                config_command.main()
            except SystemExit as error:
                status = error.code
        return status, out.getvalue(), err.getvalue()

    def test_creates_the_file_that_commands_then_read(self):
        status, out, _ = self.run_config("--latitude", "51.18", "--longitude", "-1.83", "--elevation", "100")
        self.assertEqual((status, out), (0, f"Updated {self.path}\n"))
        self.assertEqual(self.path.read_text(), "latitude = 51.18\nlongitude = -1.83\nelevation = 100\n")
        parser = argparse.ArgumentParser()
        config.add_observer_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual((args.latitude, args.longitude, args.elevation), (51.18, -1.83, 100.0))

    def test_updates_only_the_given_values(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("latitude = 51.18\n\n[capture]\niso = 800\n")
        status, _, _ = self.run_config("--latitude", "-33.9", "--aperture", "4.0", "--shutterspeed", "1/4000", "--scale-high", "16")
        self.assertEqual(status, 0)
        self.assertEqual(self.path.read_text(), (
            "latitude = -33.9\n\n[capture]\niso = 800\naperture = 4.0\nshutterspeed = \"1/4000\"\n"
            "\n[solve]\nscale_high = 16\n"
        ))

    def test_prints_the_file_without_options(self):
        status, _, err = self.run_config()
        self.assertEqual(status, f"No config file at {self.path}; see 'astrophoto config --help' to create one")
        self.path.parent.mkdir(parents=True)
        self.path.write_text("# Stonehenge\nlatitude = 51.18\n")
        self.assertEqual(self.run_config(), (0, "# Stonehenge\nlatitude = 51.18\n", ""))

    def test_invalid_values_are_rejected_before_writing(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("[solve]\nscale_low = 3\n")
        for arguments, message in [
            (["--latitude", "91"], "must be between -90 and 90"),
            (["--elevation", "nan"], "must be finite"),
            (["--timezone", "Mars/Olympus"], "unknown timezone Mars/Olympus"),
            (["--frames", "0"], "must be at least 1"),
            (["--iso", "help"], "astrophoto capture"),
            (["--scale-high", "2"], "scale_low < scale_high"),
        ]:
            with self.subTest(arguments=arguments):
                status, _, err = self.run_config(*arguments)
                self.assertEqual(status, 2)
                self.assertIn(message, err)
        self.assertEqual(self.path.read_text(), "[solve]\nscale_low = 3\n")

    def test_unreadable_file_is_left_alone(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("latitude = \n")
        status, _, err = self.run_config("--latitude", "51.18")
        self.assertEqual(status, 2)
        self.assertIn("Could not read", err)
        self.assertEqual(self.path.read_text(), "latitude = \n")


if __name__ == "__main__":
    unittest.main()
