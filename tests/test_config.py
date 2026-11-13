import argparse
import contextlib
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
        self.assertEqual(self.parse("--elevation", "0.1").elevation, 0.1)

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
        self.write("latitude = 51.18\nlongitude = -1.83\nelevation = 0.1\n")
        for name in COMMANDS:
            with self.subTest(command=name):
                module = importlib.import_module(f"astrophoto.{name}")
                out = io.StringIO()
                with patch.object(sys, "argv", [name, "--help"]), contextlib.redirect_stdout(out), \
                        self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 0)
                self.assertIn("(default: 51.18 from config)", " ".join(out.getvalue().split()))


if __name__ == "__main__":
    unittest.main()
