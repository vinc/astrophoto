import contextlib
import importlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from astrophoto import __main__ as cli


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = cli.main(list(arguments))
        return status, out.getvalue(), err.getvalue()

    def test_help_lists_every_command(self):
        status, out, _ = self.run_cli("--help")
        self.assertEqual(status, 0)
        for name in cli.COMMANDS:
            self.assertIn(f"  {name} ", out)

    def test_missing_or_unknown_command_prints_usage_and_fails(self):
        status, _, err = self.run_cli()
        self.assertEqual(status, 2)
        self.assertIn("usage: astrophoto <command>", err)
        status, _, err = self.run_cli("stak")
        self.assertEqual(status, 2)
        self.assertIn("unknown command 'stak'", err)

    def test_python_commands_run_their_main_with_the_command_as_program_name(self):
        seen = []
        module = importlib.import_module("astrophoto.weather")
        with patch.object(module, "main", side_effect=lambda: seen.append(list(sys.argv))), \
                patch.object(sys, "argv", ["astrophoto"]):
            status, _, _ = self.run_cli("weather", "--latitude", "51.18")
        self.assertEqual(status, 0)
        self.assertEqual(seen, [["astrophoto weather", "--latitude", "51.18"]])

    def test_every_python_command_has_a_main(self):
        for name in cli.COMMANDS:
            if name not in cli.SCRIPTS:
                self.assertTrue(callable(importlib.import_module(f"astrophoto.{name}").main), name)

    def test_shell_commands_run_their_script_and_return_its_status(self):
        for name in cli.SCRIPTS:
            with self.subTest(command=name), patch.object(cli.subprocess, "run") as run:
                run.return_value.returncode = 1
                status, _, _ = self.run_cli(name, "dark")
                self.assertEqual(status, 1)
                script = Path(cli.__file__).with_name(f"{name}.sh")
                self.assertTrue(script.exists())
                run.assert_called_once_with(["sh", str(script), "dark"])


if __name__ == "__main__":
    unittest.main()
