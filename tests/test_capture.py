import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "capture.sh"

class CaptureScriptTests(unittest.TestCase):
    def run_capture(self, directory, *arguments):
        executable = directory / "bin" / "gphoto2"
        executable.parent.mkdir(exist_ok=True)
        executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGUMENTS"\n')
        executable.chmod(0o755)
        log = directory / "arguments.txt"
        result = subprocess.run(
            ["sh", str(SCRIPT), *arguments], cwd=directory,
            env=dict(os.environ, PATH=f"{executable.parent}:{os.environ['PATH']}",
                     CAPTURE_ARGUMENTS=str(log)),
            capture_output=True, text=True,
        )
        arguments = log.read_text().splitlines() if log.exists() else []
        folders = [path for path in directory.iterdir() if path.is_dir() and path.name != "bin"]
        return result, arguments, folders

    def test_default_dark_and_flat_modes(self):
        for mode, suffix, shutter in [(None, "-light", "5"), ("dark", "-dark", "5"), ("flat", "-flat", "45")]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                result, arguments, folders = self.run_capture(directory, *([mode] if mode else []))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(folders), 1)
                self.assertRegex(folders[0].name, rf"^\d{{8}}-\d{{4}}{suffix}$")
                self.assertIn(f"shutterspeed={shutter}", arguments)
                self.assertIn("iso=800", arguments)
                self.assertIn("aperture=4.0", arguments)
                self.assertIn("imageformat=RAW", arguments)
                self.assertEqual(arguments[arguments.index("--frames") + 1], "20")
                self.assertEqual(arguments[-1], f"{folders[0].name}/%Y%m%d-%H%M%S.%C")

    def test_invalid_arguments_do_not_capture_or_create_a_session(self):
        for arguments in [("unknown",), ("dark", "extra")]:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temporary:
                result, commands, folders = self.run_capture(Path(temporary), *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Usage:", result.stderr)
                self.assertEqual(commands, [])
                self.assertEqual(folders, [])

if __name__ == "__main__":
    unittest.main()
