import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "astrophoto" / "capture.sh"

class CaptureScriptTests(unittest.TestCase):
    def run_capture(self, directory, *arguments, fail_stage=""):
        executable = directory / "bin" / "gphoto2"
        executable.parent.mkdir(exist_ok=True)
        executable.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" CALL "$@" >> "$CAPTURE_ARGUMENTS"\n'
            'for argument in "$@"; do\n'
            '  if [ "$argument" = "$FAIL_STAGE" ]; then exit 1; fi\n'
            'done\n'
        )
        executable.chmod(0o755)
        log = directory / "arguments.txt"
        result = subprocess.run(
            ["sh", str(SCRIPT), *arguments], cwd=directory,
            env=dict(os.environ, PATH=f"{executable.parent}:{os.environ['PATH']}",
                     CAPTURE_ARGUMENTS=str(log), FAIL_STAGE=fail_stage),
            capture_output=True, text=True,
        )
        arguments = [call.splitlines() for call in log.read_text().split("CALL\n")[1:]] if log.exists() else []
        folders = [path for path in directory.iterdir() if path.is_dir() and path.name != "bin"]
        return result, arguments, folders

    def test_default_light_dark_and_flat_modes(self):
        for mode, suffix, shutter in [(None, "-light", "5"), ("light", "-light", "5"), ("dark", "-dark", "5"), ("flat", "-flat", "50")]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                result, arguments, folders = self.run_capture(directory, *([mode] if mode else []))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(arguments), 3)
                capture, download, cleanup = arguments
                self.assertEqual(len(folders), 1)
                self.assertRegex(folders[0].name, rf"^\d{{8}}-\d{{4}}{suffix}$")
                self.assertIn(f"shutterspeed={shutter}", capture)
                self.assertIn("iso=800", capture)
                self.assertIn("aperture=4.0", capture)
                self.assertIn("imageformat=RAW", capture)
                self.assertFalse(any(argument.startswith("capturetarget=") for argument in capture))
                self.assertIn("--capture-image", capture)
                self.assertNotIn("--capture-image-and-download", capture)
                self.assertEqual(capture[capture.index("--interval") + 1], "1")
                self.assertEqual(capture[capture.index("--frames") + 1], "20")
                self.assertEqual(download, [
                    "--folder", "/", "--recurse", "--filename",
                    f"{folders[0].name}/%Y%m%d-%H%M%S.%C", "--get-all-files",
                ])
                self.assertEqual(cleanup, ["--folder", "/DCIM/100CANON", "--delete-file", "1..20"])

    def test_failed_capture_or_download_never_deletes_files(self):
        for stage, expected_calls in [("--capture-image", 1), ("--get-all-files", 2)]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                result, commands, folders = self.run_capture(Path(temporary), fail_stage=stage)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(commands), expected_calls)
                self.assertEqual(len(folders), 1)
                self.assertTrue(all("--delete-file" not in command for command in commands))

    def test_failed_deletion_reports_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, commands, _ = self.run_capture(Path(temporary), fail_stage="--delete-file")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(commands), 3)

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
