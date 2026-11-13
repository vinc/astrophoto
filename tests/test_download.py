import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "astrophoto" / "download.sh"


class DownloadScriptTests(unittest.TestCase):
    def run_download(self, directory, environment, fail=""):
        executable = directory / "bin" / "wget"
        executable.parent.mkdir(exist_ok=True)
        executable.write_text(
            '#!/bin/sh\n'
            'url=$1\n'
            '[ "$url" = "-nc" ] && url=$2\n'
            'case "$url" in *"$FAIL"*) [ -n "$FAIL" ] && exit 1 ;; esac\n'
            'touch "${url##*/}"\n'
        )
        executable.chmod(0o755)
        env = dict(os.environ, PATH=f"{executable.parent}:{os.environ['PATH']}", HOME=str(directory / "home"), FAIL=fail)
        env.pop("XDG_DATA_HOME", None)
        env.update(environment)
        return subprocess.run(["sh", str(SCRIPT)], cwd=directory, env=env, capture_output=True, text=True)

    def test_downloads_into_the_xdg_data_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = self.run_download(directory, {"XDG_DATA_HOME": str(directory / "data")})
            self.assertEqual(result.returncode, 0, result.stderr)
            files = sorted(path.name for path in (directory / "data" / "astrophoto").iterdir())
            self.assertEqual(len([name for name in files if name.startswith("index-41")]), 13)
            self.assertIn("stars.6.json", files)
            self.assertIn("constellations.lines.json", files)
            self.assertEqual(sorted(path.name for path in directory.iterdir()), ["bin", "data"])

    def test_defaults_to_local_share_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = self.run_download(directory, {}, fail="index-4110")
            self.assertEqual(result.returncode, 1)
            files = sorted(path.name for path in (directory / "home" / ".local" / "share" / "astrophoto").iterdir())
            self.assertEqual(files, [f"index-41{i:02d}.fits" for i in range(7, 10)])


if __name__ == "__main__":
    unittest.main()
