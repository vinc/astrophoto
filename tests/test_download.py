import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from astrophoto import download


class FakeResponse:
    def __init__(self, url, fail):
        self.url, self.fail = url, fail
        self.headers = {"Content-Length": "6"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.fail == "status" and self.url.endswith("index-4110.fits"):
            raise requests.HTTPError("404 Not Found")

    def iter_content(self, size):
        yield b"abc"
        if self.fail == "stream" and self.url.endswith("index-4110.fits"):
            raise requests.ConnectionError("connection reset")
        yield b"def"


class DownloadTests(unittest.TestCase):
    def run_download(self, fail="", existing=()):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        data = directory / "astrophoto"
        data.mkdir()
        for name in existing:
            (data / name).write_text("old")
        session = MagicMock()
        session.__enter__.return_value = session
        session.get.side_effect = lambda url, **kwargs: FakeResponse(url, fail)
        out = io.StringIO()
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(directory)}), patch.object(sys, "argv", ["download"]), \
                patch.object(download.requests, "Session", return_value=session), contextlib.redirect_stdout(out):
            try:
                download.main()
                error = None
            except SystemExit as exit:
                error = str(exit)
        urls = [call.args[0] for call in session.get.call_args_list]
        return data, urls, out.getvalue(), error

    def test_downloads_every_file_into_the_data_directory(self):
        data, urls, out, error = self.run_download()
        self.assertIsNone(error)
        self.assertEqual(urls, download.URLS)
        files = sorted(path.name for path in data.iterdir())
        self.assertEqual(len([name for name in files if name.startswith("index-41")]), 13)
        self.assertIn("constellations.lines.json", files)
        self.assertEqual((data / "stars.6.json").read_bytes(), b"abcdef")
        self.assertIn(f"Downloading data to {data}", out)

    def test_existing_files_are_kept_and_skipped(self):
        data, urls, _, error = self.run_download(existing=["index-4107.fits", "stars.6.json"])
        self.assertIsNone(error)
        self.assertEqual(len(urls), len(download.URLS) - 2)
        self.assertEqual((data / "index-4107.fits").read_text(), "old")

    def test_failures_stop_without_leaving_partial_files(self):
        for fail in ("status", "stream"):
            with self.subTest(fail=fail):
                data, urls, _, error = self.run_download(fail=fail)
                self.assertRegex(error, r"^Could not download https://data.astrometry.net/4100/index-4110.fits: ")
                self.assertEqual(urls[-1], f"{download.INDEXES}/index-4110.fits")
                self.assertEqual(sorted(path.name for path in data.iterdir()),
                                 [f"index-41{number:02d}.fits" for number in range(7, 10)])


if __name__ == "__main__":
    unittest.main()
