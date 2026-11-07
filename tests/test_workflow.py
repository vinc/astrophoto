import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

import crop
import stack

class WorkflowTests(unittest.TestCase):
    def test_stacking_twice_preserves_pixels_and_dimensions(self):
        rng = np.random.default_rng(42)
        y, x = np.mgrid[:128, :128]
        image = rng.normal(100, 1, (128, 128))
        for sx, sy in rng.uniform(8, 120, (35, 2)):
            image += 1000 * np.exp(-((x - sx)**2 + (y - sy)**2) / 3)
        raw = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "20260101-0000"
            directory.mkdir()
            for i in range(3):
                fits.writeto(
                    directory / f"IMG_{i:04}.fit",
                    raw + rng.normal(0, 0.2, raw.shape),
                )

            output = directory / f"{directory.name}-stack.fit"
            with contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory)
                first = fits.getdata(output).copy()
                stack.stack_directory(directory)
                second = fits.getdata(output)

            self.assertEqual(first.shape, (128, 128))
            np.testing.assert_array_equal(first, second)

    def test_crops_match_epochs_for_unsorted_and_duplicate_times(self):
        epochs = [2461042.5, 2461041.5, 2461042.5]
        calls = []

        def ephemerides(query, observer, requested_epochs):
            # Model Horizons returning rows sorted by epoch, including duplicates
            ordered = sorted(requested_epochs)
            return Table({
                "RA": [10 + jd - min(epochs) for jd in ordered],
                "DEC": [0.0] * len(ordered),
            })

        def record_crop(path, radius, ra, dec, time, query):
            calls.append((path.name, time.jd, ra))

        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for i, jd in enumerate(epochs):
                path = Path(temporary) / f"frame-{i}.new"
                fits.writeto(
                    path, np.zeros((10, 10)),
                    header=fits.Header({"DATE-AVG": Time(jd, format="jd").isot}),
                )
                paths.append(str(path))

            argv = [
                "crop.py", *paths, "--query", "606",
                "--latitude", "0", "--longitude", "0", "--elevation", "0",
            ]
            with patch.object(sys, "argv", argv), \
                    patch.object(crop, "get_ephemerides", side_effect=ephemerides), \
                    patch.object(crop, "crop_image", side_effect=record_crop):
                crop.main()

        self.assertEqual(len(calls), 3)
        self.assertEqual({name for name, _, _ in calls}, {
            "frame-0.new", "frame-1.new", "frame-2.new",
        })
        for name, jd, ra in calls:
            index = int(name.removeprefix("frame-").removesuffix(".new"))
            self.assertEqual(jd, epochs[index])
            self.assertEqual(ra, 10 + epochs[index] - min(epochs))

    def test_stack_header_keeps_metadata_and_mean_time(self):
        rng = np.random.default_rng(7)
        y, x = np.mgrid[:128, :128]
        image = rng.normal(100, 1, (128, 128))
        for sx, sy in rng.uniform(8, 120, (35, 2)):
            image += 1000 * np.exp(-((x - sx)**2 + (y - sy)**2) / 3)
        raw = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "20261106-2200"
            directory.mkdir()
            for i in range(3):
                header = fits.Header({
                    "DATE-OBS": f"2026-11-06T21:00:{i * 15:02}.000",
                    "DATE-AVG": f"2026-11-06T21:00:{i * 15 + 2:02}.500",
                    "EXPTIME": 5.0,
                    "INSTRUME": "Canon EOS 20D",
                    "RAWFILE": f"IMG_{i}.CR2",
                })
                data = (raw + rng.normal(0, 0.2, raw.shape)).astype(np.uint16)
                fits.writeto(directory / f"IMG_{i}.fit", data, header=header)

            with contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory)
            header = fits.getheader(directory / f"{directory.name}-stack.fit")

        self.assertEqual(header["NCOMBINE"], 3)
        self.assertEqual(header["DATE-OBS"], "2026-11-06T21:00:00.000")
        self.assertEqual(header["DATE-AVG"], "2026-11-06T21:00:17.500")
        self.assertEqual(header["EXPTIME"], 5.0)
        self.assertEqual(header["INSTRUME"], "Canon EOS 20D")
        self.assertEqual(header["BITPIX"], -32)
        self.assertNotIn("BZERO", header)
        self.assertNotIn("RAWFILE", header)

    def test_crop_bins_bayer_frames_and_keeps_marker_on_target(self):
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [20.0, -5.0]
        wcs.wcs.crpix = [100.5, 80.5]
        wcs.wcs.cdelt = [-0.005, 0.005]

        # Bright 2x2 block on full-resolution pixels x=120..121, y=60..61
        data = np.zeros((160, 200))
        data[::2, ::2] = 50  # Bayer-like pattern
        data[60:62, 120:122] = 1000
        ra, dec = wcs.pixel_to_world_values(120.5, 60.5)

        header = wcs.to_header()
        header["RAWFILE"] = "IMG_0001.CR2"

        calls = {}
        def record_imshow(self, crop, **kwargs):
            calls["crop"] = crop
        def record_scatter(self, xs, ys, **kwargs):
            calls["marker"] = (xs[0], ys[0])

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.new"
            fits.writeto(path, data, header=header)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("matplotlib.axes.Axes.imshow", record_imshow), \
                    patch("matplotlib.axes.Axes.scatter", record_scatter):
                crop.crop_image(path, 20, float(ra), float(dec),
                                Time("2026-11-06T21:00:00"), "test")

        image = calls["crop"]
        self.assertEqual(image.shape, (20, 20))
        mx, my = calls["marker"]
        peak = np.unravel_index(np.argmax(image), image.shape)
        self.assertAlmostEqual(mx, peak[1], places=6)
        self.assertAlmostEqual(my, peak[0], places=6)
        # No checkerboard left in the background
        background = image.copy()
        background[peak] = np.nan
        self.assertAlmostEqual(np.nanstd(background), 0.0)

    def test_convert_continues_after_unreadable_file(self):
        import convert

        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "broken.CR2"
            good = Path(temporary) / "good.CR2"
            bad.write_bytes(b"not a raw file")
            good.write_bytes(b"")
            converted = []

            def fake_convert(path, tz):
                if path == bad:
                    raise ValueError("unreadable")
                converted.append(path)

            argv = ["convert.py", str(bad), str(good)]
            with patch.object(sys, "argv", argv), \
                    patch.object(convert, "convert", side_effect=fake_convert), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaises(SystemExit):
                convert.main()

        self.assertEqual(converted, [good])

if __name__ == "__main__":
    unittest.main()
