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

if __name__ == "__main__":
    unittest.main()
