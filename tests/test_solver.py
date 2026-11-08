import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import crop
import locate

def solve(command, **kwargs):
    path = Path(command[-1])
    data, header = fits.getdata(path, header=True)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [120, 20]
    wcs.wcs.crpix = [(data.shape[1] + 1) / 2, (data.shape[0] + 1) / 2]
    wcs.wcs.cdelt = [-0.02, 0.02]
    header.update(wcs.to_header())
    output = Path(command[command.index("--new-fits") + 1])
    fits.writeto(output, data, header, overwrite=True)
    Path(command[command.index("--solved") + 1]).write_bytes(b"\x01")
    output.with_suffix(".axy").touch()

class SolverTests(unittest.TestCase):
    def test_crop_and_locate_reuse_the_same_solution_and_keep_only_new(self):
        result = Table({"RA": [120.0], "DEC": [20.0], "targetname": ["4 Vesta"]})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "frame.fit"
            fits.writeto(path, np.arange(15000).reshape(100, 150), fits.Header({"DATE-AVG": "2025-01-01T00:00:00"}))
            original = path.read_bytes()
            for module, query in [(crop, "vesta"), (crop, "pallas"), (locate, "vesta")]:
                argv = [
                    module.__name__ + ".py", str(path), "--query", query,
                    "--latitude", "0", "--longitude", "0", "--elevation", "0",
                    "--scale-low", "16", "--scale-high", "18", "--cpu-limit", "180",
                ]
                with patch.object(sys, "argv", argv), \
                        patch.object(locate.subprocess, "run", side_effect=solve) as solver, \
                        patch.object(module, "get_ephemerides", return_value=result), \
                        contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                if query == "vesta" and module is crop:
                    solver.assert_called_once()
                    command = solver.call_args.args[0]
                    for option, expected in [("--scale-low", "16.0"), ("--scale-high", "18.0"), ("--cpulimit", "180")]:
                        self.assertEqual(command[command.index(option) + 1], expected)
                    self.assertFalse(Path(command[command.index("--dir") + 1]).exists())
                else:
                    solver.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual({p.name for p in directory.iterdir()}, {
                "frame.fit", "frame.new", "frame-crop-vesta.png", "frame-crop-pallas.png", "frame-locate-vesta.png",
            })

    def test_live_solution_stays_temporary_and_does_not_use_saved_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "frame.fit"
            fits.writeto(source, np.zeros((20, 30)), fits.Header({"DATE-AVG": "2025-01-01T00:00:00"}))
            with tempfile.TemporaryDirectory() as work, patch.object(locate.subprocess, "run", side_effect=solve) as solver:
                saved = locate.solve_image(source, work, None, 3, 20, keep_solution=True)
                cached = saved.read_bytes()
                live = locate.solve_image(source, work, None, 3, 20)
                self.assertEqual(live, Path(work) / "solved.new")
                self.assertEqual(saved.read_bytes(), cached)
                self.assertEqual(solver.call_count, 2)
            self.assertFalse(live.exists())
            self.assertTrue(saved.exists())

    def test_stale_invalid_or_mismatched_cached_solutions_are_replaced(self):
        for reason in ["older", "invalid", "timestamp", "dimensions", "no_wcs"]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source = directory / "frame.fit"
                fits.writeto(source, np.zeros((20, 30)), fits.Header({"DATE-AVG": "2025-01-01T00:00:00"}))
                with tempfile.TemporaryDirectory() as work, patch.object(locate.subprocess, "run", side_effect=solve) as solver:
                    cached = locate.solve_image(source, work, None, 3, 20, keep_solution=True)
                    if reason == "older":
                        os.utime(cached, ns=(1, 1))
                    elif reason == "invalid":
                        cached.write_bytes(b"broken FITS")
                    else:
                        data, header = fits.getdata(cached, header=True)
                        if reason == "timestamp":
                            header["DATE-AVG"] = "2025-01-02T00:00:00"
                        elif reason == "dimensions":
                            data = np.zeros((10, 15))
                        else:
                            del header["CTYPE1"]
                            del header["CTYPE2"]
                        fits.writeto(cached, data, header, overwrite=True)
                    result = locate.solve_image(source, work, None, 3, 20, keep_solution=True)
                    self.assertEqual(solver.call_count, 2)
                    self.assertEqual(result, cached)
                    self.assertEqual(fits.getdata(result).shape, (20, 30))
                    self.assertEqual(fits.getheader(result)["DATE-AVG"], "2025-01-01T00:00:00")
                    self.assertTrue(WCS(fits.getheader(result)).has_celestial)

    def test_raw_solution_uses_original_stem_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "IMG_0001.CR2"
            source.write_bytes(b"raw")
            def convert(path, timezone, output):
                fits.writeto(output, np.zeros((20, 30)), fits.Header({"DATE-AVG": "2025-01-01T00:00:00"}), overwrite=True)
                return output
            with tempfile.TemporaryDirectory() as work, \
                    patch.object(locate, "convert", side_effect=convert), \
                    patch.object(locate.subprocess, "run", side_effect=solve) as solver:
                for _ in range(2):
                    output = locate.solve_image(source, work, None, 3, 20, keep_solution=True)
                    self.assertEqual(output, source.with_suffix(".new"))
                solver.assert_called_once()
            self.assertTrue(output.exists())

    def test_failed_solver_preserves_existing_cache_and_raises_a_retryable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "frame.fit"
            fits.writeto(source, np.zeros((20, 30)), fits.Header({"DATE-AVG": "2025-01-01T00:00:00"}))
            cached = source.with_suffix(".new")
            cached.write_bytes(b"previous solution")
            with patch.object(locate.subprocess, "run", side_effect=subprocess.CalledProcessError(255, "solve-field")), \
                    self.assertRaisesRegex(locate.PlateSolveError, "exit status 255"):
                locate.solve_image(source, temporary, None, 3, 20, keep_solution=True)
            self.assertEqual(cached.read_bytes(), b"previous solution")

    def test_solver_output_without_coordinates_is_not_cached(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "frame.fit"
            header = fits.Header({"DATE-AVG": "2025-01-01T00:00:00"})
            fits.writeto(source, np.zeros((20, 30)), header)
            def no_coordinates(command, **kwargs):
                fits.writeto(Path(command[command.index("--new-fits") + 1]), np.zeros((20, 30)), header)
                Path(command[command.index("--solved") + 1]).touch()
            with patch.object(locate.subprocess, "run", side_effect=no_coordinates), \
                    self.assertRaisesRegex(locate.PlateSolveError, "without celestial coordinates"):
                locate.solve_image(source, temporary, None, 3, 20, keep_solution=True)
            self.assertFalse(source.with_suffix(".new").exists())

if __name__ == "__main__":
    unittest.main()
