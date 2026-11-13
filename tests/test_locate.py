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
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS

from astrophoto import locate
from astrophoto.lib.camera import stack_display_limits

def make_header(shape, date):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [120, 20]
    wcs.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]
    wcs.wcs.cdelt = [-0.02, 0.02]
    header = wcs.to_header()
    header["DATE-AVG"] = date
    return header

def ephemerides(query, observer, epochs):
    return Table({
        "RA": [120.0] * len(epochs),
        "DEC": [20.0] * len(epochs),
        "targetname": ["4 Vesta (A807 FA)"] * len(epochs),
    })

class LocateTests(unittest.TestCase):
    def render(self, path, radius=None, source_path=None):
        images = []
        subplots = locate.plt.subplots
        def record(*args, **kwargs):
            figure, axes = subplots(*args, **kwargs)
            images.append(axes)
            return figure, axes
        with patch.object(locate.plt, "subplots", side_effect=record), \
                patch("matplotlib.figure.Figure.savefig"), \
                contextlib.redirect_stdout(io.StringIO()):
            locate.save_image(path, 120, 20, Time("2025-01-01"), "test",
                              radius=radius, source_path=source_path)
        return images[0].images[0]

    def test_stack_and_crop_reuse_saved_display_limits(self):
        data = np.arange(15000, dtype=np.float32).reshape(100, 150)
        header = make_header(data.shape, "2025-01-01T00:00:00")
        header.update({"NCOMBINE": 20, "DISPLO": 50.0, "DISPHI": 200.0})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stack.new"
            fits.writeto(path, data, header)
            original = path.read_bytes()
            for radius in [None, 8]:
                image = self.render(path, radius=radius)
                self.assertEqual(image.get_clim(), (50, 200))
            self.assertEqual(path.read_bytes(), original)

    def test_older_stack_uses_full_frame_percentiles_even_when_cropped(self):
        data = np.arange(15000, dtype=np.float32).reshape(100, 150)
        data[0, :2] = [np.nan, np.inf]
        header = make_header(data.shape, "2025-01-01T00:00:00")
        header["NCOMBINE"] = 20
        expected = np.percentile(data[np.isfinite(data)], [5, 99.7])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stack.new"
            fits.writeto(path, data, header)
            for radius in [None, 8]:
                image = self.render(path, radius=radius)
                np.testing.assert_allclose(image.get_clim(), expected)

    def test_cached_solution_uses_original_stacks_saved_limits(self):
        data = np.arange(15000, dtype=np.float32).reshape(100, 150)
        solved = make_header(data.shape, "2025-01-01T00:00:00")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, cached = directory / "stack.fit", directory / "stack.new"
            header = fits.Header({"NCOMBINE": 20, "DISPLO": 50.0, "DISPHI": 200.0})
            fits.writeto(source, data, header)
            fits.writeto(cached, data, solved)
            self.assertEqual(self.render(cached, source_path=source).get_clim(), (50, 200))

    def test_raw_preview_retains_zscale_and_stack_fallback_handles_flat_images(self):
        data = np.arange(60000, dtype=np.float32).reshape(200, 300)
        header = make_header(data.shape, "2025-01-01T00:00:00")
        header["RAWFILE"] = "test.CR2"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.new"
            fits.writeto(path, data, header)
            image = self.render(path)
            expected = ZScaleInterval().get_limits(image.get_array())
            np.testing.assert_allclose(image.get_clim(), expected)

        low, high = stack_display_limits(np.ones((10, 10)))
        self.assertEqual(low, 1)
        self.assertGreater(high, low)
        invalid = fits.Header({"DISPLO": 2, "DISPHI": 1})
        self.assertEqual(stack_display_limits(np.ones((10, 10)), invalid), (low, high))
        with self.assertRaisesRegex(ValueError, "no finite pixels"):
            stack_display_limits(np.full((10, 10), np.nan))

    def test_importing_shared_locating_code_preserves_the_selected_backend(self):
        result = subprocess.run(
            [sys.executable, "-c", "import matplotlib; import astrophoto.center; import astrophoto.crop; print(matplotlib.get_backend())"],
            cwd=locate.ROOT,
            env=dict(os.environ, MPLBACKEND="svg"),
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip().lower(), "svg")

    def test_full_frame_raw_and_stack_outputs_reuse_crop_style_without_a_gui(self):
        figures = []
        subplots = locate.plt.subplots
        def record_figure(*args, **kwargs):
            figure, axes = subplots(*args, **kwargs)
            figures.append((figure, axes))
            return figure, axes

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = directory / "later.fit"
            stack = directory / "early-stack.fit"
            header = make_header((200, 300), "2025-01-02T00:00:00.125")
            header["RAWFILE"] = "test.CR2"
            fits.writeto(raw, np.arange(60000, dtype=float).reshape(200, 300), header)
            header = make_header((100, 150), "2025-01-01T00:00:00.125")
            header["NCOMBINE"] = 20
            fits.writeto(stack, np.arange(15000, dtype=float).reshape(100, 150), header)
            originals = [path.read_bytes() for path in [raw, stack]]
            argv = [
                "locate.py", str(raw), str(stack), "--query", "Vesta",
                "--latitude", "0", "--longitude", "0", "--elevation", "0",
            ]
            with patch.object(sys, "argv", argv), \
                    patch.object(locate, "get_ephemerides", side_effect=ephemerides) as lookup, \
                    patch.object(locate.plt, "subplots", side_effect=record_figure), \
                    patch.object(locate.plt, "show") as show, \
                    patch.object(locate.plt, "pause") as pause, \
                    contextlib.redirect_stdout(io.StringIO()):
                locate.main()

            self.assertTrue((directory / "later-locate-vesta.png").is_file())
            self.assertTrue((directory / "early-stack-locate-vesta.png").is_file())
            self.assertEqual(originals, [path.read_bytes() for path in [raw, stack]])
            epochs = lookup.call_args.args[2]
            self.assertEqual(epochs, sorted(epochs))
            show.assert_not_called()
            pause.assert_not_called()

        self.assertEqual(len(figures), 2)
        for index, (figure, axes) in enumerate(figures):
            self.assertEqual(axes.images[0].get_array().shape, (100, 150))
            self.assertFalse(axes.axison)
            self.assertEqual(len(axes.collections), 1)
            self.assertEqual(len(axes.lines), 0)
            self.assertEqual(len(axes.texts), 0)
            self.assertEqual(axes.get_title(), f"4 Vesta (A807 FA)\n2025-01-0{index + 1} 00:00:00 UTC")
            expected = [74.5, 49.5] if index == 0 else [149.5, 99.5]
            np.testing.assert_allclose(axes.collections[0].get_offsets()[0], expected)
            self.assertFalse(locate.plt.fignum_exists(figure.number))

    def test_unsolved_stack_keeps_solution_and_png_after_temporary_files_are_removed(self):
        solved_paths = []
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "session-stack.fit"
            data = np.ones((100, 150), dtype=np.float32)
            fits.writeto(source, data, fits.Header({"DATE-AVG": "2025-01-01T00:00:00", "NCOMBINE": 20}))
            original = source.read_bytes()
            def solve(command, **kwargs):
                output = Path(command[command.index("--new-fits") + 1])
                marker = Path(command[command.index("--solved") + 1])
                fits.writeto(output, data, make_header(data.shape, "2025-01-01T00:00:00"))
                marker.write_bytes(b"\x01")
                solved_paths.append(output)

            argv = [
                "locate.py", str(source), "--query", "vesta",
                "--latitude", "0", "--longitude", "0", "--elevation", "0",
            ]
            with patch.object(sys, "argv", argv), \
                    patch("astrophoto.locate.subprocess.run", side_effect=solve) as solver, \
                    patch.object(locate, "get_ephemerides", side_effect=ephemerides), \
                    contextlib.redirect_stdout(io.StringIO()):
                locate.main()

            solver.assert_called_once()
            self.assertTrue((directory / "session-stack-locate-vesta.png").is_file())
            self.assertTrue((directory / "session-stack.new").is_file())
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(all(not path.exists() for path in solved_paths))

if __name__ == "__main__":
    unittest.main()
