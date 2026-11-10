import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

import camera
import stack

def response_map():
    y, x = np.mgrid[:16, :20]
    response = 1 - 0.3 * ((x - 9.5) / 10)**2 - 0.2 * ((y - 7.5) / 8)**2
    response[6:10, 8:12] *= 0.8
    for dy in range(2):
        for dx in range(2):
            plane = response[dy::2, dx::2]
            plane /= np.median(plane)
    return response

def frame_header():
    return fits.Header({
        "BLACKLVL": 128, "WHITELEV": 4095, "ISO": 800, "EXPTIME": 1 / 800,
        "INSTRUME": "test camera", "OBJECTIV": "test lens", "FOCALLEN": 75, "FNUMBER": 4,
    })

def write_flats(directory, response):
    paths = []
    colours = np.tile([[400, 800], [600, 300]], (8, 10))
    for i, brightness in enumerate([0.7, 1.0, 1.3]):
        data = 128 + brightness * colours * response
        if i == 0:
            data[0, 0] += 1000
        path = directory / f"flat-{i}.fit"
        header = frame_header()
        header["EXPTIME"] *= brightness
        fits.writeto(path, data, header)
        paths.append(path)
    return paths

class FlatTests(unittest.TestCase):
    def test_weakest_bayer_plane_warns_below_fifteen_percent_without_rejecting(self):
        for fraction in [0.10, 0.15, 0.50]:
            for y in range(2):
                for x in range(2):
                    with self.subTest(fraction=fraction, y=y, x=x), \
                            tempfile.TemporaryDirectory() as temporary:
                        path = Path(temporary) / "flat.fit"
                        header = frame_header()
                        header["WHITELEV"] = header["BLACKLVL"] + 4000
                        span = header["WHITELEV"] - header["BLACKLVL"]
                        data = np.full((16, 20), header["BLACKLVL"] + 0.5 * span)
                        data[y::2, x::2] = header["BLACKLVL"] + fraction * span
                        fits.writeto(path, data, header)
                        output, errors = io.StringIO(), io.StringIO()
                        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                            flat, _ = stack.master_flat([path])
                        self.assertIn(f"weakest Bayer plane at {fraction:.0%} of full scale", output.getvalue())
                        self.assertEqual("underexposed" in errors.getvalue(), fraction < 0.15)
                        np.testing.assert_allclose(flat, 1)

    def test_missing_white_level_skips_exposure_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flat.fit"
            header = frame_header()
            del header["WHITELEV"]
            fits.writeto(path, np.full((16, 20), 150.0), header)
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                flat, _ = stack.master_flat([path])
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(errors.getvalue(), "")
            np.testing.assert_allclose(flat, 1)

    def test_invalid_white_level_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flat.fit"
            header = frame_header()
            header["WHITELEV"] = header["BLACKLVL"]
            fits.writeto(path, np.full((16, 20), 100.0), header)
            with self.assertRaisesRegex(ValueError, "invalid WHITELEV"):
                stack.master_flat([path])

    def test_master_removes_bias_and_normalizes_brightness_and_bayer_positions(self):
        response = response_map()
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_flats(Path(temporary), response)
            flat, metadata = stack.master_flat(paths)
        np.testing.assert_allclose(flat, response, rtol=1e-6)
        self.assertEqual(metadata["FOCALLEN"], 75)
        self.assertNotIn("EXPTIME", metadata)

    def test_dark_then_flat_correction_precedes_binning_debanding_and_alignment(self):
        response = response_map()
        y, x = np.mgrid[:16, :20]
        dark = 128 + 2 * y + x % 2
        signal = np.tile([[10.0, 20.0], [30.0, 40.0]], (8, 10))
        signal[4:8, 10:14] += 50
        signal[:2, :2] = -3
        expected = signal.reshape(8, 2, 10, 2).mean(axis=(1, 3))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            flats = write_flats(directory, response)
            header = frame_header()
            header["EXPTIME"] = 5
            dark_path = directory / "dark.fit"
            fits.writeto(dark_path, dark, header)
            for i in range(2):
                fits.writeto(directory / f"light-{i}.fit", dark + signal * response, header)

            def deband(image, amount):
                self.assertEqual(amount, 0.5)
                np.testing.assert_allclose(image, expected, atol=1e-4)
                return image

            def register(image, reference):
                np.testing.assert_allclose(image, expected, atol=1e-4)
                np.testing.assert_allclose(reference, expected, atol=1e-4)
                return image.copy(), np.zeros(image.shape, dtype=bool)

            with patch.object(stack, "reduce_banding", side_effect=deband) as correction, \
                    patch.object(stack.astroalign, "register", side_effect=register) as align, \
                    contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory, dark_files=[dark_path],
                                      flat_files=[*flats, flats[0]], normalize=False, deband=0.5)
            self.assertEqual(correction.call_count, 2)
            align.assert_called_once()
            output = directory / f"{directory.name}-stack.fit"
            data, metadata = fits.getdata(output, header=True)
            np.testing.assert_allclose(data, expected, atol=1e-4)
            self.assertTrue(metadata["FLATCOR"])
            self.assertTrue(metadata["DARKCOR"])
            self.assertEqual(metadata["NFLATS"], 3)
            self.assertEqual(metadata["NCOMBINE"], 2)
            self.assertEqual(metadata["BLACKLVL"], 0)
            self.assertEqual(metadata["FLATBIAS"], "BLACKLVL")
            np.testing.assert_allclose(camera.read_image(output)[0], expected, atol=1e-4)

    def test_flat_only_removes_light_bias_before_dividing(self):
        response = response_map()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            flats = write_flats(directory, response)
            header = frame_header()
            header["EXPTIME"] = 5
            fits.writeto(directory / "light.fit", 128 + 20 * response, header)
            with contextlib.redirect_stdout(io.StringIO()):
                stack.stack_directory(directory, flat_files=flats, normalize=False)
            data, metadata = fits.getdata(directory / f"{directory.name}-stack.fit", header=True)
            np.testing.assert_allclose(data, 20, atol=1e-4)
            self.assertEqual(metadata["BLACKLVL"], 0)
            self.assertNotIn("DARKCOR", metadata)

    def test_missing_bias_saturation_and_invalid_response_fail_clearly(self):
        for case in ["missing bias", "saturated", "dark flat", "dead pixel", "nonfinite"]:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "flat.fit"
                header = frame_header()
                data = np.full((16, 20), 1000.0)
                if case == "missing bias":
                    del header["BLACKLVL"]
                elif case == "saturated":
                    data[:] = 4095
                elif case == "dark flat":
                    data[:] = 100
                elif case == "dead pixel":
                    data[0, 0] = 128
                else:
                    data[0, 0] = np.nan
                fits.writeto(path, data, header)
                with self.assertRaises(ValueError):
                    stack.master_flat([path])

    def test_flat_optical_metadata_and_shape_must_match_lights(self):
        changes = [{"ISO": 1600}, {"INSTRUME": "other camera"}, {"FOCALLEN": 100},
                   {"FNUMBER": 5.6}, {"OBJECTIV": "other lens"}, {"FLATCOR": True}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "light.fit"
            for change in changes:
                with self.subTest(change=change):
                    header = frame_header()
                    header.update(change)
                    fits.writeto(path, np.full((16, 20), 150), header, overwrite=True)
                    with self.assertRaises(ValueError):
                        stack.load(path, flat=np.ones((16, 20)), flat_metadata=frame_header())
            fits.writeto(path, np.full((8, 10), 150), frame_header(), overwrite=True)
            with self.assertRaisesRegex(ValueError, "shape"):
                stack.load(path, flat=np.ones((16, 20)))

    def test_cli_accepts_flat_and_dark_lists_and_rejects_overlap(self):
        argv = ["stack.py", ".", "--dark", "dark.fit", "--flat", "flat-1.fit", "flat-2.fit"]
        with patch.object(sys, "argv", argv), patch.object(stack, "stack_directory") as run:
            stack.main()
        self.assertEqual(run.call_args.kwargs["flat_files"], [Path("flat-1.fit"), Path("flat-2.fit")])
        with self.assertRaisesRegex(ValueError, "both a dark and a flat"):
            stack.stack_directory(Path("."), dark_files=["same.fit"], flat_files=["same.fit"])

if __name__ == "__main__":
    unittest.main()
