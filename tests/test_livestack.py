from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from astro.lib.livestack import LiveStack


def sky(dx=0.0, dy=0.0, seed=0, shape=(240, 320), stars=40, trail=False, noise=5):
    """Return a star field shifted by (dx, dy) pixels, with fresh noise for each seed."""
    positions = np.random.default_rng(7).uniform(20, np.array(shape[::-1]) - 20, size=(stars, 2))
    fluxes = np.random.default_rng(8).uniform(300, 3000, size=stars)
    yy, xx = np.indices(shape, dtype=float)
    image = np.random.default_rng(seed).normal(500, noise, shape)
    for (x, y), flux in zip(positions, fluxes):
        image += flux * np.exp(-((xx - x - dx) ** 2 + (yy - y - dy) ** 2) / (2 * 1.5**2))
    if trail:
        image[120, :] += 2000
    return image.astype(np.float32)


def write_frame(directory, name, image):
    path = Path(directory) / name
    header = fits.Header({"DATE-OBS": "2026-11-27T21:00:00.000", "DATE-AVG": "2026-11-27T21:00:02.500",
                          "RAWFILE": name.replace(".fit", ".CR2")})
    fits.writeto(path, image, header)
    return path


class LiveStackTests(unittest.TestCase):
    def test_frames_are_aligned_to_the_first_and_averaged(self):
        stack = LiveStack()
        shifts = [(0, 0), (3.0, -2.0), (-4.5, 1.5), (6.0, 4.0), (1.0, -5.0), (-2.0, -3.0)]
        for index, (dx, dy) in enumerate(shifts):
            result = stack.add(sky(dx, dy, seed=index), f"frame{index}.fit")
            # The transform maps the frame back onto the reference
            self.assertAlmostEqual(result["shift"][0], -dx, delta=0.2)
            self.assertAlmostEqual(result["shift"][1], -dy, delta=0.2)
        self.assertEqual(len(stack), 6)
        image = stack.image()
        clean = sky(noise=0)
        # Away from the borders that some frames do not cover, the stack is the field with less noise than a frame
        inner = (slice(20, -20), slice(20, -20))
        residual = image[inner] - clean[inner]
        self.assertLess(np.nanstd(residual - np.nanmedian(residual)), 3)
        # Too few frames to clip yet, so every pixel there has all six
        self.assertEqual(stack.count[inner].min(), 6)
        # The edges are only covered by the frames shifted over them
        self.assertLess(stack.count.min(), 6)

    def test_a_trail_in_one_frame_is_clipped_once_pixels_have_enough_samples(self):
        stack = LiveStack()
        for index in range(13):
            stack.add(sky(seed=index, trail=index == 11), f"frame{index}.fit")
        row = stack.image()[120, 30:290]
        self.assertLess(abs(float(np.median(row)) - 500), 20)
        # Plain noise loses only the odd sample in the three frames clipped
        self.assertLess(stack.rejected, 0.03 * 3 * stack.count.size)

    def test_a_frame_that_cannot_be_aligned_leaves_the_stack_unchanged(self):
        stack = LiveStack()
        stack.add(sky(seed=1), "frame1.fit")
        before = stack.image().copy()
        blank = np.random.default_rng(2).normal(500, 5, before.shape).astype(np.float32)
        with self.assertRaises(Exception):
            stack.add(blank, "cloud.fit")
        self.assertEqual(len(stack), 1)
        np.testing.assert_array_equal(stack.image(), before)

    def test_a_clouded_first_frame_does_not_become_the_reference(self):
        stack = LiveStack()
        cloud = np.random.default_rng(1).normal(500, 5, (240, 320)).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "Too few stars for a reference frame"):
            stack.add(cloud, "cloud.fit")
        self.assertIsNone(stack.reference)
        # The next frames with stars still stack, the first of them as the reference
        stack.add(sky(seed=1), "frame1.fit")
        stack.add(sky(3, -2, seed=2), "frame2.fit")
        self.assertEqual([path.name for path in stack.paths], ["frame1.fit", "frame2.fit"])

    def test_the_stack_is_saved_with_its_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            stack = LiveStack()
            for index, (dx, dy) in enumerate([(0, 0), (2, 1), (-1, 3)]):
                image = sky(dx, dy, seed=index)
                stack.add(image, write_frame(temporary, f"frame{index}.fit", image))
            fits_path, png_path = Path(temporary) / "live-stack.fit", Path(temporary) / "live-stack.png"
            stack.write(fits_path, png_path)
            with fits.open(fits_path) as hdus:
                self.assertEqual(hdus[0].header["NCOMBINE"], 3)
                self.assertEqual(hdus[0].header["STACKMTH"], "LIVE")
                self.assertEqual(hdus["COVERAGE"].data.max(), 3)
            self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
