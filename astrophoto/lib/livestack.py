"""Stack frames one at a time as they arrive, for a preview that improves during a series."""
from pathlib import Path

import astroalign
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

from astrophoto.lib.camera import stack_display_limits
from astrophoto.stack import normalize_background, stack_header

# Pixels need this many samples before a value far from their running mean is rejected: a spread estimated
# from fewer would also reject about one sample in twenty of plain noise
MIN_CLIP_FRAMES = 10
# Smallest spread, in ADU, assumed for a pixel, so a few identical early values do not reject every later one
MIN_SPREAD = 1.0


def find_stars(image):
    """Return the positions of the brightest stars in an image, as astroalign uses them for matching."""
    # Detecting the reference stars once spares that work on every later frame
    finder = getattr(astroalign, "_find_sources", None)
    return image if finder is None else finder(image)


class LiveStack:
    """A sigma-clipped running mean of frames aligned to the first one.

    The reference is the first frame, as later frames are not known yet; the offline stack command can still
    align a finished series to its middle frame. Each pixel keeps a running mean and variance (Welford's method)
    and, once it has MIN_CLIP_FRAMES samples, ignores values more than sigma deviations from its mean, which drops
    satellite trails, planes and hot pixels without keeping every frame in memory.
    """

    def __init__(self, sigma=3.0):
        self.sigma = sigma
        self.reference = None
        self.reference_stars = None
        self.mean = None
        self.m2 = None
        self.count = None
        self.paths = []
        self.rejected = 0

    def __len__(self):
        return len(self.paths)

    def add(self, image, path):
        """Align a frame, add it to the stack, and return its shift in pixels and rotation in degrees.

        Raise an exception, leaving the stack unchanged, when the frame cannot be aligned.
        """
        image = np.asarray(image, dtype=np.float32)
        if not np.isfinite(image).any():
            raise ValueError("No finite pixels")
        if self.reference is None:
            registered, shift, rotation = image.copy(), (0.0, 0.0), 0.0
            stars = find_stars(image)
        else:
            transform, _ = astroalign.find_transform(find_stars(image), self.reference_stars)
            registered, footprint = astroalign.apply_transform(transform, image, self.reference)
            registered = registered.astype(np.float32)
            registered[footprint] = np.nan
            registered, _ = normalize_background(registered, self.reference)
            shift = tuple(float(value) for value in transform.translation)
            rotation = float(np.degrees(transform.rotation))
        if self.reference is None:
            self.reference, self.reference_stars = image, stars
            self.mean = np.zeros(image.shape, dtype=np.float64)
            self.m2 = np.zeros(image.shape, dtype=np.float64)
            self.count = np.zeros(image.shape, dtype=np.int32)
        self._accumulate(registered)
        self.paths.append(Path(path))
        return {"shift": shift, "rotation": rotation}

    def _accumulate(self, values):
        valid = np.isfinite(values)
        clipping = valid & (self.count >= MIN_CLIP_FRAMES)
        if clipping.any():
            spread = np.sqrt(self.m2[clipping] / (self.count[clipping] - 1))
            far = np.abs(values[clipping] - self.mean[clipping]) > self.sigma * np.maximum(spread, MIN_SPREAD)
            rows, columns = np.nonzero(clipping)
            valid[rows[far], columns[far]] = False
            self.rejected += int(far.sum())
        self.count[valid] += 1
        delta = values[valid] - self.mean[valid]
        self.mean[valid] += delta / self.count[valid]
        self.m2[valid] += delta * (values[valid] - self.mean[valid])

    def image(self):
        """Return the stacked image, NaN where no frame covers it."""
        if self.mean is None:
            return None
        return np.where(self.count > 0, self.mean, np.nan).astype(np.float32)

    def write(self, fits_path, png_path):
        """Save the stack as a FITS file with a coverage extension, and as a PNG."""
        stack = self.image()
        # The first frame is the reference, so its header describes the stack's pixel grid
        header = stack_header(self.paths[0], self.paths)
        header["STACKMTH"] = ("LIVE", "Running clipped mean, first frame reference")
        header["CLIPSIG"] = (self.sigma, "Sigma threshold, running mean and spread")
        header["BGNORM"] = (True, "Additive background normalization")
        low, high = stack_display_limits(stack)
        header["DISPLO"] = (low, "Black point for the rendered PNG")
        header["DISPHI"] = (high, "White point for the rendered PNG")
        for path in self.paths:
            header.add_history(f"Input {path.name}")
        coverage = fits.ImageHDU(self.count, fits.Header({"BUNIT": "count"}), name="COVERAGE")
        fits.HDUList([fits.PrimaryHDU(stack, header), coverage]).writeto(fits_path, overwrite=True)
        plt.imsave(png_path, stack, cmap="gray", vmin=low, vmax=high)
