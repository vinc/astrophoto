#!/usr/bin/env python3

import argparse
from pathlib import Path

import astroalign
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

def load(path):
    data = fits.getdata(path).astype(np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D Bayer FITS: {path}")

    # Average each Bayer 2x2 block into one grayscale pixel
    h, w = data.shape
    data = data[:h // 2 * 2, :w // 2 * 2]

    return data.reshape(
        h // 2, 2, w // 2, 2
    ).mean(axis=(1, 3))

def stack_directory(directory):
    files = sorted(
        path for path in directory.glob("*.fit")
        if not path.name.endswith("-stack.fit")
    )

    if not files:
        raise SystemExit(f"No FITS images found in {directory}")

    reference = load(files[0])
    aligned = [reference]

    for path in files[1:]:
        image = load(path)

        try:
            registered, footprint = astroalign.register(
                image, reference
            )
            registered[footprint] = np.nan
            aligned.append(registered)
            print(f"Aligned {path.name}")
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")

    # Median stacking rejects outliers such as hot pixels
    with np.errstate(all="ignore"):
        stack = np.nanmedian(
            np.stack(aligned), axis=0
        )

    prefix = directory.name
    fits_path = directory / f"{prefix}-stack.fit"
    png_path = directory / f"{prefix}-stack.png"

    fits.writeto(
        fits_path,
        stack.astype(np.float32),
        overwrite=True,
    )

    # Contrast stretch for visual inspection
    lo, hi = np.nanpercentile(stack, [5, 99.7])

    plt.imsave(
        png_path,
        np.clip((stack - lo) / (hi - lo), 0, 1),
        cmap="gray",
    )

    print(f"Stacked {len(aligned)} images")
    print(f"Created {fits_path}")
    print(f"Created {png_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Align and median-stack FITS images"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the FITS images",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()

    if not directory.is_dir():
        parser.error(f"Not a directory: {directory}")

    stack_directory(directory)

if __name__ == "__main__":
    main()
