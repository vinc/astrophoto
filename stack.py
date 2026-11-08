#!/usr/bin/env python3

import argparse
from pathlib import Path
import re

import astroalign
import numpy as np
from astropy.io import fits
from astropy.time import Time
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

def stack_header(reference, files):
    header = fits.getheader(reference)

    # Data is rewritten as unscaled float32 2x2-binned pixels
    for key in ("BZERO", "BSCALE", "RAWFILE"):
        header.remove(key, ignore_missing=True)

    # Registration and binning invalidate the reference frame's coordinates
    coordinates = re.compile(
        r"(?:WCSAXES|WCSNAME|LONPOLE|LATPOLE|RADESYS|RADECSYS|EQUINOX)[A-Z]?|"
        r"(?:CTYPE|CUNIT|CRPIX|CRVAL|CDELT|CROTA|CNAME|CRDER|CSYER)\d+[A-Z]?|"
        r"(?:PC|CD|PV|PS)\d+_\d+[A-Z]?|(?:A|B|AP|BP)_(?:ORDER|DMAX|\d+_\d+)|"
        r"(?:CPDIS|CQDIS|D2IMDIS)\d+|(?:DP|DQ|D2IM)\d+(?:\..*)?"
    )
    for key in list(header):
        if coordinates.fullmatch(key):
            header.remove(key, ignore_missing=True, remove_all=True)

    header["NCOMBINE"] = (len(files), "Number of frames stacked")

    times = []
    for path in files:
        frame = fits.getheader(path)
        if "DATE-OBS" in frame and "DATE-AVG" in frame:
            times.append((
                Time(frame["DATE-OBS"], format="isot", scale="utc"),
                Time(frame["DATE-AVG"], format="isot", scale="utc"),
            ))

    if times:
        start = min(t for t, _ in times)
        mean = Time(np.mean([t.jd for _, t in times]),
                    format="jd", scale="utc")
        header["DATE-OBS"] = (start.isot, "First exposure start (UTC)")
        header["DATE-AVG"] = (mean.isot, "Mean exposure midpoint (UTC)")

    return header

def stack_directory(directory):
    files = sorted(
        path for path in directory.glob("*.fit")
        if not path.name.endswith("-stack.fit")
    )

    if not files:
        raise SystemExit(f"No FITS images found in {directory}")

    # The field drifts on a fixed tripod, so align to the middle
    # frame to halve the largest offset between frames
    middle = files[len(files) // 2]
    reference = load(middle)
    aligned = [reference]
    used = [middle]

    for path in files:
        if path == middle:
            continue

        image = load(path)

        try:
            registered, footprint = astroalign.register(
                image, reference
            )
            registered[footprint] = np.nan
            aligned.append(registered)
            used.append(path)
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
        header=stack_header(middle, used),
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
