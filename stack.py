#!/usr/bin/env python3

import argparse
from pathlib import Path
import re

import astroalign
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip, sigma_clipped_stats
from astropy.time import Time
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter

from camera import stack_display_limits

DEBAND_WINDOW = 65

def read_frame(path):
    data, header = fits.getdata(path, header=True)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D Bayer FITS: {path}")
    if "NCOMBINE" in header or "STACKMTH" in header:
        raise ValueError(f"Expected an unstacked Bayer FITS: {path}")

    return data.astype(np.float32), header

def check_dark_match(path, data, header, shape, metadata):
    metadata = metadata or {}
    if data.shape != shape:
        raise ValueError(f"{path}: image shape {data.shape} does not match dark shape {shape}")
    for key in ("INSTRUME", "ISO", "EXPTIME"):
        if key in header and key in metadata and header[key] != metadata[key]:
            raise ValueError(f"{path}: {key}={header[key]} does not match dark {metadata[key]}")
    if header.get("DARKCOR", False):
        raise ValueError(f"{path}: dark correction has already been applied")

def master_dark(paths):
    images = []
    metadata = {}
    for path in paths:
        data, header = read_frame(path)
        shape = images[0].shape if images else data.shape
        check_dark_match(path, data, header, shape, metadata)
        if not np.isfinite(data).all():
            raise ValueError(f"{path}: dark frame contains non-finite pixels")
        for key in ("INSTRUME", "ISO", "EXPTIME"):
            if key in header:
                metadata[key] = header[key]
        images.append(data)

    # Detector patterns stay fixed: combine without registration or normalization
    dark, _, _ = combine_frames(images, method="median")
    return dark, metadata

def reduce_banding(image, amount):
    if not np.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError("--deband must be between 0 and 1")
    if amount == 0 or image.shape[0] < 3:
        return image

    values = np.ma.masked_invalid(image)
    level = np.ma.median(values, axis=1).filled(np.nan)
    noise = mad_std(values, axis=1).filled(np.nan)
    bright = np.isfinite(image) & (image > (level + 3 * noise)[:, None])
    # Grow the bright-pixel mask to exclude stellar wings as well as cores
    mask = ~np.isfinite(image) | binary_dilation(bright, iterations=2)
    sky = np.ma.array(image, mask=mask)
    profile = np.ma.median(sky, axis=1).filled(np.nan).astype(float)
    valid = np.isfinite(profile) & (sky.count(axis=1) >= max(8, image.shape[1] // 4))
    if valid.sum() < 3:
        return image

    rows = np.arange(image.shape[0])
    profile = np.interp(rows, rows[valid], profile[valid])
    window = min(DEBAND_WINDOW, (len(rows) - 1) // 2 * 2 + 1)
    # A quadratic smoothing filter preserves broad gradients, including at edges
    baseline = savgol_filter(profile, window, polyorder=2, mode="interp")
    correction = profile - baseline
    correction[~valid] = 0
    correction[valid] -= correction[valid].mean()
    return image - (amount * correction).astype(np.float32)[:, None]

def load(path, dark=None, dark_metadata=None, deband=0):
    data, header = read_frame(path)
    if dark is not None:
        check_dark_match(path, data, header, dark.shape, dark_metadata)
        # The raw master includes the bias; subtract it once and retain negatives
        data -= dark

    # Average each Bayer 2x2 block into one grayscale pixel
    h, w = data.shape
    data = data[:h // 2 * 2, :w // 2 * 2]

    binned = data.reshape(
        h // 2, 2, w // 2, 2
    ).mean(axis=(1, 3))
    return reduce_banding(binned, deband)

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

def normalize_background(image, reference):
    overlap = np.isfinite(image) & np.isfinite(reference)
    if not overlap.any():
        raise ValueError("No finite overlap with the reference frame")
    # Compare the same sky area, excluding stars and isolated outliers
    _, background, _ = sigma_clipped_stats(image[overlap], sigma=3, maxiters=5)
    _, target, _ = sigma_clipped_stats(reference[overlap], sigma=3, maxiters=5)
    offset = float(target - background)
    return image + offset, offset

def clipping_std(values, axis=None):
    values = np.ma.masked_invalid(values, copy=False)
    deviation = mad_std(values, axis=axis).filled(0)
    scale = np.ma.max(np.ma.abs(values), axis=axis).filled(0)
    # Identical quantized samples can have zero MAD despite float32 rounding
    tolerance = 8 * np.finfo(np.float32).eps * np.maximum(scale, 1)
    return np.maximum(deviation, tolerance)

def combine_frames(images, method="mean", sigma=3):
    shape = images[0].shape
    result = np.full(shape, np.nan, dtype=np.float32)
    coverage = np.zeros(shape, dtype=np.int32)
    accepted = np.zeros(shape, dtype=np.int32)

    # Limit temporary cubes to a strip rather than copying the whole sequence
    for y in range(0, shape[0], 128):
        rows = slice(y, y + 128)
        values = np.ma.masked_invalid(np.stack([image[rows] for image in images]))
        counts = values.count(axis=0)
        coverage[rows] = counts
        if method == "mean":
            clipped = sigma_clip(
                values, sigma=sigma, maxiters=5, axis=0,
                cenfunc="median", stdfunc=clipping_std, copy=True,
            )
            # Sparse borders do not have enough samples for reliable clipping
            clipped.mask = np.where(counts >= 5, clipped.mask, values.mask)
            combined = clipped.mean(axis=0)
            accepted[rows] = clipped.count(axis=0)
        else:
            combined = np.ma.median(values, axis=0)
            accepted[rows] = counts
        result[rows] = combined.filled(np.nan)

    return result, coverage, accepted

def coverage_crop(coverage, minimum):
    # Largest rectangle whose pixels all meet the coverage threshold
    heights = np.zeros(coverage.shape[1], dtype=int)
    best_area = 0
    best = None
    for y, row in enumerate(coverage >= minimum):
        heights = np.where(row, heights + 1, 0)
        columns = []
        for x in range(len(heights) + 1):
            height = int(heights[x]) if x < len(heights) else 0
            start = x
            while columns and columns[-1][1] > height:
                start, previous = columns.pop()
                area = previous * (x - start)
                if area > best_area:
                    best_area = area
                    best = (slice(y + 1 - previous, y + 1), slice(start, x))
            if height and (not columns or columns[-1][1] < height):
                columns.append((start, height))
    if best is None:
        raise ValueError("No pixels meet the requested coverage")
    return best

def stack_directory(directory, method="mean", normalize=True, sigma=3,
                    min_coverage=0, output_prefix=None, dark_files=None, deband=0):
    dark_files = list(dict.fromkeys(Path(path).resolve() for path in (dark_files or [])))
    dark_paths = set(dark_files)
    files = sorted(
        path for path in directory.glob("*.fit")
        if not path.name.endswith("-stack.fit") and path.resolve() not in dark_paths
    )

    if not files:
        raise SystemExit(f"No FITS images found in {directory}")

    dark, dark_metadata = master_dark(dark_files) if dark_files else (None, None)
    if dark_files:
        print(f"Combined {len(dark_files)} dark frames")

    # The field drifts on a fixed tripod, so align to the middle
    # frame to halve the largest offset between frames
    middle = files[len(files) // 2]
    reference = load(middle, dark, dark_metadata, deband=deband)
    if not np.isfinite(reference).any():
        raise ValueError(f"No finite pixels in reference frame: {middle}")
    aligned = [reference]
    used = [middle]
    offsets = [0.0]

    for path in files:
        if path == middle:
            continue

        image = load(path, dark, dark_metadata, deband=deband)

        try:
            registered, footprint = astroalign.register(
                image, reference
            )
            registered[footprint] = np.nan
            offset = 0.0
            if normalize:
                registered, offset = normalize_background(registered, reference)
            aligned.append(registered)
            used.append(path)
            offsets.append(offset)
            print(f"Aligned {path.name}")
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")

    stack, coverage, accepted = combine_frames(aligned, method, sigma)
    rows, columns = slice(0, stack.shape[0]), slice(0, stack.shape[1])
    if min_coverage > 0:
        minimum = int(np.ceil(min_coverage * len(aligned)))
        rows, columns = coverage_crop(coverage, minimum)
        stack = stack[rows, columns]
        coverage = coverage[rows, columns]
        accepted = accepted[rows, columns]

    prefix = output_prefix or directory.name
    fits_path = directory / f"{prefix}-stack.fit"
    png_path = directory / f"{prefix}-stack.png"
    coverage_path = directory / f"{prefix}-coverage.png"

    header = stack_header(middle, used)
    if dark_files:
        header["DARKCOR"] = (True, "Master dark subtracted before binning")
        header["NDARKS"] = (len(dark_files), "Number of median-combined dark frames")
        # camera.read_image must not subtract the raw bias a second time
        header["BLACKLVL"] = (0.0, "Raw bias removed with master dark")
        header.remove("WHITELEV", ignore_missing=True)
        for path in dark_files:
            header.add_history(f"Dark {path.name}")
    header["STACKMTH"] = "SIGMEAN" if method == "mean" else "MEDIAN"
    header["BGNORM"] = (normalize, "Additive background normalization")
    header["DEBAND"] = (deband, "Row correction fraction before alignment")
    if deband:
        header["DEBWIN"] = (min(DEBAND_WINDOW, (reference.shape[0] - 1) // 2 * 2 + 1),
                            "Smoothing window in binned detector rows")
    if method == "mean":
        header["CLIPSIG"] = (sigma, "Sigma threshold, median/MAD clipping")
        header["CLIPITER"] = (5, "Maximum clipping iterations")
    header["MINCOV"] = (min_coverage, "Minimum input coverage fraction for crop")
    header["CROPX"] = (columns.start, "Crop origin in binned reference pixels")
    header["CROPY"] = (rows.start, "Crop origin in binned reference pixels")
    for path, offset in zip(used, offsets):
        header.add_history(f"Input {path.name}; background offset {offset:.6g} ADU")

    lo, hi = stack_display_limits(stack)
    header["DISPLO"] = (lo, "Black point used for PNG previews")
    header["DISPHI"] = (hi, "White point used for PNG previews")

    count_header = fits.Header({"BUNIT": "count"})
    fits.HDUList([
        fits.PrimaryHDU(stack, header),
        fits.ImageHDU(coverage, count_header, name="COVERAGE"),
        fits.ImageHDU(accepted, count_header, name="ACCEPTED"),
    ]).writeto(fits_path, overwrite=True)

    plt.imsave(
        png_path, stack, cmap="gray", vmin=lo, vmax=hi,
    )
    plt.imsave(coverage_path, coverage, cmap="gray", vmin=0, vmax=len(aligned))

    print(f"Stacked {len(aligned)} images")
    print(f"Created {fits_path}")
    print(f"Created {png_path}")
    print(f"Created {coverage_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Align, normalize backgrounds, and sigma-clip mean-stack FITS images"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the FITS images",
    )
    parser.add_argument("--method", choices=["mean", "median"], default="mean",
                        help="Combination method (default: sigma-clipped mean)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable additive background normalization")
    parser.add_argument("--sigma", type=float, default=3,
                        help="Clipping threshold for mean stacking (default: 3)")
    parser.add_argument("--min-coverage", type=float, default=0,
                        help="Crop to this fraction of aligned frames; 0 keeps the full field (default: 0)")
    parser.add_argument("--output-prefix",
                        help="Output basename within the input directory (default: directory name)")
    parser.add_argument("--dark", nargs="+", type=Path,
                        help="Unstacked dark FITS files; median-combine and subtract before binning")
    parser.add_argument("--deband", type=float, default=0, metavar="AMOUNT",
                        help="Horizontal row correction strength, 0 to 1 (default: 0, disabled)")
    args = parser.parse_args()
    if not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma must be finite and positive")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be between 0 and 1")
    if not np.isfinite(args.deband) or not 0 <= args.deband <= 1:
        parser.error("--deband must be between 0 and 1")
    if args.output_prefix is not None and (
        not args.output_prefix or Path(args.output_prefix).name != args.output_prefix
        or args.output_prefix in {".", ".."}
    ):
        parser.error("--output-prefix must be a basename")

    directory = args.directory.resolve()

    if not directory.is_dir():
        parser.error(f"Not a directory: {directory}")

    try:
        stack_directory(directory, args.method, not args.no_normalize, args.sigma,
                        args.min_coverage, args.output_prefix, dark_files=args.dark,
                        deband=args.deband)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

if __name__ == "__main__":
    main()
