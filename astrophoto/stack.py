import argparse
from pathlib import Path
import re
import sys

import astroalign
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip, sigma_clipped_stats
from astropy.time import Time
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter
from skimage.transform import warp

from astrophoto.lib.camera import stack_display_limits
from astrophoto.lib.color import BAYER_PATTERNS, bin_rgb, color_metadata, luminance, rgb_display_header, render_rgb, saturation_flags

DEBAND_WINDOW = 65

def read_frame(path):
    data, header = fits.getdata(path, header=True)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D Bayer FITS: {path}")
    if "NCOMBINE" in header or "STACKMTH" in header:
        raise ValueError(f"Expected an unstacked Bayer FITS: {path}")

    return data.astype(np.float32), header

def check_calibration_match(path, data, header, shape, metadata, kind, keys):
    metadata = metadata or {}
    if data.shape != shape:
        raise ValueError(f"{path}: image shape {data.shape} does not match {kind} shape {shape}")
    for key in keys:
        if key in header and key in metadata and header[key] != metadata[key]:
            raise ValueError(f"{path}: {key}={header[key]} does not match {kind} {metadata[key]}")

def check_dark_match(path, data, header, shape, metadata):
    check_calibration_match(path, data, header, shape, metadata, "dark",
                            ("INSTRUME", "ISO", "EXPTIME"))
    if header.get("DARKCOR", False):
        raise ValueError(f"{path}: dark correction has already been applied")

FLAT_KEYS = ("INSTRUME", "ISO", "OBJECTIV", "FOCALLEN", "FNUMBER")

def check_flat_match(path, data, header, shape, metadata):
    check_calibration_match(path, data, header, shape, metadata, "flat", FLAT_KEYS)
    if header.get("FLATCOR", False) or header.get("DARKCOR", False):
        raise ValueError(f"{path}: calibration has already been applied")

def black_level(path, header):
    if "BLACKLVL" not in header:
        raise ValueError(f"{path}: missing BLACKLVL; convert the original RAW image first")
    level = float(header["BLACKLVL"])
    if not np.isfinite(level) or level < 0:
        raise ValueError(f"{path}: invalid BLACKLVL")
    return level

def normalize_flat(data):
    # Keep the flat illumination's colour out of the four Bayer pixel positions
    for y in range(2):
        for x in range(2):
            plane = data[y::2, x::2]
            level = float(np.median(plane))
            if not np.isfinite(level) or level <= 0:
                raise ValueError("Flat frame has no positive signal in a Bayer plane")
            plane /= level
    return data

def master_flat(paths):
    images = []
    metadata = {}
    for path in paths:
        data, header = read_frame(path)
        shape = images[0].shape if images else data.shape
        check_flat_match(path, data, header, shape, metadata)
        if min(data.shape) < 2 or not np.isfinite(data).all():
            raise ValueError(f"{path}: flat frame needs finite pixels in all Bayer planes")
        if "WHITELEV" in header and np.percentile(data, 99) >= 0.98 * header["WHITELEV"]:
            raise ValueError(f"{path}: flat frame is saturated; use a shorter exposure")
        # Short flats use their own RAW black level, not the long-exposure dark
        data -= black_level(path, header)
        if "WHITELEV" in header:
            span = float(header["WHITELEV"]) - float(header["BLACKLVL"])
            if not np.isfinite(span) or span <= 0:
                raise ValueError(f"{path}: invalid WHITELEV; must exceed BLACKLVL")
            level = min(float(np.median(data[y::2, x::2]))
                        for y in range(2) for x in range(2)) / span
            print(f"{path.name}: weakest Bayer plane at {level:.0%} of full scale")
            if level < 0.15:
                print(f"Warning: {path}: flat frame may be underexposed; "
                      "consider a longer exposure", file=sys.stderr)
        images.append(normalize_flat(data))
        for key in FLAT_KEYS:
            if key in header:
                metadata[key] = header[key]

    flat, _, _ = combine_frames(images, method="median")
    if not np.isfinite(flat).all() or np.any(flat <= 0):
        raise ValueError("Master flat contains invalid or non-positive response pixels")
    return normalize_flat(flat), metadata

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
    if image.ndim == 3:
        return np.stack([reduce_banding(image[..., channel], amount)
                         for channel in range(3)], axis=-1)

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

def load(path, dark=None, dark_metadata=None, deband=0, flat=None, flat_metadata=None,
         rgb=False, bayer_pattern=None, return_saturation=False):
    data, header = read_frame(path)
    if rgb:
        metadata = color_metadata(path, header, bayer_pattern)
        if return_saturation:
            # Detect sensor clipping before dark/flat correction changes the levels
            saturated = saturation_flags(data, header, metadata["BAYERPAT"])
    if dark is not None:
        check_dark_match(path, data, header, dark.shape, dark_metadata)
        # The raw master includes the bias; subtract it once and retain negatives
        data -= dark
    if flat is not None:
        check_flat_match(path, data, header, flat.shape, flat_metadata)
        if dark is None:
            data -= black_level(path, header)
        data /= flat

    if rgb:
        if dark is None and flat is None:
            data -= black_level(path, header)
        image = reduce_banding(bin_rgb(data, metadata["BAYERPAT"]), deband)
        return (image, saturated) if return_saturation else image

    h, w = data.shape
    binned = data[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return reduce_banding(binned, deband)

def stack_header(reference, files):
    header = fits.getheader(reference)

    # Data is rewritten as unscaled float32 2x2-binned pixels
    for key in ("BZERO", "BSCALE", "RAWFILE", "BAYERPAT"):
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
    if image.ndim == 3:
        channels = [normalize_background(image[..., channel], reference[..., channel])
                    for channel in range(3)]
        return np.stack([data for data, _ in channels], axis=-1), [offset for _, offset in channels]
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

def align_saturation(flags, transform, shape, footprint):
    result = np.zeros(shape, dtype=np.uint8)
    for bit in (1, 2, 4):
        mask = (flags & bit) != 0
        if not mask.any():
            continue
        # Include the full 4x4 source support of cubic image interpolation
        mask = binary_dilation(mask, structure=np.ones((3, 3)))
        aligned = warp(mask.astype(np.float32), inverse_map=transform.inverse,
                       output_shape=shape, order=1, mode="constant", cval=0,
                       preserve_range=True)
        result[aligned > 0] |= bit
    result[footprint] = 0
    return result

def combine_frames(images, method="mean", sigma=3, saturation=None):
    shape = images[0].shape
    result = np.full(shape, np.nan, dtype=np.float32)
    coverage = np.zeros(shape, dtype=np.int32)
    accepted = np.zeros(shape, dtype=np.int32)
    saturated = np.zeros(shape, dtype=np.int32) if saturation is not None else None

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
            contributors = clipped
        else:
            combined = np.ma.median(values, axis=0)
            accepted[rows] = counts
            contributors = values
        result[rows] = combined.filled(np.nan)
        if saturation is not None:
            flags = np.stack([mask[rows] for mask in saturation])
            channels = (flags[..., None] & np.array([1, 2, 4], dtype=np.uint8)) != 0
            saturated[rows] = np.sum(channels & ~np.ma.getmaskarray(contributors), axis=0)

    if saturation is not None:
        fraction = np.divide(saturated, accepted, out=np.zeros(shape, dtype=np.float32), where=accepted > 0)
        return result, coverage, accepted, fraction.max(axis=-1)
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
                    min_coverage=0, output_prefix=None, dark_files=None, deband=0,
                    flat_files=None, rgb=False, bayer_pattern=None):
    dark_files = list(dict.fromkeys(Path(path).resolve() for path in (dark_files or [])))
    flat_files = list(dict.fromkeys(Path(path).resolve() for path in (flat_files or [])))
    if set(dark_files) & set(flat_files):
        raise ValueError("A frame cannot be supplied as both a dark and a flat")
    calibration_paths = set(dark_files) | set(flat_files)
    files = sorted(
        path for path in directory.glob("*.fit")
        if not path.name.endswith("-stack.fit") and path.resolve() not in calibration_paths
    )

    if not files:
        raise SystemExit(f"No FITS images found in {directory}")

    dark, dark_metadata = master_dark(dark_files) if dark_files else (None, None)
    if dark_files:
        print(f"Combined {len(dark_files)} dark frames")
    flat, flat_metadata = master_flat(flat_files) if flat_files else (None, None)
    if flat_files:
        print(f"Combined {len(flat_files)} flat frames")

    # The field drifts on a fixed tripod, so align to the middle
    # frame to halve the largest offset between frames
    middle = files[len(files) // 2]
    colour = color_metadata(middle, fits.getheader(middle), bayer_pattern) if rgb else None
    colour_options = {"rgb": True, "bayer_pattern": bayer_pattern, "return_saturation": True} if rgb else {}
    reference = load(middle, dark, dark_metadata, deband=deband,
                     flat=flat, flat_metadata=flat_metadata, **colour_options)
    if rgb:
        reference, reference_saturation = reference
        saturation = [reference_saturation]
    if not np.isfinite(reference).any():
        raise ValueError(f"No finite pixels in reference frame: {middle}")
    aligned = [reference]
    used = [middle]
    offsets = [[0.0] * 3 if rgb else 0.0]

    for path in files:
        if path == middle:
            continue

        image = load(path, dark, dark_metadata, deband=deband,
                     flat=flat, flat_metadata=flat_metadata, **colour_options)
        if rgb:
            image, image_saturation = image

        try:
            if rgb:
                transform, _ = astroalign.find_transform(luminance(image), luminance(reference))
                channels = [astroalign.apply_transform(transform, image[..., channel], reference[..., channel])
                            for channel in range(3)]
                registered = np.stack([data for data, _ in channels], axis=-1)
                footprint = np.logical_or.reduce([mask for _, mask in channels])
                registered_saturation = align_saturation(image_saturation, transform, reference.shape[:2], footprint)
            else:
                registered, footprint = astroalign.register(image, reference)
            registered[footprint] = np.nan
            offset = [0.0] * 3 if rgb else 0.0
            if normalize:
                registered, offset = normalize_background(registered, reference)
            aligned.append(registered)
            used.append(path)
            offsets.append(offset)
            if rgb:
                saturation.append(registered_saturation)
            print(f"Aligned {path.name}")
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")

    if rgb:
        stack, coverage, accepted, saturation = combine_frames(aligned, method, sigma, saturation=saturation)
        coverage = coverage.min(axis=-1)
    else:
        stack, coverage, accepted = combine_frames(aligned, method, sigma)
    rows, columns = slice(0, stack.shape[0]), slice(0, stack.shape[1])
    if min_coverage > 0:
        minimum = int(np.ceil(min_coverage * len(aligned)))
        rows, columns = coverage_crop(coverage, minimum)
        stack = stack[rows, columns]
        coverage = coverage[rows, columns]
        accepted = accepted[rows, columns]
        if rgb:
            saturation = saturation[rows, columns]

    prefix = output_prefix or (f"{directory.name}-rgb" if rgb else directory.name)
    fits_path = directory / f"{prefix}-stack.fit"
    png_path = directory / f"{prefix}-stack.png"
    coverage_path = directory / f"{prefix}-coverage.png"

    header = stack_header(middle, used)
    if dark_files:
        header["DARKCOR"] = (True, "Master dark subtracted before binning")
        header["NDARKS"] = (len(dark_files), "Number of median-combined dark frames")
        for path in dark_files:
            header.add_history(f"Dark {path.name}")
    if flat_files:
        header["FLATCOR"] = (True, "Master flat applied before binning")
        header["NFLATS"] = (len(flat_files), "Number of normalized flat frames")
        header["FLATBIAS"] = ("BLACKLVL", "Flat bias from RAW black-level metadata")
        for path in flat_files:
            header.add_history(f"Flat {path.name}")
    if dark_files or flat_files or rgb:
        # camera.read_image must not subtract the raw bias a second time
        header["BLACKLVL"] = (0.0, "Raw bias removed during calibration")
        header.remove("WHITELEV", ignore_missing=True)
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
        text = ", ".join(f"{value:.6g}" for value in offset) if rgb else f"{offset:.6g}"
        header.add_history(f"Input {path.name}; background offset {text} ADU")

    primary = luminance(stack) if rgb else stack
    if rgb:
        header["RGB"] = (True, "Linear colour data in RGB extension")
        header["RAWPATT"] = (colour["BAYERPAT"], "Reference RAW Bayer layout")
    lo, hi = stack_display_limits(primary)
    header["DISPLO"] = (lo, "Black point for the rendered PNG")
    header["DISPHI"] = (hi, "White point for the rendered PNG")

    count_header = fits.Header({"BUNIT": "count"})
    accepted_header = count_header.copy()
    if rgb:
        accepted_header["COLORSPC"] = "RGB"
    hdus = [
        fits.PrimaryHDU(primary, header),
        fits.ImageHDU(coverage, count_header, name="COVERAGE"),
        fits.ImageHDU(np.moveaxis(accepted, -1, 0) if rgb else accepted, accepted_header, name="ACCEPTED"),
    ]
    if rgb:
        colour_header = rgb_display_header(stack, colour)
        colour_header["BUNIT"] = "ADU"
        colour_header["SATEXT"] = ("SATURATED", "Accepted sensor clipping fractions")
        hdus.append(fits.ImageHDU(np.moveaxis(stack, -1, 0), colour_header, name="RGB"))
        saturation_header = fits.Header({"BUNIT": "fraction", "SATTHR": 0.98})
        saturation_header.add_comment("Maximum accepted clipping fraction across R, G and B")
        hdus.append(fits.ImageHDU(saturation, saturation_header, name="SATURATED"))
    fits.HDUList(hdus).writeto(fits_path, overwrite=True)

    if rgb:
        plt.imsave(png_path, render_rgb(stack, colour_header, saturation))
    else:
        plt.imsave(png_path, stack, cmap="gray", vmin=lo, vmax=hi)
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
    parser.add_argument("--flat", nargs="+", type=Path,
                        help="Unstacked flat FITS files; normalize and divide before binning")
    parser.add_argument("--deband", type=float, default=0, metavar="AMOUNT",
                        help="Horizontal row correction strength, 0 to 1 (default: 0, disabled)")
    parser.add_argument("--rgb", action="store_true",
                        help="Keep RGB colour at the usual 2x2-binned size; save with an -rgb prefix")
    parser.add_argument("--bayer-pattern", choices=BAYER_PATTERNS,
                        help="Bayer layout override for older FITS files; requires --rgb")
    args = parser.parse_args()
    if not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma must be finite and positive")
    if args.bayer_pattern is not None and not args.rgb:
        parser.error("--bayer-pattern requires --rgb")
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
                        deband=args.deband, flat_files=args.flat, rgb=args.rgb,
                        bayer_pattern=args.bayer_pattern)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

if __name__ == "__main__":
    main()
