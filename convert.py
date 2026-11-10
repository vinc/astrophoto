#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import exifread
import rawpy
from astropy.io import fits

from color import raw_color_metadata

def exif_value(tags, key):
    tag = tags.get(key)
    return str(tag) if tag is not None else None


def exif_float(tags, key):
    value = exif_value(tags, key)
    return float(Fraction(value)) if value is not None else None


def convert(path, tz, output=None):
    with rawpy.imread(str(path)) as raw:
        # Preserve the original Bayer mosaic without demosaicing
        data = raw.raw_image_visible.copy()
        black_levels = list(raw.black_level_per_channel)
        white_level = int(raw.white_level)
        colour = raw_color_metadata(raw)

    with path.open("rb") as f:
        tags = exifread.process_file(f, details=False)

    date = exif_value(tags, "EXIF DateTimeOriginal")
    exposure = exif_float(tags, "EXIF ExposureTime")

    if date is None or exposure is None:
        raise ValueError(f"{path}: missing exposure timestamp or duration")

    local = datetime.strptime(
        date, "%Y:%m:%d %H:%M:%S"
    ).replace(tzinfo=tz)

    start = local.astimezone(timezone.utc)
    midpoint = start + timedelta(seconds=exposure / 2)

    hdu = fits.PrimaryHDU(data)
    hdr = hdu.header

    hdr["BUNIT"] = ("ADU", "Raw sensor values")
    hdr["RAWFILE"] = (path.name, "Original CR2 filename")

    hdr["TIMESYS"] = ("UTC", "Time reference system")
    hdr["DATE-OBS"] = (
        start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "Exposure start (UTC)",
    )
    hdr["DATE-AVG"] = (
        midpoint.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "Exposure midpoint (UTC)",
    )
    hdr["EXPTIME"] = (exposure, "Exposure duration (seconds)")

    metadata = {
        "ISO": ("EXIF ISOSpeedRatings", int),
        "FNUMBER": ("EXIF FNumber", float),
        "FOCALLEN": ("EXIF FocalLength", float),
    }

    for key, (exif_key, cast) in metadata.items():
        value = exif_float(tags, exif_key)
        if value is not None:
            hdr[key] = cast(value)

    camera = exif_value(tags, "Image Model")
    lens = exif_value(tags, "EXIF LensModel")

    if camera:
        hdr["INSTRUME"] = camera
    if lens:
        hdr["OBJECTIV"] = lens

    hdr["BLACKLVL"] = (
        float(sum(black_levels) / len(black_levels)),
        "Mean raw sensor black level",
    )
    hdr["WHITELEV"] = (white_level, "Raw sensor white level")
    for key, value in colour.items():
        hdr[key] = value

    output = Path(output) if output is not None else path.with_suffix(".fit")
    hdu.writeto(output, overwrite=True)

    print(f"{path} -> {output}")
    print(f"  Start:    {start.isoformat()}")
    print(f"  Midpoint: {midpoint.isoformat()}")
    print(f"  Exposure: {exposure} s")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Convert Canon CR2 files to FITS."
    )
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="CR2 files to convert",
    )
    parser.add_argument(
        "--timezone", default="Europe/Paris",
        help="Camera clock timezone (default: Europe/Paris)",
    )
    args = parser.parse_args()

    tz = ZoneInfo(args.timezone)

    failed = 0

    for path in args.files:
        try:
            convert(path, tz)
        except Exception as exc:
            print(f"{path}: ERROR: {exc}")
            failed += 1

    if failed:
        raise SystemExit(f"{failed} of {len(args.files)} files failed")

if __name__ == "__main__":
    main()
