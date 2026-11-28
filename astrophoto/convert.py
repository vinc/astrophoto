import argparse
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import os
from pathlib import Path
import re
import tempfile

import exifread
import rawpy
from astropy.io import fits

from astrophoto.lib.color import raw_color_metadata
from astrophoto.lib.config import camera_timezone

def exif_value(tags, key):
    tag = tags.get(key)
    return str(tag) if tag is not None else None


def exif_float(tags, key):
    value = exif_value(tags, key)
    return float(Fraction(value)) if value is not None else None


def exif_pixel_size(tags):
    """Return the sensor pixel width in micrometers from the EXIF focal plane resolution, or None."""
    resolution = exif_float(tags, "EXIF FocalPlaneXResolution")
    unit = tags.get("EXIF FocalPlaneResolutionUnit")
    if not resolution or unit is None:
        return None
    values = getattr(unit, "values", None)
    code = values[0] if values else str(unit)
    microns = {2: 25400, 3: 10000, "2": 25400, "3": 10000}.get(code)
    if microns is None:
        text = str(unit).lower()
        microns = 25400 if "inch" in text else 10000 if "centim" in text else None
    if microns is None:
        return None
    size = microns / resolution
    # Anything outside this range is not a plausible camera sensor pixel
    return size if 1 <= size <= 20 else None


def camera_temperature(tags):
    """Return the sensor temperature in Celsius from Canon maker notes, or None when they do not record one."""
    # exifread shows it as "26 C", having removed Canon's offset of 128
    match = re.fullmatch(r"(-?\d+) C", exif_value(tags, "MakerNote CameraTemperature") or "")
    if match is None:
        return None
    temperature = int(match.group(1))
    # Other makers' notes, or a misread one, can hold anything
    return temperature if -40 <= temperature <= 80 else None


def read_tags(path):
    """Return the EXIF tags of a RAW file, with its maker notes when they can be read."""
    with path.open("rb") as f:
        try:
            return exifread.process_file(f, details=True)
        # Maker notes are proprietary, and a camera exifread does not know could trip it up
        except Exception:
            f.seek(0)
            return exifread.process_file(f, details=False)


def convert(path, tz, output=None):
    with rawpy.imread(str(path)) as raw:
        # Preserve the original Bayer mosaic without demosaicing
        data = raw.raw_image_visible.copy()
        black_levels = list(raw.black_level_per_channel)
        white_level = int(raw.white_level)
        color = raw_color_metadata(raw)

    tags = read_tags(path)

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

    pixel = exif_pixel_size(tags)
    if pixel is not None:
        hdr["XPIXSZ"] = (round(pixel, 3), "Pixel width in micrometers, including binning")
        hdr["XBINNING"] = (1, "Pixels combined horizontally")

    temperature = camera_temperature(tags)
    if temperature is not None:
        hdr["CCD-TEMP"] = (temperature, "Sensor temperature (C) from the maker notes")

    if camera:
        hdr["INSTRUME"] = camera
    if lens:
        hdr["OBJECTIV"] = lens

    hdr["BLACKLVL"] = (
        float(sum(black_levels) / len(black_levels)),
        "Mean raw sensor black level",
    )
    hdr["WHITELEV"] = (white_level, "Raw sensor white level")
    for key, value in color.items():
        hdr[key] = value

    output = Path(output) if output is not None else path.with_suffix(".fit")
    # Written under a temporary name, then renamed, so a stack scanning for FITS frames never reads half a file;
    # each conversion has a name of its own, so two converting the same file at once cannot mix their writes
    handle, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        hdu.writeto(temporary, overwrite=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"{path} -> {output}")
    print(f"  Start:    {start.isoformat()}")
    print(f"  Midpoint: {midpoint.isoformat()}")
    print(f"  Exposure: {exposure} s")
    if temperature is not None:
        print(f"  Sensor:   {temperature} °C")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Convert Canon CR2 files to FITS."
    )
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="CR2 files to convert",
    )
    args = parser.parse_args()

    tz = camera_timezone(parser)

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
