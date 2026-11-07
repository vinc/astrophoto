#!/usr/bin/env python3

import argparse
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")

from astropy.io import fits
from astropy.time import Time

from center import solve_image
from crop import crop_image, get_ephemerides

def main():
    parser = argparse.ArgumentParser(description="Mark a Solar System body in full-frame images and save PNGs")
    parser.add_argument("files", nargs="+", type=Path, help="Saved RAW or FITS images, including stacks")
    parser.add_argument("--query", required=True, help="Solar System body name or designation")
    parser.add_argument("--latitude", type=float, required=True, help="Observer latitude in degrees")
    parser.add_argument("--longitude", type=float, required=True, help="Observer longitude in degrees")
    parser.add_argument("--elevation", type=float, required=True, help="Observer elevation in km")
    parser.add_argument("--timezone", default="Europe/Paris", help="Camera clock timezone (default: Europe/Paris)")
    parser.add_argument("--scale-low", type=float, default=3, help="Minimum image width in degrees (default: 3)")
    parser.add_argument("--scale-high", type=float, default=20, help="Maximum image width in degrees (default: 20)")
    args = parser.parse_args()
    if not (-90 <= args.latitude <= 90 and -180 <= args.longitude <= 180):
        parser.error("Invalid observer latitude or longitude")
    if not (0 < args.scale_low < args.scale_high):
        parser.error("Scale limits must satisfy 0 < --scale-low < --scale-high")

    timezone = ZoneInfo(args.timezone)
    observer = {"lon": args.longitude, "lat": args.latitude, "elevation": args.elevation}
    failed = False
    with tempfile.TemporaryDirectory(prefix="astrophoto-locate-") as temporary:
        observations = []
        for index, path in enumerate(args.files):
            try:
                directory = Path(temporary) / str(index)
                directory.mkdir()
                solved = solve_image(path, directory, timezone, args.scale_low, args.scale_high)
                header = fits.getheader(solved)
                time = Time(header["DATE-AVG"], format="isot", scale="utc")
                observations.append((path, solved, time))
            except Exception as exc:
                print(f"{path}: ERROR: {exc}")
                failed = True

        if not observations:
            raise SystemExit("No valid observations")

        observations.sort(key=lambda observation: observation[2].jd)
        epochs = [time.jd for _, _, time in observations]
        result = get_ephemerides(args.query, observer, epochs)
        if len(result) != len(observations):
            raise SystemExit("Horizons returned a different number of epochs than requested")

        for (path, solved, time), position in zip(observations, result):
            try:
                crop_image(
                    solved, None, float(position["RA"]), float(position["DEC"]),
                    time, args.query, target_name=str(position["targetname"]), source_path=path,
                )
            except Exception as exc:
                print(f"{path}: ERROR: {exc}")
                failed = True

    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
