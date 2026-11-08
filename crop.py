#!/usr/bin/env python3

import argparse
from pathlib import Path

from astropy.io import fits
from astropy.time import Time

from locate import get_ephemerides, save_image

def crop_image(path, radius, ra, dec, time, query, target_name=None, source_path=None):
    return save_image(
        path, ra, dec, time, query, target_name=target_name,
        radius=radius, source_path=source_path,
    )

def main():
    parser = argparse.ArgumentParser(
        description="Locate a Solar System body in plate-solved FITS images"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Plate-solved FITS files",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Solar System body name or designation",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        required=True,
        help="Observer latitude in degrees",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        required=True,
        help="Observer longitude in degrees",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        required=True,
        help="Observer elevation in km",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=100,
        help="Crop radius in pixels (default: 100)",
    )
    args = parser.parse_args()

    observations = []

    for path in args.files:
        try:
            header = fits.getheader(path)
            time = Time(
                header["DATE-AVG"],
                format="isot",
                scale="utc",
            )
            observations.append((path, time))
        except Exception as exc:
            print(f"{path}: ERROR reading timestamp: {exc}")

    if not observations:
        raise SystemExit("No valid observations")

    observer = {
        "lon": args.longitude,
        "lat": args.latitude,
        "elevation": args.elevation,
    }

    # Horizons returns discrete epochs in chronological order
    observations.sort(key=lambda observation: observation[1].jd)
    epochs = [time.jd for _, time in observations]

    result = get_ephemerides(args.query, observer, epochs)

    for (path, time), position in zip(observations, result):
        try:
            crop_image(
                path,
                args.radius,
                float(position["RA"]),
                float(position["DEC"]),
                time,
                args.query,
                target_name=str(position["targetname"]),
            )
        except Exception as exc:
            print(f"{path}: ERROR: {exc}")

if __name__ == "__main__":
    main()
