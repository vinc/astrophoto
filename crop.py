#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons

def get_ephemerides(query, observer, epochs):
    def lookup(target):
        return Horizons(
            id=target,
            location=observer,
            epochs=epochs,
        ).ephemerides()

    try:
        return lookup(query)
    except ValueError as exc:
        if "Ambiguous target name" not in str(exc):
            raise

        matches = []

        for line in str(exc).splitlines():
            match = re.match(r"^\s*(-?\d+)\s+(.+)$", line)
            if not match:
                continue

            object_id, fields = match.groups()
            name = re.split(r"\s{2,}", fields.strip())[0]

            if name.casefold() == query.strip().casefold():
                matches.append(object_id)

        if len(matches) != 1:
            raise

        print(f"Resolved {query} to Horizons ID {matches[0]}")
        return lookup(matches[0])

def crop_image(path, radius, ra, dec, time, query):
    data, header = fits.getdata(path, header=True)
    wcs = WCS(header)

    x, y = wcs.world_to_pixel_values(ra, dec)
    height, width = data.shape

    print(f"{path}: {query} at ({x:.1f}, {y:.1f})")

    if not (0 <= x < width and 0 <= y < height):
        print("  Skipped: object is outside the image")
        return

    x0 = max(0, int(x) - radius)
    x1 = min(width, int(x) + radius)
    y0 = max(0, int(y) - radius)
    y1 = min(height, int(y) + radius)

    crop = data[y0:y1, x0:x1]
    vmin, vmax = ZScaleInterval().get_limits(crop)

    fig, ax = plt.subplots()

    ax.imshow(
        crop,
        cmap="gray",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    ax.scatter(
        [x - x0],
        [y - y0],
        facecolors="none",
        edgecolors="red",
        s=150,
    )

    ax.set_title(f"{query} — {time.utc.isot} UTC")

    name = re.sub(r"[^a-z0-9_-]+", "-", query.lower()).strip("-")
    output = path.with_name(f"{path.stem}-crop-{name}.png")

    fig.savefig(output, dpi=150)
    plt.close(fig)

    print(f"  Saved {output}")

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
            )
        except Exception as exc:
            print(f"{path}: ERROR: {exc}")

if __name__ == "__main__":
    main()
