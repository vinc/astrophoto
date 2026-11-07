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

def crop_image(path, radius, ra, dec, time, query, target_name=None):
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

    # Frames from convert.py hold the raw Bayer mosaic: average each
    # 2x2 block so the colour channels don't show as a checkerboard
    bayer = "RAWFILE" in header

    if bayer:
        x0, y0 = x0 // 2 * 2, y0 // 2 * 2
        x1 = x0 + (x1 - x0) // 2 * 2
        y1 = y0 + (y1 - y0) // 2 * 2

    crop = data[y0:y1, x0:x1].astype(float)

    if bayer:
        h, w = crop.shape
        crop = crop.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    vmin, vmax = ZScaleInterval().get_limits(crop)

    fig, ax = plt.subplots(subplot_kw={"projection": wcs})

    ax.imshow(
        crop,
        cmap="gray",
        origin="lower",
        extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5),
        vmin=vmin,
        vmax=vmax,
    )

    ax.scatter(
        [x],
        [y],
        facecolors="none",
        edgecolors="red",
        s=150,
    )

    ax.invert_yaxis()
    ax.coords["ra"].set_axislabel("Right Ascension")
    ax.coords["ra"].set_major_formatter("hh:mm:ss")
    ax.coords["dec"].set_axislabel("Declination")
    ax.coords["dec"].set_major_formatter("dd:mm:ss")
    ax.set_title(f"{target_name or query}\n{time.utc.isot.split(".")[0]} UTC")

    name = re.sub(r"[^a-z0-9_-]+", "-", query.lower()).strip("-")
    output = path.with_name(f"{path.stem}-crop-{name}.png")

    fig.savefig(output, dpi=150, bbox_inches="tight")
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
                target_name=str(position["targetname"]),
            )
        except Exception as exc:
            print(f"{path}: ERROR: {exc}")

if __name__ == "__main__":
    main()
