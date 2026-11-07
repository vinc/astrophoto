#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates
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

def align_image(data, wcs, ra, dec):
    # Resample the full frame onto a north-up RA/Dec grid before cropping
    scale = float(np.sqrt(np.prod(proj_plane_pixel_scales(wcs))))
    aligned_wcs = WCS(naxis=2)
    aligned_wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    aligned_wcs.wcs.crval = [ra, 0]
    aligned_wcs.wcs.crpix = [1, 1]
    aligned_wcs.wcs.cdelt = [-scale, scale]

    height, width = data.shape
    edge_x = np.linspace(-0.5, width - 0.5, 128)
    edge_y = np.linspace(-0.5, height - 0.5, 128)
    xs = np.concatenate([edge_x, edge_x, np.full(128, -0.5),
                         np.full(128, width - 0.5)])
    ys = np.concatenate([np.full(128, -0.5), np.full(128, height - 0.5),
                         edge_y, edge_y])
    lon, lat = wcs.pixel_to_world_values(xs, ys)
    px, py = aligned_wcs.world_to_pixel_values(lon, lat)
    left, bottom = np.floor([px.min(), py.min()]).astype(int)
    right, top = np.ceil([px.max(), py.max()]).astype(int)
    aligned_wcs.wcs.crpix -= [left, bottom]
    aligned = np.empty((top - bottom + 1, right - left + 1))

    for row in range(0, aligned.shape[0], 128):
        yy, xx = np.mgrid[row:min(row + 128, aligned.shape[0]), :aligned.shape[1]]
        lon, lat = aligned_wcs.pixel_to_world_values(xx, yy)
        sx, sy = wcs.world_to_pixel_values(lon, lat)
        aligned[row:row + len(yy)] = map_coordinates(
            data, [sy, sx], order=1, mode="constant", cval=np.nan,
        )

    return aligned, aligned_wcs

def crop_image(path, radius, ra, dec, time, query, target_name=None):
    data, header = fits.getdata(path, header=True)
    wcs = WCS(header)

    x, y = wcs.world_to_pixel_values(ra, dec)
    height, width = data.shape

    print(f"{path}: {query} at ({x:.1f}, {y:.1f})")

    if not (0 <= x < width and 0 <= y < height):
        print("  Skipped: object is outside the image")
        return

    data = data.astype(float)

    # Bin the full Bayer frame before resampling its colour mosaic
    if "RAWFILE" in header:
        h, w = data.shape
        data = data[:h // 2 * 2, :w // 2 * 2]
        data = data.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        wcs = wcs.slice((slice(None, None, 2), slice(None, None, 2)))
        radius = max(1, radius // 2)

    data, wcs = align_image(data, wcs, ra, dec)
    x, y = wcs.world_to_pixel_values(ra, dec)
    height, width = data.shape
    x0 = max(0, int(x) - radius)
    x1 = min(width, int(x) + radius)
    y0 = max(0, int(y) - radius)
    y1 = min(height, int(y) + radius)
    crop = data[y0:y1, x0:x1]
    finite = crop[np.isfinite(crop)]
    if not finite.size:
        print("  Skipped: crop has no valid pixels")
        return
    vmin, vmax = ZScaleInterval().get_limits(finite)

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

    ax.coords["ra"].set_axislabel("Right Ascension")
    ax.coords["ra"].set_major_formatter("hh:mm:ss")
    ax.coords["dec"].set_axislabel("Declination")
    ax.coords["dec"].set_major_formatter("dd:mm:ss")
    ax.set_title(f"{target_name or query}\n{time.utc.isot.split('.')[0]} UTC")

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
