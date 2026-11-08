#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import subprocess
import tempfile
from zoneinfo import ZoneInfo

import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons

from convert import convert

ROOT = Path(__file__).resolve().parent

class PlateSolveError(RuntimeError):
    pass

def solve_image(path, directory, timezone, scale_low, scale_high):
    path = path.resolve()
    directory = Path(directory).resolve()
    if path.suffix.lower() not in {".fit", ".fits", ".new"}:
        path = convert(path, timezone, output=directory / "capture.fit")
    header = fits.getheader(path)
    if "DATE-AVG" not in header:
        raise ValueError(f"{path}: missing DATE-AVG; convert the original RAW image first")
    if WCS(header, fix=False).has_celestial:
        return path

    output = directory / "solved.new"
    marker = directory / "solved.solved"
    marker.unlink(missing_ok=True)
    subprocess.run([
        "solve-field", "--backend-config", str(ROOT / "astrometry.cfg"),
        "--scale-units", "degwidth", "--scale-low", str(scale_low),
        "--scale-high", str(scale_high), "--downsample", "2",
        "--no-plots", "--overwrite", "--cpulimit", "60",
        "--dir", str(directory), "--new-fits", str(output),
        "--solved", str(marker), str(path),
    ], cwd=ROOT, check=True)
    if not marker.exists() or not output.exists():
        raise PlateSolveError("Plate solving failed; check focus, exposure, indexes, and scale limits")
    return output

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

def save_image(path, ra, dec, time, query, target_name=None, radius=None, source_path=None):
    data, header = fits.getdata(path, header=True)
    wcs = WCS(header)

    x, y = wcs.world_to_pixel_values(ra, dec)
    height, width = data.shape

    source_path = Path(path if source_path is None else source_path)
    print(f"{source_path}: {query} at ({x:.1f}, {y:.1f})")

    if not (0 <= x < width and 0 <= y < height):
        print("  Skipped: object is outside the image")
        return

    if radius is None:
        x0, x1, y0, y1 = 0, width, 0, height
    else:
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

    fig, ax = plt.subplots(figsize=(10, 7) if radius is None else None)

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
    ax.set_axis_off()
    ax.set_title(f"{target_name or query}\n{time.utc.iso.split('.')[0]} UTC")

    name = re.sub(r"[^a-z0-9_-]+", "-", query.lower()).strip("-")
    kind = "locate" if radius is None else "crop"
    output = source_path.with_name(f"{source_path.stem}-{kind}-{name}.png")

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved {output}")

def main():
    matplotlib.use("Agg")
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
                save_image(
                    solved, float(position["RA"]), float(position["DEC"]),
                    time, args.query, target_name=str(position["targetname"]), source_path=path,
                )
            except Exception as exc:
                print(f"{path}: ERROR: {exc}")
                failed = True

    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
