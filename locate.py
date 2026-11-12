#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import warnings
from zoneinfo import ZoneInfo

import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons

from convert import convert
from camera import display_limits, stack_display_limits
from color import read_rgb, read_saturation, render_rgb

ROOT = Path(__file__).resolve().parent
EPHEMERIS_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "astrophoto" / "horizons"

class PlateSolveError(RuntimeError):
    pass

def _save_solution(output, cached, source, source_info, timezone, raw):
    with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as saved:
        temporary = Path(saved.name)
        try:
            with output.open("rb") as solved:
                shutil.copyfileobj(solved, saved)
            saved.close()
            if raw:
                with fits.open(temporary, mode="update") as solved:
                    header = solved[0].header
                    header["RAWFILE"] = source.name
                    header["RAWSIZE"] = (source_info.st_size, "Original RAW size in bytes")
                    header["RAWMTIME"] = (source_info.st_mtime_ns, "Original RAW modification time in ns")
                    header["RAWTZ"] = (str(timezone), "Camera clock timezone used for conversion")
            temporary.chmod(stat.S_IMODE(source_info.st_mode))
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)

def solve_image(path, directory, timezone, scale_low, scale_high, cpu_limit=60, keep_solution=False):
    path = path.resolve()
    source = path
    source_info = source.stat()
    directory = Path(directory).resolve()
    raw = path.suffix.lower() not in {".fit", ".fits", ".new"}
    cached = source.with_suffix(".new")
    if cached == source:
        cached = source.with_name(f"{source.stem}-solved.new")
    previous = None
    if keep_solution and cached.exists() and cached.stat().st_mtime_ns >= source_info.st_mtime_ns:
        try:
            candidate = fits.getheader(cached)
            if "DATE-AVG" in candidate and WCS(candidate, fix=False).has_celestial:
                previous = candidate
        except (OSError, ValueError):
            pass
    # Provenance lets RAW caches be checked without decoding the sensor data
    if raw and previous is not None:
        identity = {
            "RAWFILE": source.name, "RAWSIZE": source_info.st_size,
            "RAWMTIME": source_info.st_mtime_ns, "RAWTZ": str(timezone),
        }
        if all(previous.get(key) == value for key, value in identity.items()):
            return cached
        if any(key in previous for key in ("RAWSIZE", "RAWMTIME", "RAWTZ")):
            previous = None
    if raw:
        path = convert(path, timezone, output=directory / "capture.fit")
    header = fits.getheader(path)
    if "DATE-AVG" not in header:
        raise ValueError(f"{path}: missing DATE-AVG; convert the original RAW image first")
    if WCS(header, fix=False).has_celestial:
        return path

    if previous is not None:
        same_frame = all(previous.get(key) == header.get(key) for key in ("DATE-AVG", "NAXIS1", "NAXIS2"))
        if raw:
            same_frame = same_frame and previous.get("RAWFILE") == header.get("RAWFILE")
        if same_frame:
            if raw:
                _save_solution(cached, cached, source, source_info, timezone, raw=True)
            return cached

    output = directory / "solved.new"
    marker = directory / "solved.solved"
    marker.unlink(missing_ok=True)
    try:
        subprocess.run([
            "solve-field", "--backend-config", str(ROOT / "astrometry.cfg"),
            "--scale-units", "degwidth", "--scale-low", str(scale_low),
            "--scale-high", str(scale_high), "--downsample", "2",
            "--no-plots", "--overwrite", "--cpulimit", str(cpu_limit),
            "--dir", str(directory), "--new-fits", str(output),
            "--solved", str(marker), str(path),
        ], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise PlateSolveError(f"Plate solving failed with exit status {exc.returncode}") from exc
    if not marker.exists() or not output.exists():
        raise PlateSolveError("Plate solving failed; check focus, exposure, indexes, and scale limits")
    if not WCS(fits.getheader(output), fix=False).has_celestial:
        raise PlateSolveError("Plate solving returned an image without celestial coordinates")
    if header.get("RGB", False):
        # Some solve-field versions only copy the primary image into the result
        with fits.open(path) as source_hdus, fits.open(output, mode="update") as solved_hdus:
            for name in ("RGB", "SATURATED"):
                if name in source_hdus and name not in solved_hdus:
                    solved_hdus.append(source_hdus[name].copy())
            solved_hdus[0].header["RGB"] = True
    if keep_solution:
        _save_solution(output, cached, source, source_info, timezone, raw=raw)
        return cached
    return output

def _query_ephemerides(query, observer, epochs):
    def lookup(target):
        return Horizons(
            id=target,
            location=dict(observer),
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

def get_ephemerides(query, observer, epochs):
    # Cache individual epochs so different image lists can share positions
    location = {key: float(value) for key, value in observer.items()}
    location.setdefault("body", 399.0)
    epochs = [float(epoch) for epoch in epochs]
    paths, positions = {}, {}
    for epoch in set(epochs):
        identity = json.dumps([1, query.strip(), location, epoch.hex()], sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()
        paths[epoch] = EPHEMERIS_CACHE / f"{key}.ecsv"
        try:
            cached = Table.read(paths[epoch], format="ascii.ecsv")
            if len(cached) == 1 and {"RA", "DEC", "targetname"} <= set(cached.colnames):
                positions[epoch] = cached
        except (OSError, ValueError):
            pass

    missing = sorted(set(epochs) - positions.keys())
    if missing:
        result = _query_ephemerides(query, observer, missing)
        if len(result) != len(missing):
            raise ValueError("Horizons returned a different number of epochs than requested")
        for index, epoch in enumerate(missing):
            positions[epoch] = result[index:index + 1]
            temporary = None
            try:
                EPHEMERIS_CACHE.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=EPHEMERIS_CACHE, delete=False) as saved:
                    temporary = Path(saved.name)
                positions[epoch].write(temporary, format="ascii.ecsv", overwrite=True)
                temporary.replace(paths[epoch])
            except OSError as exc:
                warnings.warn(f"Could not cache Horizons position: {exc}", RuntimeWarning)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    return vstack([positions[epoch] for epoch in epochs], metadata_conflicts="silent")

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

    display_header = header
    if source_path != Path(path) and source_path.suffix.lower() in {".fit", ".fits", ".new"}:
        # Cached plate solutions may predate the saved display limits
        display_header = fits.getheader(source_path)
    colour_path = source_path if source_path.suffix.lower() in {".fit", ".fits", ".new"} else Path(path)
    rgb, colour_header = read_rgb(colour_path)
    if rgb is None and colour_path != Path(path):
        colour_path = Path(path)
        rgb, colour_header = read_rgb(colour_path)
    if rgb is not None:
        if rgb.shape[:2] != data.shape:
            raise ValueError("RGB data dimensions do not match the plate solution")
        saturation = read_saturation(colour_path)
        # Feather highlights in the full image so crops keep identical transitions
        crop = render_rgb(rgb, colour_header, saturation)[y0:y1, x0:x1]
    if "NCOMBINE" in display_header or "STACKMTH" in display_header:
        vmin, vmax = stack_display_limits(data, display_header)
    else:
        vmin, vmax = display_limits(crop)

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
    parser.add_argument("--cpu-limit", type=int, default=60, help="Plate-solving CPU limit in seconds (default: 60)")
    args = parser.parse_args()
    if not (-90 <= args.latitude <= 90 and -180 <= args.longitude <= 180):
        parser.error("Invalid observer latitude or longitude")
    if not (0 < args.scale_low < args.scale_high):
        parser.error("Scale limits must satisfy 0 < --scale-low < --scale-high")
    if args.cpu_limit <= 0:
        parser.error("--cpu-limit must be positive")

    timezone = ZoneInfo(args.timezone)
    observer = {"lon": args.longitude, "lat": args.latitude, "elevation": args.elevation}
    failed = False
    with tempfile.TemporaryDirectory(prefix="astrophoto-locate-") as temporary:
        observations = []
        for index, path in enumerate(args.files):
            try:
                directory = Path(temporary) / str(index)
                directory.mkdir()
                solved = solve_image(
                    path, directory, timezone, args.scale_low, args.scale_high,
                    cpu_limit=args.cpu_limit, keep_solution=True,
                )
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
