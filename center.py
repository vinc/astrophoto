#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import tempfile
import time
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers
from astropy.wcs import NoConvergence, WCS

from camera import capture, display_limits, preview_figure, read_image
from convert import convert
from crop import get_ephemerides

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

def pointing_offset(center, target, time, observer):
    location = EarthLocation.from_geodetic(
        observer["lon"] * u.deg, observer["lat"] * u.deg,
        observer["elevation"] * u.km,
    )
    frame = AltAz(obstime=time, location=location)
    # Bundled Earth rotation data is sufficient for manual pointing
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
        center_altaz = center.transform_to(frame)
        target_altaz = target.transform_to(frame)
    azimuth = (target_altaz.az - center_altaz.az).wrap_at(180 * u.deg).deg
    altitude = (target_altaz.alt - center_altaz.alt).deg
    return float(azimuth), float(altitude)

def angle_text(degrees):
    if abs(degrees) >= 1:
        return f"{abs(degrees):.2f} degrees"
    return f"{abs(degrees) * 60:.1f} arcmin"

def inspect_image(path, query, observer, coordinates=None):
    data, header = fits.getdata(path, header=True)
    if data.ndim != 2:
        raise ValueError("Expected a 2D plate-solved FITS image")
    time = Time(header["DATE-AVG"], format="isot", scale="utc")
    wcs = WCS(header).celestial
    height, width = data.shape
    center = wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2)
    if coordinates is None:
        position = get_ephemerides(query, observer, [time.jd])[0]
        target = SkyCoord(float(position["RA"]) * u.deg, float(position["DEC"]) * u.deg)
        target_name = str(position["targetname"])
    else:
        target = SkyCoord(coordinates[0] * u.deg, coordinates[1] * u.deg)
        target_name = query
    pan, tilt = pointing_offset(center, target, time, observer)
    try:
        x, y = map(float, wcs.world_to_pixel(target))
    except NoConvergence:
        x, y = np.nan, np.nan
    inside = bool(0 <= x < width and 0 <= y < height)
    directions = (
        f"Pan {'right' if pan >= 0 else 'left'} {angle_text(pan)} and "
        f"{'raise' if tilt >= 0 else 'lower'} {angle_text(tilt)}"
    )
    print(f"{target_name}: {time.utc.iso.split('.')[0]} UTC")
    print(f"  Separation from centre: {angle_text(center.separation(target).deg)}")
    print(f"  {directions}")
    print(f"  Target is {'inside' if inside else 'outside'} the frame")
    return {
        "x": x, "y": y, "inside": inside, "pan": pan, "tilt": tilt,
        "title": directions, "target_name": target_name,
    }

def show_pointing(figure, path, query, result, interactive=True):
    data, _, _ = read_image(path, bin_raw=False)
    height, width = data.shape
    cx, cy = (width - 1) / 2, (height - 1) / 2
    figure.clear()
    ax = figure.subplots()
    low, high = display_limits(data)
    ax.imshow(data, cmap="gray", origin="upper", vmin=low, vmax=high)
    ax.plot(cx, cy, "+", color="red", markersize=15, markeredgewidth=2)
    target_name = result.get("target_name", query)
    x, y = result["x"], result["y"]
    if result["inside"]:
        ax.scatter([x], [y], facecolors="none", edgecolors="red", linewidths=2, s=150)
        ax.annotate(target_name, (x, y), xytext=(10, 10), textcoords="offset points", color="red")
    elif np.isfinite([x, y]).all():
        dx, dy = x - cx, y - cy
        fraction = 0.95 * min(cx / abs(dx) if dx else np.inf, cy / abs(dy) if dy else np.inf)
        edge = (cx + dx * fraction, cy + dy * fraction)
        horizontal = "center" if abs(edge[0] - cx) < cx / 4 else "left" if dx < 0 else "right"
        vertical = "center" if abs(edge[1] - cy) < cy / 4 else "top" if dy < 0 else "bottom"
        ax.annotate(
            target_name, xy=(cx, cy), xytext=edge, color="red",
            ha=horizontal, va=vertical,
            arrowprops={"arrowstyle": "<-", "color": "red", "lw": 2, "shrinkA": 8, "shrinkB": 0},
        )
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_axis_off()
    ax.set_title(f"{target_name}\n{result['title']}")
    figure.tight_layout()
    figure.canvas.draw_idle()
    if interactive:
        plt.pause(0.1)

def main():
    parser = argparse.ArgumentParser(description="Guide manual camera pointing using a plate-solved exposure")
    parser.add_argument("file", nargs="?", type=Path, help="Saved RAW or FITS image; omit to capture a new image")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--query", help="Solar System body name or designation")
    target.add_argument("--ra", type=float, help="Fixed ICRS right ascension in degrees; requires --dec")
    parser.add_argument("--dec", type=float, help="Fixed ICRS declination in degrees")
    parser.add_argument("--latitude", type=float, required=True, help="Observer latitude in degrees")
    parser.add_argument("--longitude", type=float, required=True, help="Observer longitude in degrees")
    parser.add_argument("--elevation", type=float, required=True, help="Observer elevation in km")
    parser.add_argument("--timezone", default="Europe/Paris", help="Camera clock timezone (default: Europe/Paris)")
    parser.add_argument("--scale-low", type=float, default=3, help="Minimum image width in degrees (default: 3)")
    parser.add_argument("--scale-high", type=float, default=20, help="Maximum image width in degrees (default: 20)")
    parser.add_argument("--shutterspeed", help="Optional gphoto2 shutter speed for new captures")
    parser.add_argument("--loop", action="store_true", help="Repeat new captures until Ctrl-C or the window is closed")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--plot", action="store_true", help="Open a preview window for a saved image")
    preview.add_argument("--no-plot", action="store_true", help="Print corrections without a preview window")
    parser.add_argument("--save-preview", type=Path, help="Save an annotated PNG without requiring a GUI")
    args = parser.parse_args()
    if (args.ra is None) != (args.dec is None):
        parser.error("Provide both --ra and --dec, or use --query")
    if args.ra is not None and not (0 <= args.ra < 360 and -90 <= args.dec <= 90):
        parser.error("--ra must be in [0, 360) and --dec in [-90, 90]")
    if not (-90 <= args.latitude <= 90 and -180 <= args.longitude <= 180):
        parser.error("Invalid observer latitude or longitude")
    if not (0 < args.scale_low < args.scale_high):
        parser.error("Scale limits must satisfy 0 < --scale-low < --scale-high")
    if args.loop and args.file:
        parser.error("--loop requires new captures, so omit the input file")

    timezone = ZoneInfo(args.timezone)
    observer = {"lon": args.longitude, "lat": args.latitude, "elevation": args.elevation}
    coordinates = None if args.ra is None else (args.ra, args.dec)
    query = args.query or f"RA {args.ra:g}, Dec {args.dec:g}"
    interactive = not args.no_plot and (args.file is None or args.plot)
    figure = preview_figure(figsize=(10, 7)) if interactive else None
    if figure is None and args.save_preview:
        figure = plt.figure(figsize=(10, 7))
    try:
        while True:
            if figure is not None and not plt.fignum_exists(figure.number):
                break
            with tempfile.TemporaryDirectory(prefix="astrophoto-center-") as temporary:
                path = args.file or capture(temporary, args.shutterspeed)
                try:
                    solved = solve_image(path, temporary, timezone, args.scale_low, args.scale_high)
                except PlateSolveError as exc:
                    if not args.loop:
                        raise
                    print(f"{exc}; retrying with the next capture", flush=True)
                    if figure is not None:
                        figure.suptitle("No solution; adjust the camera before the next exposure")
                else:
                    result = inspect_image(solved, query, observer, coordinates)
                    if figure is not None:
                        show_pointing(figure, solved, query, result, interactive=interactive)
                        if args.save_preview:
                            figure.savefig(args.save_preview, dpi=150)
                            print(f"  Saved {args.save_preview}")
            if not args.loop:
                break
            if interactive:
                plt.pause(1)
            else:
                time.sleep(1)
        if interactive and not args.loop:
            plt.show()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        if figure is not None:
            plt.close(figure)

if __name__ == "__main__":
    main()
