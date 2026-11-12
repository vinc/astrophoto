#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

import matplotlib
from matplotlib import patheffects
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, get_constellation
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from locate import ROOT, get_ephemerides, solve_image

CATALOG = ROOT / "index"
WIDTH, HEIGHT = 1200, 900
SURFACE = "#1a1a19"
TEXT = "#f0efec"
MUTED = "#c3c2b7"
GRID = "#383835"
FIGURES = "#5f5e5a"
NAMES = "#8a8983"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]

def unit_vectors(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)

def to_radec(vectors):
    x, y, z = np.moveaxis(np.asarray(vectors), -1, 0)
    return np.degrees(np.arctan2(y, x)) % 360, np.degrees(np.arctan2(z, np.hypot(x, y)))

def separation(ra, dec, center):
    cosine = unit_vectors(ra, dec) @ unit_vectors(*center)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

def footprint(header, samples=32):
    # Edges are sampled rather than joined corner to corner because
    # straight lines on the sensor are curves on the chart
    wcs = WCS(header).celestial
    if not wcs.has_celestial:
        raise ValueError("Image has no plate solution")
    width, height = header["NAXIS1"], header["NAXIS2"]
    xs = np.linspace(-0.5, width - 0.5, samples)
    ys = np.linspace(-0.5, height - 0.5, samples)
    x = np.concatenate([xs, np.full(samples, width - 0.5), xs[::-1], np.full(samples, -0.5)])
    y = np.concatenate([np.full(samples, -0.5), ys, np.full(samples, height - 0.5), ys[::-1]])
    ra, dec = wcs.pixel_to_world_values(x, y)
    return np.asarray(ra) % 360, np.asarray(dec)

def chart_center(outlines):
    total = sum(unit_vectors(ra, dec).sum(axis=0) for ra, dec in outlines)
    if np.linalg.norm(total) < 1e-9:
        total = unit_vectors(*outlines[0]).sum(axis=0)
    ra, dec = to_radec(total / np.linalg.norm(total))
    return float(ra), float(dec)

def default_fov(outlines, center):
    # Fit the projected outlines rather than their angular spread, which
    # underestimates how far the projection pushes distant images out
    unit = chart_wcs(center, WIDTH)
    x, y = unit.world_to_pixel_values(*map(np.concatenate, zip(*outlines)))
    half_width = np.abs(x - (WIDTH - 1) / 2).max()
    half_height = np.abs(y - (HEIGHT - 1) / 2).max() * WIDTH / HEIGHT
    return float(max(30, 2.5 * max(half_width, half_height)))

def chart_reach(wcs):
    corners = wcs.pixel_to_world_values([-0.5, WIDTH - 0.5, WIDTH - 0.5, -0.5], [-0.5, -0.5, HEIGHT - 0.5, HEIGHT - 0.5])
    return float(min(separation(*corners, wcs.wcs.crval).max() + 5, 170))

def chart_wcs(center, fov):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---STG", "DEC--STG"]
    wcs.wcs.crval = center
    wcs.wcs.crpix = [(WIDTH + 1) / 2, (HEIGHT + 1) / 2]
    wcs.wcs.cdelt = [-fov / WIDTH, fov / WIDTH]
    return wcs

def load_features(name):
    path = CATALOG / name
    if not path.exists():
        raise SystemExit(f"Missing {path}; run 'sh index/download.sh'")
    with path.open() as file:
        return json.load(file)["features"]

def load_stars(magnitude):
    rows = [(*f["geometry"]["coordinates"], f["properties"]["mag"]) for f in load_features("stars.6.json")]
    ra, dec, mag = np.array(rows, dtype=float).T
    keep = mag <= magnitude
    return ra[keep] % 360, dec[keep], mag[keep]

def figure_lines(center, reach):
    for feature in load_features("constellations.lines.json"):
        for line in feature["geometry"]["coordinates"]:
            ra, dec = np.array(line, dtype=float).T
            vectors = unit_vectors(ra, dec)
            distances = separation(ra, dec, center)
            for i, (a, b) in enumerate(zip(vectors, vectors[1:])):
                if max(distances[i], distances[i + 1]) > reach:
                    continue
                # Normalised linear interpolation stays on the great circle
                t = np.linspace(0, 1, 16)[:, None]
                path = (1 - t) * a + t * b
                yield to_radec(path / np.linalg.norm(path, axis=1, keepdims=True))

def figure_names(center, reach):
    for feature in load_features("constellations.json"):
        ra, dec = feature["geometry"]["coordinates"]
        if separation(ra, dec, center) <= reach:
            yield ra % 360, dec, feature["properties"]["name"]

def star_sizes(mag, magnitude):
    return 2.5 * (magnitude + 1.5 - mag) ** 2

def inside(x, y, margin=0):
    return (-0.5 + margin <= x) & (x <= WIDTH - 0.5 - margin) & (-0.5 + margin <= y) & (y <= HEIGHT - 0.5 - margin)

def draw_chart(outlines, output, labels=None, fov=None, magnitude=6.0, track=None, title=None):
    center = chart_center(outlines)
    if fov is None:
        fov = default_fov(outlines, center)
    wcs = chart_wcs(center, fov)
    reach = chart_reach(wcs)

    fig = plt.figure(figsize=(12, 9), facecolor=SURFACE)
    ax = fig.add_subplot(projection=wcs)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.5, WIDTH - 0.5)
    ax.set_ylim(-0.5, HEIGHT - 0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ra_axis, dec_axis = ax.coords
    ra_axis.set_major_formatter("hh:mm")
    dec_axis.set_major_formatter("dd")
    for axis, name, side in [(ra_axis, "Right ascension", "b"), (dec_axis, "Declination", "l")]:
        axis.set_axislabel(name, color=MUTED)
        axis.set_axislabel_position(side)
        axis.set_ticklabel(color=MUTED, simplify=False, exclude_overlapping=True)
        axis.set_ticklabel_position(side)
        axis.set_ticks(color=GRID)
        axis.grid(color=GRID, linewidth=0.6)

    for ra, dec in figure_lines(center, reach):
        x, y = wcs.world_to_pixel_values(ra, dec)
        ax.plot(x, y, color=FIGURES, linewidth=0.8, zorder=1)

    for ra, dec, name in figure_names(center, reach):
        x, y = wcs.world_to_pixel_values(ra, dec)
        if inside(x, y, margin=40):
            ax.text(x, y, name.upper(), color=NAMES, fontsize=8, ha="center", va="center", zorder=1)

    ra, dec, mag = load_stars(magnitude)
    near = separation(ra, dec, center) <= reach
    x, y = wcs.world_to_pixel_values(ra[near], dec[near])
    order = np.argsort(-mag[near])
    ax.scatter(x[order], y[order], s=star_sizes(mag[near][order], magnitude),
               color=TEXT, edgecolors=SURFACE, linewidths=0.5, zorder=2)

    legend = labels is not None and len(outlines) > 1
    # Past the palette, a shared colour beats generated hues nobody can tell apart
    shared = len(outlines) > len(SERIES)
    for index, (ra, dec) in enumerate(outlines):
        x, y = wcs.world_to_pixel_values(np.append(ra, ra[0]), np.append(dec, dec[0]))
        label = None
        if legend and not shared:
            label = labels[index]
        elif legend and index == 0:
            label = f"{len(outlines)} images"
        ax.plot(x, y, color=SERIES[0 if shared else index], linewidth=2, label=label, zorder=3)

    if track is not None:
        name, ra, dec = track
        x, y = wcs.world_to_pixel_values(np.asarray(ra), np.asarray(dec))
        ax.plot(x, y, color=TEXT, linewidth=1, linestyle=":", zorder=4)
        ax.scatter(x, y, s=64, facecolors="none", edgecolors=TEXT, linewidths=2, zorder=4)
        ax.annotate(name, (x[-1], y[-1]), xytext=(10, 10), textcoords="offset points",
                    color=TEXT, fontsize=10, zorder=4,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)])

    if legend:
        box = ax.legend(loc="lower left", facecolor=SURFACE, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
        box.set_zorder(5)

    if title is None:
        title = get_constellation(SkyCoord(*center, unit="deg"))
    ax.set_title(title, color=TEXT)

    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output

def describe_times(times):
    days = sorted({time.utc.iso[:10] for time in times})
    if len(times) == 1:
        return f"{times[0].utc.iso.split('.')[0][:16]} UTC"
    return days[0] if len(days) == 1 else f"{days[0]} to {days[-1]}"

def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Draw a star chart locating plate-solved images in the sky")
    parser.add_argument("files", nargs="+", type=Path, help="Saved RAW or FITS images, including stacks")
    parser.add_argument("--fov", type=float, help="Chart width in degrees at its centre (default: fit the images, at least 30)")
    parser.add_argument("--magnitude", type=float, default=6.0, help="Faintest star magnitude, up to 6 (default: 6)")
    parser.add_argument("--output", type=Path, help="PNG path (default: next to the earliest image)")
    parser.add_argument("--query", help="Solar System body to mark at each image time")
    parser.add_argument("--latitude", type=float, help="Observer latitude in degrees, with --query")
    parser.add_argument("--longitude", type=float, help="Observer longitude in degrees, with --query")
    parser.add_argument("--elevation", type=float, help="Observer elevation in km, with --query")
    parser.add_argument("--timezone", default="Europe/Paris", help="Camera clock timezone (default: Europe/Paris)")
    parser.add_argument("--scale-low", type=float, default=3, help="Minimum image width in degrees (default: 3)")
    parser.add_argument("--scale-high", type=float, default=20, help="Maximum image width in degrees (default: 20)")
    parser.add_argument("--cpu-limit", type=int, default=60, help="Plate-solving CPU limit in seconds (default: 60)")
    args = parser.parse_args()
    observer_given = [value is not None for value in (args.latitude, args.longitude, args.elevation)]
    if args.query and not all(observer_given):
        parser.error("--query needs --latitude, --longitude and --elevation")
    if args.query and not (-90 <= args.latitude <= 90 and -180 <= args.longitude <= 180):
        parser.error("Invalid observer latitude or longitude")
    if args.fov is not None and not (0 < args.fov <= 360):
        parser.error("--fov must be between 0 and 360 degrees")
    if not (0 < args.scale_low < args.scale_high):
        parser.error("Scale limits must satisfy 0 < --scale-low < --scale-high")
    if args.cpu_limit <= 0:
        parser.error("--cpu-limit must be positive")

    timezone = ZoneInfo(args.timezone)
    failed = False
    observations = []
    with tempfile.TemporaryDirectory(prefix="astrophoto-chart-") as temporary:
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
                observations.append((path, footprint(header), time))
            except Exception as exc:
                print(f"{path}: ERROR: {exc}")
                failed = True

    if not observations:
        raise SystemExit("No valid observations")

    observations.sort(key=lambda observation: observation[2].jd)
    paths, outlines, times = map(list, zip(*observations))
    labels = [f"{time.utc.iso[:16]} UTC" for time in times]

    track = None
    title = None
    if args.query:
        observer = {"lon": args.longitude, "lat": args.latitude, "elevation": args.elevation}
        result = get_ephemerides(args.query, observer, [time.jd for time in times])
        if len(result) != len(times):
            raise SystemExit("Horizons returned a different number of epochs than requested")
        name = str(result["targetname"][0])
        track = (name, np.asarray(result["RA"], dtype=float), np.asarray(result["DEC"], dtype=float))
        title = f"{name}\n{describe_times(times)}"
    else:
        center = SkyCoord(*chart_center(outlines), unit="deg")
        title = f"{get_constellation(center)}\n{describe_times(times)}"

    output = args.output
    if output is None:
        suffix = "" if not args.query else "-" + re.sub(r"[^a-z0-9_-]+", "-", args.query.lower()).strip("-")
        output = paths[0].with_name(f"{paths[0].stem}-chart{suffix}.png")
    draw_chart(outlines, output, labels=labels, fov=args.fov,
               magnitude=min(args.magnitude, 6.0), track=track, title=title)
    print(f"Saved {output}")

    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
