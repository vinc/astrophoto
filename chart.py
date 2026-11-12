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
WIDE = 120
SPREAD = 150
ANTIPODE = 175
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

def image_wcs(header):
    # The default fixes only add MJD keywords derived from DATE-OBS and DATE-AVG,
    # warning about it each time
    return WCS(header, fix=False).celestial

def contains(header, outline, ra, dec):
    center = chart_center([outline])
    # Far from the frame the SIP distortion inversion diverges
    if separation(ra, dec, center) > separation(*outline, center).max():
        return False
    x, y = image_wcs(header).world_to_pixel_values(ra, dec)
    return bool(0 <= x < header["NAXIS1"] and 0 <= y < header["NAXIS2"])

def footprint(header, samples=32):
    # Edges are sampled rather than joined corner to corner because
    # straight lines on the sensor are curves on the chart
    wcs = image_wcs(header)
    if not wcs.has_celestial:
        raise ValueError("Image has no plate solution")
    width, height = header["NAXIS1"], header["NAXIS2"]
    xs = np.linspace(-0.5, width - 0.5, samples)
    ys = np.linspace(-0.5, height - 0.5, samples)
    x = np.concatenate([xs, np.full(samples, width - 0.5), xs[::-1], np.full(samples, -0.5)])
    y = np.concatenate([np.full(samples, -0.5), ys, np.full(samples, height - 0.5), ys[::-1]])
    ra, dec = wcs.pixel_to_world_values(x, y)
    return np.asarray(ra) % 360, np.asarray(dec)

def enclosing_center(points):
    # Smallest enclosing cap: coarse search over evenly spread directions,
    # then local refinement with a shrinking step
    i = np.arange(2000) + 0.5
    z = 1 - 2 * i / len(i)
    phi = np.pi * (1 + np.sqrt(5)) * i
    candidates = np.stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z], axis=-1)
    center = candidates[np.argmax((candidates @ points[::4].T).min(axis=1))]
    step = np.radians(5)
    while step > 1e-6:
        u = np.cross(center, [0, 0, 1] if abs(center[2]) < 0.9 else [1, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(center, u)
        moves = [center + step * (a * u + b * v) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        moves = np.array(moves) / np.linalg.norm(moves, axis=1, keepdims=True)
        best = moves[np.argmax((moves @ points.T).min(axis=1))]
        if np.allclose(best, center):
            step /= 2
        center = best
    return center

def chart_center(outlines):
    points = np.concatenate([unit_vectors(ra, dec) for ra, dec in outlines])
    center = points.sum(axis=0)
    norm = np.linalg.norm(center)
    # The mean direction suits images within a hemisphere; beyond, it can land
    # near the antipode of some of them, where the projection blows up
    if norm < 1e-9 or (points @ (center / norm)).min() < 0:
        center = enclosing_center(points)
    else:
        center = center / norm
    radius = np.degrees(np.arccos(np.clip((points @ center).min(), -1, 1)))
    if radius > SPREAD:
        raise SystemExit(f"Images are spread over too much of the sky to chart together "
                         f"(more than {SPREAD}° from any centre); chart fewer at a time")
    ra, dec = to_radec(center)
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
            vectors = unit_vectors(*np.array(line, dtype=float).T)
            for a, b in zip(vectors, vectors[1:]):
                # Sampling every degree keeps a segment crossing the chart even
                # when both of its stars lie beyond the reach margin
                angle = np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
                t = np.linspace(0, 1, int(np.ceil(angle)) + 2)[:, None]
                path = (1 - t) * a + t * b
                ra, dec = to_radec(path / np.linalg.norm(path, axis=1, keepdims=True))
                distances = separation(ra, dec, center)
                if distances.min() > reach:
                    continue
                yield ra, np.where(distances > ANTIPODE, np.nan, dec)

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

    if fov <= WIDE:
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

    # Past the palette, a shared colour beats generated hues nobody can tell apart
    shared = len(outlines) > len(SERIES)
    legend = labels is not None and len(outlines) > 1 and not shared
    for index, (ra, dec) in enumerate(outlines):
        x, y = wcs.world_to_pixel_values(np.append(ra, ra[0]), np.append(dec, dec[0]))
        label = labels[index] if legend else None
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

def local_time(time, timezone):
    return time.to_datetime(timezone=timezone).strftime("%Y-%m-%d %H:%M %Z")

def describe_times(times, timezone):
    if len(times) == 1:
        return local_time(times[0], timezone)
    first, last = local_time(times[0], timezone), local_time(times[-1], timezone)
    if first[:10] == last[:10]:
        return f"{first[:10]} {first[11:16]} to {last[11:]}"
    return f"{first[:10]} to {last[:10]}"

def default_output(paths, query=None, directory=Path(".")):
    suffix = "" if not query else "-" + re.sub(r"[^a-z0-9_-]+", "-", query.lower()).strip("-")
    if len(paths) == 1:
        return paths[0].with_name(f"{paths[0].stem}-chart{suffix}.png")
    name = re.sub(r"^\d{8}-\d{4}-", "", paths[0].stem) + f"-chart{suffix}"
    numbers = [int(path.stem[-4:]) for path in directory.glob(f"{name}-[0-9][0-9][0-9][0-9].png")]
    return directory / f"{name}-{max(numbers, default=0) + 1:04d}.png"

def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Draw a star chart locating plate-solved images in the sky")
    parser.add_argument("files", nargs="+", type=Path, help="Saved RAW or FITS images, including stacks")
    parser.add_argument("--fov", type=float, help="Chart width in degrees at its centre (default: fit the images, at least 30)")
    parser.add_argument("--magnitude", type=float, default=6.0, help="Faintest star magnitude, up to 6 (default: 6)")
    parser.add_argument("--output", type=Path, help="PNG path (default: next to a single image, or <kind>-chart-NNNN.png in the current directory)")
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
                observations.append((path, header, footprint(header), time))
            except Exception as exc:
                print(f"{path}: ERROR: {exc}")
                failed = True

    if not observations:
        raise SystemExit("No valid observations")

    observations.sort(key=lambda observation: observation[3].jd)

    track = None
    if args.query:
        observer = {"lon": args.longitude, "lat": args.latitude, "elevation": args.elevation}
        result = get_ephemerides(args.query, observer, [observation[3].jd for observation in observations])
        if len(result) != len(observations):
            raise SystemExit("Horizons returned a different number of epochs than requested")
        name = str(result["targetname"][0])
        positions = zip(np.asarray(result["RA"], dtype=float), np.asarray(result["DEC"], dtype=float))
        visible = []
        for observation, (ra, dec) in zip(observations, positions):
            if contains(observation[1], observation[2], ra, dec):
                visible.append((observation, ra, dec))
            else:
                print(f"{observation[0]}: Skipped: {name} is outside the image")
        if not visible:
            raise SystemExit(f"{name} is in none of the images")
        observations, ra, dec = map(list, zip(*visible))
        track = (name, np.array(ra), np.array(dec))

    paths, _, outlines, times = map(list, zip(*observations))
    labels = [local_time(time, timezone) for time in times]

    if track is None and len(outlines) == 1:
        name = get_constellation(SkyCoord(*chart_center(outlines), unit="deg"))
    elif track is None:
        name = f"{len(outlines)} images"
    title = f"{name}\n{describe_times(times, timezone)}"

    output = args.output
    if output is None:
        output = default_output(paths, args.query)
    draw_chart(outlines, output, labels=labels, fov=args.fov,
               magnitude=min(args.magnitude, 6.0), track=track, title=title)
    print(f"Saved {output}")

    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
