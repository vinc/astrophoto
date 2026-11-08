#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import label

from camera import capture, display_limits, preview_figure, read_image

class FocusRangeError(ValueError):
    pass

def measure_stars(data, saturated=None, radius=12, max_stars=20):
    background, _, noise = sigma_clipped_stats(data)
    inner = data[1:-1, 1:-1]
    peaks = np.isfinite(inner) & (inner > background + 8 * noise)
    for dy in range(3):
        for dx in range(3):
            if (dy, dx) != (1, 1):
                peaks &= inner > data[dy:dy + inner.shape[0], dx:dx + inner.shape[1]]
    ys, xs = np.nonzero(peaks)
    ys, xs = ys + 1, xs + 1
    order = np.argsort(data[ys, xs])[::-1]
    stars = []
    out_of_range = 0

    for index in order:
        x, y = int(xs[index]), int(ys[index])
        if min(x, y, data.shape[1] - 1 - x, data.shape[0] - 1 - y) < radius:
            continue
        if any(np.hypot(x - sx, y - sy) < 2 * radius for sx, sy, _ in stars):
            continue
        stamp = data[y - radius:y + radius + 1, x - radius:x + radius + 1]
        if not np.isfinite(stamp).all():
            continue
        if saturated is not None and saturated[
            y - radius:y + radius + 1, x - radius:x + radius + 1
        ].any():
            continue

        sky = np.median(np.concatenate([stamp[0], stamp[-1], stamp[1:-1, 0], stamp[1:-1, -1]]))
        core = stamp[radius - 1:radius + 2, radius - 1:radius + 2]
        # A real star must extend beyond an isolated detector pixel
        if np.count_nonzero(core > sky + 3 * noise) < 3:
            continue
        yy, xx = np.indices(stamp.shape)
        distance = np.hypot(xx - radius, yy - radius)
        aperture = distance <= radius
        rim = (distance > radius - 1.5) & (distance <= radius)
        threshold = max(5 * noise, 0.05 * (core.max() - sky))
        # Separate stellar footprints in one aperture would inflate its HFR
        components, _ = label((stamp - sky > threshold) & aperture, structure=np.ones((3, 3)))
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        sizes[components[radius, radius]] = 0
        if np.any(sizes >= 3):
            continue
        # Bright wings at the aperture edge make a truncated HFR misleading
        if np.count_nonzero(stamp[rim] - sky > threshold) >= 3:
            out_of_range += 1
            continue
        signal = stamp - sky
        flux = np.maximum(signal, 0)
        flux *= aperture
        positive_total = flux.sum()
        total = signal[aperture].sum()
        if total <= 10 * noise * np.sqrt(aperture.sum()) or flux.max() > 0.9 * total:
            continue
        cx, cy = (flux * xx).sum() / positive_total, (flux * yy).sum() / positive_total
        distances = np.hypot(xx - cx, yy - cy)
        radii = distances[aperture]
        pixels = np.argsort(radii)
        radii = radii[pixels]
        # Keep negative sky residuals so noise does not become extra star flux
        cumulative = np.cumsum(signal[aperture][pixels])
        index = int(np.flatnonzero(cumulative >= total / 2)[0])
        if index == 0:
            hfr = radii[0]
        else:
            hfr = np.interp(total / 2, cumulative[index - 1:index + 1], radii[index - 1:index + 1])
        if 0 < hfr < radius / 2:
            stars.append((x + cx - radius, y + cy - radius, hfr))
        else:
            out_of_range += 1
        if len(stars) == max_stars:
            break
    if not stars and out_of_range:
        raise FocusRangeError("HFR out of range: improve rough focus or increase --radius")
    return stars

def show_focus(figure, data, stars, title, interactive=True):
    figure.clear()
    field, detail = figure.subplots(1, 2)
    low, high = display_limits(data)
    field.imshow(data, cmap="gray", origin="lower", vmin=low, vmax=high)
    if stars:
        field.scatter(*np.array(stars)[:, :2].T, facecolors="none", edgecolors="red", s=80)
        height, width = data.shape
        candidates = [
            star for star in stars
            if 30 <= round(star[0]) < width - 30 and 30 <= round(star[1]) < height - 30
        ]
        x, y, _ = min(
            candidates or stars,
            key=lambda star: (star[0] - (width - 1) / 2)**2 + (star[1] - (height - 1) / 2)**2,
        )
        x, y = int(round(x)), int(round(y))
    else:
        y, x = np.array(data.shape) // 2
    patch = Cutout2D(data, (x, y), (61, 61), mode="partial", fill_value=np.nan).data
    low = float(np.nanmedian(patch))
    high = max(float(np.nanmax(patch)), low + 1)
    detail.set_facecolor("0.15")
    detail.imshow(patch, cmap="gray", origin="lower", vmin=low, vmax=high, interpolation="nearest")
    field.set_title("Measured stars")
    detail.set_title("Star detail" if stars else "Centre detail")
    figure.suptitle(title)
    figure.tight_layout()
    figure.canvas.draw_idle()
    if interactive:
        plt.pause(0.1)

def main():
    parser = argparse.ArgumentParser(description="Measure manual focus from stars in RAW or FITS images")
    parser.add_argument("files", nargs="*", type=Path, help="Saved images; omit to capture repeatedly")
    parser.add_argument("--shutterspeed", default="1/4", help="gphoto2 shutter speed for new captures (default: 1/4)")
    parser.add_argument("--interval", type=float, default=1, help="Pause between captures in seconds (default: 1)")
    parser.add_argument("--frames", type=int, default=0, help="Number of new captures; 0 repeats until stopped")
    parser.add_argument("--radius", type=int, default=12, help="Aperture radius in binned image pixels (default: 12)")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--plot", action="store_true", help="Open a preview window for saved images")
    preview.add_argument("--no-plot", action="store_true", help="Print measurements without a preview window")
    parser.add_argument("--save-preview", type=Path, help="Save an annotated PNG without requiring a GUI")
    args = parser.parse_args()
    if args.interval < 0 or args.frames < 0:
        parser.error("--interval and --frames must not be negative")
    if args.radius < 3:
        parser.error("--radius must be at least 3")
    if args.save_preview and len(args.files) > 1:
        parser.error("--save-preview accepts one saved image at a time")

    interactive = not args.no_plot and (not args.files or args.plot)
    figure = preview_figure(figsize=(11, 5)) if interactive else None
    if figure is None and args.save_preview:
        figure = plt.figure(figsize=(11, 5))
    best = np.inf
    count = 0
    if not args.files:
        print("Adjust focus between captures; lower HFR is better. Ctrl-C or close the window to stop")
    try:
        while args.files or not args.frames or count < args.frames:
            if figure is not None and not plt.fignum_exists(figure.number):
                break
            with tempfile.TemporaryDirectory(prefix="astrophoto-focus-") as temporary:
                try:
                    path = args.files[count] if args.files else capture(temporary, args.shutterspeed)
                except subprocess.CalledProcessError as exc:
                    print(f"{exc}; retrying with the next capture", flush=True)
                    if figure is not None:
                        figure.suptitle("Capture failed; retrying")
                        figure.canvas.draw_idle()
                    if interactive:
                        plt.pause(max(args.interval, 0.01))
                    else:
                        time.sleep(max(args.interval, 0.1))
                    continue
                data, saturated, scale = read_image(path)
                try:
                    stars = measure_stars(data, saturated, radius=args.radius)
                except FocusRangeError as exc:
                    stars = []
                    title = str(exc)
                else:
                    title = "No usable stars: check rough focus, exposure, or saturation"
                if stars:
                    hfr = float(np.median([star[2] for star in stars])) * scale
                    best = min(best, hfr)
                    title = f"HFR {hfr:.2f} px | {len(stars)} stars"
                    if not args.files:
                        title += f" | best {best:.2f}"
                print(f"{path.name}: {title}", flush=True)
                if figure is not None:
                    show_focus(figure, data, stars, title, interactive=interactive)
                    if args.save_preview:
                        figure.savefig(args.save_preview, dpi=150)
                        print(f"  Saved {args.save_preview}")
            count += 1
            if args.files and count == len(args.files):
                break
            if not args.files and (not args.frames or count < args.frames):
                if interactive:
                    plt.pause(max(args.interval, 0.01))
                else:
                    time.sleep(args.interval)
        if interactive and args.files:
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
