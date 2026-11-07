#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats

from camera import capture, display_limits, read_image

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
        flux = np.maximum(stamp - sky, 0)
        yy, xx = np.indices(stamp.shape)
        aperture = np.hypot(xx - radius, yy - radius) <= radius
        flux *= aperture
        total = flux.sum()
        if total <= 0 or flux.max() > 0.9 * total:
            continue
        cx, cy = (flux * xx).sum() / total, (flux * yy).sum() / total
        distances = np.hypot(xx - cx, yy - cy)
        pixels = np.argsort(distances.ravel())
        cumulative = np.cumsum(flux.ravel()[pixels])
        hfr = np.interp(total / 2, cumulative, distances.ravel()[pixels])
        if 0 < hfr < radius / 2:
            stars.append((x + cx - radius, y + cy - radius, hfr))
        if len(stars) == max_stars:
            break
    return stars

def show_focus(figure, data, stars, title):
    figure.clear()
    field, detail = figure.subplots(1, 2)
    low, high = display_limits(data)
    field.imshow(data, cmap="gray", origin="upper", vmin=low, vmax=high)
    if stars:
        field.scatter(*np.array(stars)[:, :2].T, facecolors="none", edgecolors="red", s=80)
        x, y, _ = stars[0]
        x, y = int(round(x)), int(round(y))
    else:
        y, x = np.array(data.shape) // 2
    patch = data[max(0, y - 30):y + 31, max(0, x - 30):x + 31]
    low = float(np.nanmedian(patch))
    high = max(float(np.nanmax(patch)), low + 1)
    detail.imshow(patch, cmap="gray", origin="upper", vmin=low, vmax=high, interpolation="nearest")
    field.set_title("Measured stars")
    detail.set_title("Star detail" if stars else "Centre detail")
    figure.suptitle(title)
    figure.tight_layout()
    figure.canvas.draw_idle()
    plt.pause(0.1)

def main():
    parser = argparse.ArgumentParser(description="Measure manual focus from stars in RAW or FITS images")
    parser.add_argument("files", nargs="*", type=Path, help="Saved images; omit to capture repeatedly")
    parser.add_argument("--shutterspeed", default="1/4", help="gphoto2 shutter speed for new captures (default: 1/4)")
    parser.add_argument("--interval", type=float, default=1, help="Pause between captures in seconds (default: 1)")
    parser.add_argument("--frames", type=int, default=0, help="Number of new captures; 0 repeats until stopped")
    parser.add_argument("--no-plot", action="store_true", help="Print measurements without a preview window")
    args = parser.parse_args()
    if args.interval < 0 or args.frames < 0:
        parser.error("--interval and --frames must not be negative")

    figure = None if args.no_plot else plt.figure(figsize=(11, 5))
    best = np.inf
    count = 0
    if not args.files:
        print("Adjust focus between captures; lower HFR is better. Ctrl-C or close the window to stop")
    try:
        while args.files or not args.frames or count < args.frames:
            if figure is not None and not plt.fignum_exists(figure.number):
                break
            with tempfile.TemporaryDirectory(prefix="astrophoto-focus-") as temporary:
                path = args.files[count] if args.files else capture(temporary, args.shutterspeed)
                data, saturated, scale = read_image(path)
                stars = measure_stars(data, saturated)
                if stars:
                    hfr = float(np.median([star[2] for star in stars])) * scale
                    best = min(best, hfr)
                    title = f"HFR {hfr:.2f} px | best {best:.2f} | {len(stars)} stars"
                else:
                    title = "No usable stars: check rough focus, exposure, or saturation"
                print(f"{path.name}: {title}", flush=True)
                if figure is not None:
                    show_focus(figure, data, stars, title)
            count += 1
            if args.files and count == len(args.files):
                break
            if not args.files and (not args.frames or count < args.frames):
                if figure is None:
                    time.sleep(args.interval)
                else:
                    plt.pause(max(args.interval, 0.01))
        if figure is not None and args.files:
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
