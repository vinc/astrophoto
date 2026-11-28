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
from scipy.optimize import least_squares

from astro.lib.camera import (
    CAMERAS, capture, display_limits, ensure_back_button_focus, image_optics, live_preview, move_focus, pixel_scale,
    preview_figure, read_image,
)
from astro.lib.gphoto import FOCUS_STEPS, camera_lock
from astro.lib.config import camera_profile, command_settings

class FocusRangeError(ValueError):
    pass

class AutofocusError(RuntimeError):
    pass

DEFAULTS = {"shutterspeed": "1/4", "iso": None}
# The aperture radius was tuned on the 6.4 micrometer pixels of the EOS 20D
RADIUS, RADIUS_PIXEL_SIZE = 12, 6.4

# Steps driven past a position and back, so the lens always arrives moving toward infinity
BACKLASH = 3

def default_radius(camera):
    return RADIUS if camera is None else max(3, round(RADIUS * RADIUS_PIXEL_SIZE / camera.pixel_size))

def hfr_text(hfr, path, camera):
    """Format an HFR in sensor pixels, adding arcseconds when the optics are known."""
    text = f"HFR {hfr:.2f} px"
    # Live-view pixels are not sensor pixels, so their scale is unknown
    if Path(path).suffix.lower() in {".jpg", ".jpeg"}:
        return text
    try:
        model, focal, binning = image_optics(path)
    except (OSError, ValueError, KeyError):
        return text
    camera = CAMERAS.get(model, camera)
    if camera is not None and focal is not None and binning is not None:
        text += f' ({hfr * pixel_scale(camera, focal, binning):.1f}")'
    return text

def noise_floor(path):
    """Return the smallest noise to assume for an image: a dark 8-bit live-view frame can show none at all."""
    return 0.5 if Path(path).suffix.lower() in {".jpg", ".jpeg"} else 0.0

def measure_stars(data, saturated=None, radius=12, max_stars=20, min_noise=0.0):
    background, _, noise = sigma_clipped_stats(data)
    noise = max(float(noise), min_noise)
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

def frame_hfr(data, saturated, scale, radius, min_noise=0.0):
    """Return the stars and median HFR of a frame, with a title describing the measurement."""
    try:
        stars = measure_stars(data, saturated, radius=radius, min_noise=min_noise)
    except FocusRangeError as exc:
        return [], None, str(exc)
    if not stars:
        return [], None, "No usable stars: check rough focus, exposure, or saturation"
    return stars, float(np.median([star[2] for star in stars])) * scale, None

def measure_frame(path, radius):
    """Read an image and return its data, stars, median HFR in sensor pixels, and a problem when there is none."""
    data, saturated, scale = read_image(path)
    stars, hfr, problem = frame_hfr(data, saturated, scale, radius, noise_floor(path))
    return data, stars, hfr, problem

def fit_minimum(positions, hfrs):
    """Return the focus position at the bottom of the V-curve, or None when the scan did not pass through it."""
    pairs = [(position, hfr) for position, hfr in zip(positions, hfrs) if hfr is not None]
    if len(pairs) < 3:
        return None
    positions, hfrs = (np.array(values, dtype=float) for values in zip(*pairs))

    # Defocused stars grow along a hyperbola whose asymptotes form the V
    def residuals(parameters):
        best, slope, center = parameters
        return np.sqrt(best**2 + (slope * (positions - center)) ** 2) - hfrs

    # A heavy-tailed loss keeps a stray reading, such as noise taken for a star, from dragging the fit
    scale = 0.1 * float(np.median(hfrs))
    span = float(np.ptp(positions)) or 1.0
    fits = [
        least_squares(residuals, [hfrs.min(), np.ptp(hfrs) / span, center], loss="cauchy", f_scale=scale)
        for center in positions
    ]
    center = float(min(fits, key=lambda fit: fit.cost).x[2])
    # The scan must have passed through the minimum, with a reading on each side of it
    if not positions.min() < center < positions.max():
        return None
    if not (positions < center).any() or not (positions > center).any():
        return None
    return center

def show_focus(figure, data, stars, title, interactive=True, history=None, xlabel="Frame"):
    figure.clear()
    if history:
        field, detail, trend = figure.subplots(1, 3, gridspec_kw={"width_ratios": [2, 1, 1]})
        xs, ys = zip(*history)
        trend.plot(xs, [np.nan if y is None else y for y in ys], "o-", color="tab:red")
        trend.set_xlabel(xlabel)
        trend.set_ylabel("HFR (px)")
        trend.set_title("Lower is better")
    else:
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
    detail.set_title("Star detail" if stars else "Center detail")
    figure.suptitle(title)
    figure.tight_layout()
    figure.canvas.draw_idle()
    if interactive:
        plt.pause(0.1)

def autofocus(span, step, zoom, radius, iso=None, samples=3, show=None, log=print):
    """Scan the focus through a V-curve in live view and leave the lens at its minimum HFR.

    Each measurement is passed to show(data, stars, title, history), and progress messages to log(text).
    Return the chosen step of the scan and its HFR; raise AutofocusError when no minimum is found.
    """
    # Another camera command between the scan's moves would lose track of the lens position
    with camera_lock:
        return _autofocus(span, step, zoom, radius, iso, samples, show or (lambda *_: None), log)

def _autofocus(span, step, zoom, radius, iso, samples, show, log):
    try:
        model = ensure_back_button_focus()
    except (subprocess.CalledProcessError, ValueError, IndexError) as exc:
        log(f"Could not check whether the shutter button autofocuses: {exc}")
    else:
        if model:
            log(f"Moved autofocus off the shutter button of the {model} so captures keep this focus")
    # Every measured position is reached moving toward infinity, so the lens backlash is always taken up
    move_focus(-(span + BACKLASH), step)
    move_focus(BACKLASH, step)
    positions, hfrs = [], []

    def measure(position, drive):
        # The median of a few frames outvotes a frame where noise passed for a star
        readings = []
        for sample in range(samples):
            with tempfile.TemporaryDirectory(prefix="astro-focus-") as temporary:
                path = live_preview(temporary, zoom, drive=drive if sample == 0 else 0, step=step, iso=iso)
                data, stars, hfr, problem = measure_frame(path, radius)
            if hfr is not None:
                readings.append((hfr, data, stars))
        if not readings:
            log(f"Step {position:+d}: {problem}")
            return data, stars, None, problem
        hfr = float(np.median([reading[0] for reading in readings]))
        _, data, stars = min(readings, key=lambda reading: abs(reading[0] - hfr))
        title = f"HFR {hfr:.2f} px | {len(stars)} stars | {len(readings)} of {samples} frames"
        log(f"Step {position:+d}: {title}")
        return data, stars, hfr, title

    for position in range(-span, span + 1):
        data, stars, hfr, title = measure(position, 0 if position == -span else 1)
        positions.append(position)
        hfrs.append(hfr)
        show(data, stars, f"Scanning step {position:+d} | {title}", list(zip(positions, hfrs)))
    target = fit_minimum(positions, hfrs)
    if target is None:
        # Leave the lens where it started rather than at the end of a failed scan
        move_focus(-(span + BACKLASH), step)
        move_focus(BACKLASH, step)
        raise AutofocusError(
            f"No focus minimum within {span} steps of the start: improve rough focus, "
            "aim at a brighter star, or increase --span"
        )
    target = int(round(target))
    move_focus(-(span - target + BACKLASH), step)
    move_focus(BACKLASH, step)
    # The lens is at the minimum; a frame or two may still miss the stars, so check a few times before giving up
    for _ in range(3):
        data, stars, hfr, title = measure(target, 0)
        if hfr is not None:
            break
    else:
        raise AutofocusError(f"Moved to step {target:+d} of the scan, but could not measure the focus there")
    log(f"Focused at step {target:+d} of the scan")
    show(data, stars, f"Focused at step {target:+d} | {title}", list(zip(positions, hfrs)))
    return target, hfr

def main():
    parser = argparse.ArgumentParser(
        description="Measure focus from stars in RAW, FITS, or live-view images",
        epilog="Capture settings can also be set in the [focus] table of config.toml. "
               "--live and --auto need a Canon EOS camera with live view; --auto also needs the lens switch on AF.",
    )
    parser.add_argument("files", nargs="*", type=Path, help="Saved images; omit to capture repeatedly")
    parser.add_argument("--shutterspeed", help="gphoto2 shutter speed for new captures (default: 1/4)")
    parser.add_argument("--iso", help="Optional gphoto2 ISO for new captures")
    parser.add_argument("--live", action="store_true", help="Measure zoomed live-view frames instead of RAW captures")
    parser.add_argument("--auto", action="store_true", help="Drive the lens through a V-curve in live view and stop at best focus")
    parser.add_argument("--zoom", type=int, choices=(1, 5, 10), default=5, help="Live-view zoom (default: 5)")
    parser.add_argument("--step", type=int, choices=FOCUS_STEPS, default=2, help="Lens drive step size for --auto (default: 2)")
    parser.add_argument("--span", type=int, default=8, help="Steps scanned on each side of the start for --auto (default: 8)")
    parser.add_argument("--samples", type=int, default=3, help="Live-view frames measured at each step for --auto (default: 3)")
    parser.add_argument("--interval", type=float, default=1, help="Pause between captures in seconds (default: 1)")
    parser.add_argument("--frames", type=int, default=0, help="Number of new captures; 0 repeats until stopped")
    parser.add_argument(
        "--radius", type=int,
        help="Aperture radius in binned image pixels (default: 12, scaled to the configured camera's pixel size)",
    )
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--plot", action="store_true", help="Open a preview window for saved images")
    preview.add_argument("--no-plot", action="store_true", help="Print measurements without a preview window")
    parser.add_argument("--save-preview", type=Path, help="Save an annotated PNG without requiring a GUI")
    args = parser.parse_args()
    settings = command_settings(parser, "focus", DEFAULTS)
    settings.update({key: getattr(args, key) for key in DEFAULTS if getattr(args, key) is not None})
    camera = camera_profile(parser)
    if (args.live or args.auto) and camera is not None and not camera.live_view:
        parser.error(f"--live and --auto need live view, which the {camera.name} does not have")
    if args.radius is None:
        args.radius = default_radius(camera)
    if args.interval < 0 or args.frames < 0:
        parser.error("--interval and --frames must not be negative")
    if args.radius < 3:
        parser.error("--radius must be at least 3")
    if args.save_preview and len(args.files) > 1:
        parser.error("--save-preview accepts one saved image at a time")
    if (args.live or args.auto) and args.files:
        parser.error("--live and --auto capture new frames, so omit the input files")
    if args.span < 2:
        parser.error("--span must be at least 2")
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    interactive = not args.no_plot and (not args.files or args.plot)
    figure = preview_figure(figsize=(14, 5)) if interactive else None
    if figure is None and args.save_preview:
        figure = plt.figure(figsize=(14, 5))
    best = np.inf
    count = 0
    history = []
    if args.auto:
        def show(data, stars, title, history):
            if figure is not None:
                show_focus(figure, data, stars, title, interactive, history=history, xlabel="Focus step")

        try:
            autofocus(args.span, args.step, args.zoom, args.radius, settings["iso"], args.samples, show,
                      log=lambda text: print(text, flush=True))
            if args.save_preview:
                figure.savefig(args.save_preview, dpi=150)
                print(f"  Saved {args.save_preview}")
            if interactive:
                plt.show()
        except KeyboardInterrupt:
            pass
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            raise SystemExit(str(exc)) from exc
        finally:
            if figure is not None:
                plt.close(figure)
        return
    if not args.files:
        print("Adjust focus between captures; lower HFR is better. Ctrl-C or close the window to stop")
    try:
        while args.files or not args.frames or count < args.frames:
            if figure is not None and not plt.fignum_exists(figure.number):
                break
            with tempfile.TemporaryDirectory(prefix="astro-focus-") as temporary:
                try:
                    if args.files:
                        path = args.files[count]
                    elif args.live:
                        path = live_preview(temporary, args.zoom, iso=settings["iso"])
                    else:
                        path = capture(temporary, settings["shutterspeed"], settings["iso"])
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
                data, stars, hfr, title = measure_frame(path, args.radius)
                if stars:
                    best = min(best, hfr)
                    title = f"{hfr_text(hfr, path, camera)} | {len(stars)} stars"
                    if not args.files:
                        title += f" | best {best:.2f}"
                if not args.files:
                    history = (history + [(count + 1, hfr)])[-50:]
                print(f"{path.name}: {title}", flush=True)
                if figure is not None:
                    show_focus(figure, data, stars, title, interactive=interactive, history=history)
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
