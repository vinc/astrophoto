import argparse
from collections import deque
import contextlib
from datetime import datetime, timezone
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit
import warnings

from astropy.io import fits
from matplotlib import image as mpimage, style
from matplotlib.figure import Figure
import numpy as np

from astro import capture as capture_command, center, convert, focus, stack, visible, weather
from astro.capture import (
    CAMERA_SETTINGS, DEFAULTS as CAPTURE_DEFAULTS, KINDS, TARGETS, capture_series, series_directory,
)
from astro.lib.camera import capture, display_limits, live_preview, move_focus, stack_display_limits
from astro.lib.catalog import AmbiguousObject
from astro.lib.config import (
    SOLVE, add_observer_arguments, camera_profile, camera_timezone, command_settings, observer_location, solve_scale,
)
from astro.locate import PlateSolveError
from astro.lib.gphoto import ENGLISH, FOCUS_STEPS, camera_lock, choice_label, parse_config
from astro.lib import library
from astro.lib.livestack import LiveStack
from astro.lib.localtime import parse_local_time
from astro.lib.system import CpuUsage, acpi_temperature, memory_percent

DEFAULTS = {"host": "0.0.0.0", "port": "8000"}
ZOOMS = (1, 5, 10)
# Seconds between checks of the camera model and battery, which only matter on battery power
STATUS_INTERVAL = 600
# Seconds between readings of the computer's CPU, memory, temperature and free disk space, which cost nothing
SYSTEM_INTERVAL = 5
# Seconds between keep-alive comments on an idle event stream
KEEPALIVE = 15
# Events about something that just happened, which a browser connecting later has no use for
TRANSIENT = {"error", "autofocus"}
# Widest image sent to browsers; larger frames are binned down
PREVIEW_WIDTH = 1600
# Seconds between saves of the live stack during a series, as writing a full frame takes a few seconds
SAVE_INTERVAL = 60
# Series directories as capture names them, with a number added for a second series in the same minute
SERIES = re.compile(r"\d{8}-\d{4}-(light|dark|flat)(-\d+)?")
# Besides a series name, darks and flats may come from the library, which picks the masters that suit the lights
CALIBRATION_CHOICES = ("library", "none")
# What the Visible tab lists unless asked otherwise, as the visible command does
VISIBLE_DEFAULTS = {"limit": 10.0, "min_altitude": 10.0}


class RequestError(ValueError):
    """A request the server refuses, with the HTTP status to answer."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class Stopped(Exception):
    pass


class Hub:
    """Send events to every connected browser, keeping the latest of each kind for browsers that connect later."""

    def __init__(self, send_logs=True):
        self.lock = threading.Lock()
        self.clients = set()
        self.latest = {}
        self.logs = deque(maxlen=200)
        # The server console has every log line, so browsers only get them for debugging
        self.send_logs = send_logs

    def publish(self, kind, /, **data):
        event = {"type": kind, "time": time.time(), **data}
        with self.lock:
            if kind == "log":
                self.logs.append(event)
                if not self.send_logs:
                    return
            elif kind not in TRANSIENT:
                self.latest[kind] = event
            for client in self.clients:
                try:
                    client.put_nowait(event)
                except queue.Full:
                    # A stalled browser misses events rather than holding up the camera
                    pass

    def log(self, text):
        print(text, flush=True)
        # Once the server's output is wrapped, printing the line is what sends it to browsers
        if getattr(sys.stdout, "hub", None) is not self:
            self.publish("log", text=text)

    def subscribe(self):
        client = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.add(client)
            return client, [*self.latest.values(), *(self.logs if self.send_logs else ())]

    def unsubscribe(self, client):
        with self.lock:
            self.clients.discard(client)


class Timestamped:
    """Wrap an output stream so each line starts with the date and time, like [2026-11-27T21:05:03] text.

    Once given a hub, each line also goes to browsers' log, except indented detail lines, such as those convert
    prints for every frame, which the page shows elsewhere. Threads print at once, and print writes a line and its
    newline separately, so each thread's unfinished line is held until its newline arrives.
    """

    def __init__(self, stream, error=False):
        self.stream = stream
        self.error = error
        self.hub = None
        self.lock = threading.Lock()
        self.partial = {}

    def write(self, text):
        with self.lock:
            thread = threading.get_ident()
            *lines, rest = (self.partial.pop(thread, "") + text).split("\n")
            for line in lines:
                self.stream.write(f"[{datetime.now():%Y-%m-%dT%H:%M:%S}] {line}\n" if line else "\n")
                if self.hub is not None and line.strip() and not line[0].isspace():
                    self.hub.publish("log", text=line, **({"error": True} if self.error else {}))
            if rest:
                self.partial[thread] = rest
            self.stream.flush()
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        # Anything else, such as encoding or fileno, is the wrapped stream's
        return getattr(self.stream, name)


class Images:
    """The latest rendered PNG of each view, numbered so browsers can tell when one changes."""

    def __init__(self):
        self.lock = threading.Lock()
        self.images = {}

    def put(self, name, data):
        with self.lock:
            version = self.images.get(name, (0, b""))[0] + 1
            self.images[name] = (version, data)
            return f"/api/image/{name}?v={version}"

    def get(self, name):
        with self.lock:
            return self.images.get(name, (0, None))[1]


class Jobs:
    """Run one job at a time in the background, stopping it on request, and announce it as an event."""

    def __init__(self, hub, event="job"):
        self.hub = hub
        self.event = event
        self.lock = threading.Lock()
        self.thread = None
        self.name = None
        self.stop_event = threading.Event()

    def running(self):
        with self.lock:
            return self.name if self.thread is not None and self.thread.is_alive() else None

    def start(self, name, function):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RequestError(f"Busy with {self.name}; stop it first", 409)
            self.stop_event = threading.Event()
            self.name = name
            self.thread = threading.Thread(target=self._run, args=(name, function, self.stop_event), daemon=True)
            self.hub.publish(self.event, name=name)
            self.thread.start()

    def _run(self, name, function, stop):
        try:
            function(stop)
        except Stopped:
            self.hub.log(f"Stopped {name}")
        # gphoto2 or solve-field being missing raises SystemExit, which would end this thread silently
        except (Exception, SystemExit) as exc:
            message = error_text(exc)
            self.hub.log(f"{name.capitalize()} failed: {message}")
            self.hub.publish("error", message=message)
        finally:
            with self.lock:
                if self.thread is threading.current_thread():
                    self.name = None
            self.hub.publish(self.event, name=None)

    def stop(self):
        with self.lock:
            name = self.name if self.thread is not None and self.thread.is_alive() else None
            self.stop_event.set()
        return name

    @contextlib.contextmanager
    def held(self):
        """Yield the running job's name, or None, keeping any other job from starting until the block ends."""
        with self.lock:
            yield self.name if self.thread is not None and self.thread.is_alive() else None

    def join(self, timeout=None):
        thread = self.thread
        if thread is not None:
            thread.join(timeout)


def error_text(exc):
    """Describe an error in one line, using gphoto2's own message when it has one."""
    if isinstance(exc, subprocess.CalledProcessError):
        lines = [line.strip(" *") for line in (exc.stderr or "").splitlines() if line.strip(" *")]
        return lines[-1] if lines else f"{Path(str(exc.cmd[0] if exc.cmd else 'command')).name} failed"
    return str(exc) or type(exc).__name__


def camera_status():
    """Return the connected camera's model and battery level, or None for each that is unknown."""
    # Status checks run every few seconds, so a missing camera is reported quietly rather than through run_gphoto2
    try:
        result = subprocess.run(
            ["gphoto2", "--get-config", "cameramodel", "--get-config", "batterylevel"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=ENGLISH, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    values = [line.split(":", 1)[1].strip() for line in result.stdout.splitlines() if line.startswith("Current:")]
    return (values + [None, None])[:2]


def camera_choices():
    """Return the connected camera's (index, label) choices for each capture setting, or None when unknown."""
    command = ["gphoto2"]
    for name in CAMERA_SETTINGS:
        command += ["--get-config", name]
    # A camera that lacks one setting fails the whole command, so these are read apart from the status
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=ENGLISH,
                                timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    settings = parse_config(result.stdout)
    if result.returncode or len(settings) != len(CAMERA_SETTINGS):
        return None
    return {name: choices for name, (_, choices) in zip(CAMERA_SETTINGS, settings)}


def render(draw, size):
    """Draw a figure with draw(figure) and return it as PNG bytes."""
    # A dark figure keeps the page from lighting up the field at night
    with style.context("dark_background"):
        figure = Figure(figsize=size)
        draw(figure)
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=80)
    return output.getvalue()


def render_focus(data, stars, title, history, xlabel):
    return render(
        lambda figure: focus.show_focus(figure, data, stars, title, interactive=False, history=history, xlabel=xlabel),
        (14, 5),
    )


def render_pointing(path, query, result):
    return render(lambda figure: center.show_pointing(figure, path, query, result, interactive=False), (10, 7))


def shrink(data):
    """Bin an image, gray or color, down to at most PREVIEW_WIDTH pixels wide."""
    factor = -(-data.shape[1] // PREVIEW_WIDTH)
    if factor == 1:
        return data
    height, width = data.shape[0] // factor * factor, data.shape[1] // factor * factor
    blocks = data[:height, :width].reshape(height // factor, factor, width // factor, factor, *data.shape[2:])
    # Uncovered stack edges are NaN; a block with any data keeps it
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


def preview_png(data, low, high):
    """Return an image as a gray PNG between two levels, small enough for a browser."""
    output = io.BytesIO()
    mpimage.imsave(output, np.nan_to_num(shrink(data), nan=low), cmap="gray", vmin=low, vmax=high, format="png")
    return output.getvalue()


def preview_file(path):
    """Return a saved PNG, gray or color, small enough for a browser."""
    output = io.BytesIO()
    mpimage.imsave(output, np.clip(shrink(mpimage.imread(path)), 0, 1), format="png")
    return output.getvalue()


class App:
    def __init__(self, focus_settings, camera=None, radius=None, status_interval=STATUS_INTERVAL,
                 center_settings=None, observer=None, timezone=None, scale=None, capture_settings=None, root=None,
                 stack_settings=None, debug=False):
        self.debug = debug
        self.hub = Hub(send_logs=debug)
        self.images = Images()
        self.jobs = Jobs(self.hub)
        self.focus_settings = focus_settings
        self.center_settings = dict(center.DEFAULTS) if center_settings is None else center_settings
        self.observer = observer
        self.timezone = timezone
        self.scale = (SOLVE["scale_low"], SOLVE["scale_high"]) if scale is None else scale
        self.camera = camera
        self.radius = focus.default_radius(camera) if radius is None else radius
        self.status_interval = status_interval
        self.closing = threading.Event()
        self.status_wanted = threading.Event()
        # The camera's choices for each capture setting, read with its status, or None while no camera answers
        self.choices = None
        default_target = "card" if camera is None else camera.capture_target
        self.capture_settings = capture_settings or {
            kind: {**CAPTURE_DEFAULTS, **capture_command.KIND_DEFAULTS.get(kind, {}), "target": default_target}
            for kind in KINDS
        }
        self.root = Path.cwd() if root is None else Path(root)
        self.stack_settings = {"deband": 1.0} if stack_settings is None else stack_settings
        # Captured frames wait here to be converted, measured and stacked, so the camera never waits for them
        self.processing = queue.Queue()
        self.convert_lock = threading.Lock()
        # Stacking only uses the computer, so it runs beside camera jobs rather than instead of them
        self.stacking = Jobs(self.hub, event="stacking")
        self.series_lock = threading.Lock()
        # library.toml is read and written from the capture, stacking and request threads
        self.library_lock = threading.Lock()
        # The series being captured, which library.toml describes once it ends
        self.capturing = None
        self.series = None
        self.live = None
        self.saved = 0.0
        # Forecasts and orbital elements are cached in files, which one request at a time keeps whole
        self.weather_lock = threading.Lock()
        self.visible_lock = threading.Lock()
        self.processor = threading.Thread(target=self.process_frames, daemon=True)
        self.processor.start()

    def config(self):
        return {
            "camera": None if self.camera is None else self.camera.name,
            "live_view": self.camera is None or self.camera.live_view,
            "focus": {
                "shutterspeed": self.focus_settings["shutterspeed"],
                "iso": self.focus_settings["iso"],
                "zoom": 5, "step": 2, "span": 8, "samples": 3, "interval": 1,
            },
            "center": {
                "shutterspeed": self.center_settings["shutterspeed"],
                "iso": self.center_settings["iso"],
                "observer": self.observer is not None,
            },
            "capture": self.capture_settings,
            "stack": self.stack_settings,
            "observer": self.observer is not None,
            "weather": {"model": weather.DEFAULT_MODEL},
            "visible": VISIBLE_DEFAULTS,
            "kinds": KINDS,
            "targets": list(TARGETS),
            "zooms": ZOOMS,
            "steps": FOCUS_STEPS,
            "debug": self.debug,
        }

    # Status

    def watch_status(self):
        while not self.closing.is_set():
            self.check_status()
            self.status_wanted.wait(self.status_interval)
            self.status_wanted.clear()

    def watch_system(self, root="/"):
        cpu = CpuUsage(root)
        while not self.closing.wait(SYSTEM_INTERVAL):
            self.check_system(cpu, root)

    def check_system(self, cpu, root="/"):
        self.hub.publish("system", cpu=cpu.percent(), memory=memory_percent(root), temperature=acpi_temperature(root),
                         disk_free=shutil.disk_usage(self.root).free)

    def check_status(self):
        model = battery = None
        busy = not camera_lock.acquire(blocking=False)
        if not busy:
            try:
                model, battery = camera_status()
            finally:
                camera_lock.release()
        previous = self.hub.latest.get("status", {})
        if busy:
            # The camera is in use, so keep what was last known of it
            model, battery = previous.get("camera"), previous.get("battery")
        self.hub.publish("status", camera=model, battery=battery, busy=busy)
        if not busy:
            # The aperture choices change with the lens, so they are read again with every status
            choices = None
            if model is not None and camera_lock.acquire(blocking=False):
                try:
                    choices = camera_choices()
                finally:
                    camera_lock.release()
            if choices != self.choices or "choices" not in self.hub.latest:
                self.choices = choices
                self.publish_choices()

    def publish_choices(self):
        """Send the camera's labels for each setting, and the defaults of each tab as labels, without indexes."""
        choices = self.choices
        settings = None if choices is None else {name: [label for _, label in choices[name]] for name in choices}

        def labeled(values, keys):
            # A value the camera does not offer is sent as configured, and refused if it is chosen
            return {key: values[key] if choices is None or values[key] is None
                    else choice_label(values[key], choices[key]) or values[key] for key in keys}

        self.hub.publish(
            "choices", settings=settings,
            capture={kind: {**values, **labeled(values, CAMERA_SETTINGS)} for kind, values in self.capture_settings.items()},
            focus=labeled(self.focus_settings, ("iso", "shutterspeed")),
            center=labeled(self.center_settings, ("iso", "shutterspeed")),
        )

    def refresh_camera(self, body):
        """Read the camera's status and choices again soon, as after changing the lens."""
        self.status_wanted.set()
        return {"ok": True}

    # Focus

    def need_live_view(self):
        if self.camera is not None and not self.camera.live_view:
            raise RequestError(f"The {self.camera.name} does not have live view")

    def focus_start(self, body):
        live = bool(body.get("live", True))
        if live:
            self.need_live_view()
        zoom = choice(body, "zoom", ZOOMS, 5)
        iso = self.camera_setting("iso", text(body, "iso", self.focus_settings["iso"]))
        shutterspeed = self.camera_setting("shutterspeed", text(body, "shutterspeed", self.focus_settings["shutterspeed"]))
        interval = number(body, "interval", 1, minimum=0)
        self.jobs.start("focus", lambda stop: self.focus_loop(stop, live, zoom, iso, shutterspeed, interval))
        return {"ok": True}

    def focus_loop(self, stop, live, zoom, iso, shutterspeed, interval):
        best = None
        history = []
        count = 0
        source = f"live view at zoom {zoom}" if live else f"RAW captures at {shutterspeed}"
        self.hub.log(f"Measuring focus from {source}; lower HFR is better")
        while not stop.is_set():
            with tempfile.TemporaryDirectory(prefix="astro-focus-") as temporary:
                try:
                    path = live_preview(temporary, zoom, iso=iso) if live else capture(temporary, shutterspeed, iso)
                except subprocess.CalledProcessError as exc:
                    self.hub.log(f"{error_text(exc)}; retrying with the next capture")
                    stop.wait(max(interval, 1))
                    continue
                data, stars, hfr, title = focus.measure_frame(path, self.radius)
                count += 1
                if hfr is not None:
                    best = hfr if best is None else min(best, hfr)
                    title = f"{focus.hfr_text(hfr, path, self.camera)} | {len(stars)} stars | best {best:.2f}"
            history = (history + [(count, hfr)])[-50:]
            image = self.images.put("focus", render_focus(data, stars, title, history, "Frame"))
            self.hub.publish("focus", title=title, hfr=hfr, best=best, stars=len(stars), frame=count, image=image)
            stop.wait(interval)

    def focus_auto(self, body):
        self.need_live_view()
        span = integer(body, "span", 8, minimum=2)
        step = choice(body, "step", FOCUS_STEPS, 2)
        samples = integer(body, "samples", 3, minimum=1)
        zoom = choice(body, "zoom", ZOOMS, 5)
        iso = self.camera_setting("iso", text(body, "iso", self.focus_settings["iso"]))
        self.jobs.start("autofocus", lambda stop: self.autofocus(stop, span, step, zoom, iso, samples))
        return {"ok": True}

    def autofocus(self, stop, span, step, zoom, iso, samples):
        def show(data, stars, title, history):
            image = self.images.put("focus", render_focus(data, stars, title, history, "Focus step"))
            hfr = history[-1][1] if history else None
            self.hub.publish("focus", title=title, hfr=hfr, stars=len(stars), image=image)
            if stop.is_set():
                raise Stopped()

        self.hub.log(f"Autofocus over {span} steps of size {step} on each side")
        try:
            target, hfr = focus.autofocus(span, step, zoom, self.radius, iso, samples, show=show, log=self.hub.log)
        except Stopped:
            self.hub.log("The lens was left where the scan stopped")
            raise
        self.hub.publish("autofocus", step=target, hfr=hfr)

    def focus_move(self, body):
        self.need_live_view()
        size = choice(body, "size", FOCUS_STEPS, 2)
        steps = integer(body, "steps", None)
        if steps is None or steps == 0 or abs(steps) > 20:
            raise RequestError("steps must be a non-zero whole number from -20 to 20")
        # A job starting between the check and the move would get the move, so no job starts until it is done
        with self.jobs.held() as running:
            if running == "autofocus":
                raise RequestError("Autofocus is driving the lens; stop it first", 409)
            # A capture releases the camera between frames, where a move would change the focus mid-series
            if running not in (None, "focus"):
                raise RequestError(f"The lens can only be moved while idle or focusing, not during {running}", 409)
            # Between two live-view frames, the camera lock lets this move in
            try:
                move_focus(steps, size)
            except subprocess.CalledProcessError as exc:
                raise RequestError(f"Could not move the focus: {error_text(exc)}", 502) from exc
        self.hub.log(f"Moved focus {'Far' if steps > 0 else 'Near'} {size} x{abs(steps)}")
        return {"ok": True}

    # Center

    def center_start(self, body):
        if self.observer is None:
            raise RequestError("Set latitude, longitude and elevation in config.toml, or pass them to the server")
        query = body.get("query")
        if query is not None and not isinstance(query, str):
            raise RequestError("query must be text")
        query = (query or "").strip() or None
        if query is None:
            ra, dec = body.get("ra"), body.get("dec")
            if ra is None or dec is None:
                raise RequestError("Give a target name, or both RA and Dec")
            ra, dec = number(body, "ra", None), number(body, "dec", None)
            if not (0 <= ra < 360 and -90 <= dec <= 90):
                raise RequestError("RA must be in [0, 360) and Dec in [-90, 90] degrees")
        else:
            ra = dec = None
        settings = {
            key: self.camera_setting(key, text(body, key, self.center_settings[key])) for key in ("shutterspeed", "iso")
        }
        loop = bool(body.get("loop", False))
        try:
            target = center.resolve_target(query, ra, dec)
        except AmbiguousObject as exc:
            raise RequestError(str(exc)) from exc
        self.jobs.start("center", lambda stop: self.center_loop(stop, target, settings, loop))
        return {"ok": True, "target": target.query}

    def center_loop(self, stop, target, settings, loop):
        near = None
        self.hub.log(f"Pointing at {target.query}" + ("; move the camera between captures" if loop else ""))
        while not stop.is_set():
            with tempfile.TemporaryDirectory(prefix="astro-center-") as temporary:
                try:
                    solved, result = center.point(
                        temporary, target, self.observer, settings, self.timezone, self.scale,
                        camera=self.camera, near=near, log=lambda text: print(text, flush=True),
                    )
                except (PlateSolveError, subprocess.CalledProcessError) as exc:
                    if not loop:
                        raise
                    # The camera may have moved further than the hinted search area
                    near = None
                    self.hub.log(f"{error_text(exc)}; retrying with the next capture")
                    stop.wait(1)
                    continue
                near = (*result["center"], center.NEAR_RADIUS)
                image = self.images.put("center", render_pointing(solved, target.query, result))
            self.hub.publish(
                "center", target=result["target_name"], title=result["title"], report=result["report"],
                pan=result["pan"], tilt=result["tilt"], inside=result["inside"], image=image,
            )
            if not loop:
                break
            stop.wait(1)

    # Capture

    def capture_start(self, body):
        kind = choice(body, "kind", KINDS, "light")
        defaults = self.capture_settings[kind]
        settings = {
            "frames": integer(body, "frames", defaults["frames"], minimum=1),
            # The camera needs every setting for a series, so an empty field falls back to the configured one
            **{key: self.camera_setting(key, text(body, key, defaults[key]) or defaults[key])
               for key in CAMERA_SETTINGS},
            "target": choice(body, "target", tuple(TARGETS), defaults["target"]),
        }
        directory = self.root / series_directory(kind)
        # Series started within the same minute would otherwise share a directory and a live stack
        base, number = directory, 1
        while directory.exists():
            number += 1
            directory = base.with_name(f"{base.name}-{number}")
        self.jobs.start("capture", lambda stop: self.capture(stop, directory, kind, settings))
        return {"ok": True, "directory": directory.name}

    def camera_setting(self, key, value):
        """Return a camera setting by its label, from a label or a gphoto2 index, when the camera's choices are known.

        None, which leaves the camera's current setting, stays None.
        """
        choices = (self.choices or {}).get(key)
        if not choices or value is None:
            return value
        label = choice_label(value, choices)
        if label is None:
            raise RequestError(f"{key} {value} is not one of the camera's choices: {', '.join(l for _, l in choices)}")
        return label

    def capture(self, stop, directory, kind, settings):
        with self.series_lock:
            self.series = {
                "directory": directory.name, "kind": kind, "planned": settings["frames"],
                "target": settings["target"], "frames": [],
            }
        self.publish_series()
        self.capturing = directory.name
        if settings["target"] == "card":
            self.hub.log("The series is downloaded once every frame is taken, and cannot be stopped before")
        try:
            capture_series(directory, settings, log=self.hub.log, stop=stop,
                           on_frame=lambda path: self.frame_captured(path, directory, kind))
        finally:
            self.capturing = None
            if directory.is_dir():
                self.record_series(directory.name)
            # Once its frames are processed, the series' stack is saved
            self.processing.put((directory,))

    def frame_captured(self, path, directory, kind):
        with self.series_lock:
            rows = self.series["frames"]
            rows.append({"name": path.name, "status": "captured"})
            row = rows[-1]
        self.publish_series()
        self.processing.put((path, directory, kind, row))

    def publish_series(self):
        with self.series_lock:
            series = json.loads(json.dumps(self.series))
        self.hub.publish("series", **series)

    def update_row(self, row, **values):
        with self.series_lock:
            row.update(values)
        self.publish_series()

    def process_frames(self):
        while True:
            item = self.processing.get()
            if item is None:
                return
            if len(item) == 1:
                self.save_stack(item[0])
                continue
            path, directory, kind, row = item
            try:
                self.process_frame(path, directory, kind, row)
            except Exception as exc:
                self.hub.log(f"Could not process {path.name}: {error_text(exc)}")
                self.update_row(row, status="failed", problem=error_text(exc))

    def convert_raw(self, path):
        """Convert a RAW file to FITS unless that is already done, one file at a time.

        Frames arriving from a capture and a calibrated stack of RAW files may otherwise convert the same file at once.
        """
        with self.convert_lock:
            fits_path = path.with_suffix(".fit")
            if fits_path.exists():
                return fits_path
            return convert.convert(path, self.timezone)

    def process_frame(self, path, directory, kind, row):
        fits_path = self.convert_raw(path)
        data, stars, hfr, problem = focus.measure_frame(fits_path, self.radius)
        values = {"stars": len(stars), "background": float(np.nanmedian(data)), "status": "converted"}
        temperature = fits.getheader(fits_path).get("CCD-TEMP")
        if isinstance(temperature, (int, float)):
            values["temperature"] = temperature
        if hfr is not None:
            values.update(hfr=hfr, hfr_text=focus.hfr_text(hfr, fits_path, self.camera))
        else:
            values["problem"] = problem
        low, high = display_limits(data)
        image = self.images.put("frame", preview_png(data, low, high))
        self.hub.publish("frame", name=fits_path.name, image=image)
        if kind == "light":
            values.update(self.stack_frame(fits_path, directory))
        self.update_row(row, **values)

    def stack_frame(self, fits_path, directory):
        """Add a light frame to the live stack of its series, returning how it went."""
        if self.live is None or self.live[0] != directory:
            # The directory, its stack, and how many frames the saved copy holds
            self.live = (directory, LiveStack(), 0)
        live = self.live[1]
        try:
            result = live.add(stack.load(fits_path), fits_path)
        except Exception as exc:
            return {"status": "not stacked", "problem": f"Could not align: {error_text(exc)}"}
        image = live.image()
        low, high = stack_display_limits(image)
        self.stack_image = self.images.put("stack", preview_png(image, low, high))
        if time.monotonic() - self.saved >= SAVE_INTERVAL:
            self.save_stack(directory)
        else:
            self.publish_stack(None)
        return {"status": "stacked", "shift": [round(value, 1) for value in result["shift"]]}

    def save_stack(self, directory):
        """Write the live stack of a series into its directory, unless it is already saved."""
        if self.live is None or self.live[0] != directory or self.live[2] == len(self.live[1]):
            return
        directory, live, _ = self.live
        prefix = f"{directory.name}-live"
        fits_out, png_out = directory / f"{prefix}-stack.fit", directory / f"{prefix}-stack.png"
        self.saved = time.monotonic()
        try:
            live.write(fits_out, png_out)
        # A full disk must not stop the frames that follow from being converted and stacked
        except Exception as exc:
            self.hub.log(f"Could not save the live stack of {directory.name}: {error_text(exc)}")
            self.publish_stack(None)
            return
        self.live = (directory, live, len(live))
        self.publish_stack(fits_out.name)

    def publish_stack(self, saved):
        directory, live, count = self.live
        self.hub.publish("stack", directory=directory.name, frames=len(live), image=self.stack_image,
                         saved=saved, saved_frames=count)

    # Calibrated stack

    def directories(self):
        """Return the series directories in the session, newest first, by kind of frame."""
        found = {kind: [] for kind in KINDS}
        for path in sorted(self.root.iterdir(), reverse=True) if self.root.is_dir() else []:
            match = SERIES.fullmatch(path.name)
            if match and path.is_dir():
                found[match.group(1)].append(path.name)
        return found

    def series_frames(self, name, kind):
        """Return the FITS frames of a series directory picked in the page, refusing any other path."""
        if name not in self.directories()[kind]:
            raise RequestError(f"No {kind} series named {name}")
        return sorted(path for path in (self.root / name).glob("*.fit") if library.is_frame(path))

    def unconverted(self, name):
        """Return the RAW files of a series directory that have no FITS frame yet."""
        return sorted(path for path in (self.root / name).iterdir()
                      if path.suffix.lower() == ".cr2" and not path.with_suffix(".fit").exists())

    def check_series(self, name, kind):
        """Refuse a series directory with neither FITS frames nor RAW files to convert."""
        if not self.series_frames(name, kind) and not self.unconverted(name):
            raise RequestError(f"No frames in {name} yet")

    def stack_start(self, body):
        lights = body.get("lights")
        if not isinstance(lights, str):
            raise RequestError("Choose a light series")
        self.check_series(lights, "light")
        darks = body.get("darks", "library")
        flats = body.get("flats", "library")
        if not isinstance(darks, str) or not isinstance(flats, str):
            raise RequestError("darks and flats must be series names, library, or none")
        for name, kind in ((darks, "dark"), (flats, "flat")):
            if name not in CALIBRATION_CHOICES:
                self.check_series(name, kind)
        rgb = bool(body.get("rgb", False))
        deband = number(body, "deband", self.stack_settings["deband"], minimum=0)
        if deband > 1:
            raise RequestError("deband must be from 0 to 1")
        options = {"rgb": rgb, "deband": deband}
        self.stacking.start("stacking", lambda stop: self.calibrated_stack(lights, darks, flats, options))
        return {"ok": True}

    def read_library(self):
        """Return library.toml, adding the series it lacks, except one still being captured."""
        with self.library_lock:
            return library.update(self.root, skip=() if self.capturing is None else (self.capturing,))

    def fill_library(self):
        """Describe in library.toml the series it lacks, such as those captured before the server started."""
        try:
            self.read_library()
        except (OSError, ValueError) as exc:
            self.hub.log(f"Could not update {library.NAME}: {error_text(exc)}")

    def record_series(self, name):
        """Describe a series that just ended in library.toml."""
        try:
            with self.library_lock:
                library.record(self.root, name)
        except (OSError, ValueError) as exc:
            self.hub.log(f"Could not add {name} to {library.NAME}: {error_text(exc)}")

    def calibration_master(self, name, kind, progress):
        """Return the master of a dark or flat series, combining its frames first when they are newer."""
        path = library.master_path(self.root, name)
        frames = self.series_frames(name, kind)
        if path.exists() and all(frame.stat().st_mtime <= path.stat().st_mtime for frame in frames):
            return path
        combine = stack.master_dark if kind == "dark" else stack.master_flat
        data, metadata = combine(frames, progress)
        stack.write_master(path, data, metadata, frames)
        self.hub.log(f"Saved the master {kind} of {name} as {name}/{path.name}")
        return path

    def calibrated_stack(self, lights, darks, flats, options):
        directory = self.root / lights
        start = time.monotonic()
        last = {"stage": None, "time": 0.0}

        def progress(stage, done, total):
            now = time.monotonic()
            # Rows are combined many times a second; a browser needs a few updates a second at most
            if stage == last["stage"] and done < total and now - last["time"] < 0.5:
                return
            last.update(stage=stage, time=now)
            self.hub.publish("stack_progress", directory=lights, stage=stage, done=done, total=total,
                             elapsed=round(now - start, 1))

        # Series captured outside the server only have their RAW files
        series = [(lights, "light")] + [(name, kind) for name, kind in ((darks, "dark"), (flats, "flat"))
                                        if name not in CALIBRATION_CHOICES]
        raw = [path for name, _ in series for path in self.unconverted(name)]
        if raw:
            self.hub.log(f"Converting {len(raw)} RAW files to FITS")
        for index, path in enumerate(raw):
            progress("Converting", index, len(raw))
            try:
                self.convert_raw(path)
            except Exception as exc:
                self.hub.log(f"Could not convert {path.name}, leaving it out: {error_text(exc)}")
        if raw:
            progress("Converting", len(raw), len(raw))
        for name, kind in series:
            if not self.series_frames(name, kind):
                raise ValueError(f"No usable frames in {name}")

        # The library picks the masters that suit the lights, and none when it has none
        chosen = {"dark": darks, "flat": flats}
        unmatched = []
        if "library" in chosen.values():
            entries = self.read_library()
            light = entries.get(lights, {})
            for kind, name in chosen.items():
                if name == "library":
                    match = library.find(self.root, entries, kind, light)
                    chosen[kind] = match or "none"
                    if match is None:
                        unmatched.append(kind)
                        self.hub.log(f"No {kind} in the library suits {lights}, so it is stacked without one")
                    else:
                        self.hub.log(f"Using the master {kind} of {match} from the library")
        calibration = [text for text in (
            None if chosen["dark"] == "none" else f"darks from {chosen['dark']}",
            None if chosen["flat"] == "none" else f"flats from {chosen['flat']}",
            f"deband {options['deband']:g}" if options["deband"] else None,
        ) if text]
        self.hub.log(f"Stacking {lights}" + (f" with {' and '.join(calibration)}" if calibration else ""))
        masters = {kind: None if name == "none" else self.calibration_master(name, kind, progress)
                   for kind, name in chosen.items()}
        stack.stack_directory(directory, progress=progress, dark_master=masters["dark"],
                              flat_master=masters["flat"], **options)
        prefix = f"{directory.name}-rgb" if options["rgb"] else directory.name
        png = directory / f"{prefix}-stack.png"
        image = self.images.put("calibrated", preview_file(png))
        self.hub.log(f"Saved {prefix}-stack.fit and .png in {lights}")
        self.hub.publish("calibrated", directory=lights, file=f"{prefix}-stack.fit", image=image,
                         darks=chosen["dark"], flats=chosen["flat"], rgb=options["rgb"], deband=options["deband"],
                         library=[kind for kind, name in (("dark", darks), ("flat", flats)) if name == "library"],
                         unmatched=unmatched)

    # Weather and visible bodies

    def observer_position(self):
        """Latitude and longitude in degrees and elevation in meters, as the weather and visible commands take"""
        if self.observer is None:
            raise RequestError("Set latitude, longitude and elevation in config.toml to see the weather and the sky")
        return self.observer["lat"], self.observer["lon"], self.observer["elevation"] * 1000

    def weather(self, query):
        position = self.observer_position()
        text_time = query_time(query)
        start = parse_time_text(text_time) if text_time else datetime.now().astimezone()
        start = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        model = query.get("model", "").strip() or weather.DEFAULT_MODEL
        with self.weather_lock:
            times, values = run_command(weather.night_forecast, *position, start, model, flag(query, "update"))
        times = [time.astimezone() for time in times]
        return {
            "model": model,
            "hours": [
                {"time": time.isoformat(), "hour": time.hour, "date": f"{time:%a %d}", "rating": rating}
                for time, rating in zip(times, weather.ratings(values))
            ],
            "groups": [[{"label": label, "cells": cells} for label, cells in group] for group in weather.rows(values)],
        }

    def visible(self, query):
        position = self.observer_position()
        text_time = query_time(query)
        start = parse_time_text(text_time) if text_time else datetime.now().astimezone()
        limit = query_number(query, "limit", VISIBLE_DEFAULTS["limit"])
        min_altitude = query_number(query, "min_altitude", VISIBLE_DEFAULTS["min_altitude"])
        if not -90 <= min_altitude <= 90:
            raise RequestError("min_altitude must be from -90 to 90 degrees")
        with self.visible_lock:
            report = run_command(visible.sky_report, *position, start, limit, min_altitude, flag(query, "update"))
        local = report["time"].astimezone()
        return {
            **report,
            "time": local.isoformat(),
            "label": f"{local:%Y-%m-%d %H:%M %Z}",
            "limit": limit,
            "min_altitude": min_altitude,
            "bodies": [{**body, "compass": visible.compass(body["az"])} for body in report["bodies"]],
        }

    def stop(self, body):
        name = self.jobs.stop()
        return {"ok": True, "stopped": name}

    def close(self, timeout=None):
        """Stop the running job after its current step, then finish processing the frames it captured."""
        self.closing.set()
        self.status_wanted.set()
        self.jobs.stop()
        self.jobs.join(timeout)
        self.processing.put(None)
        self.processor.join(timeout)
        self.stacking.join(timeout)


# Request validation

def choice(body, key, choices, default):
    value = body.get(key, default)
    if value not in choices or isinstance(value, bool):
        raise RequestError(f"{key} must be one of {', '.join(map(str, choices))}")
    return value


def integer(body, key, default, minimum=None):
    value = body.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        raise RequestError(f"{key} must be a whole number" + ("" if minimum is None else f" of at least {minimum}"))
    return value


def number(body, key, default, minimum=None):
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (minimum is not None and value < minimum):
        raise RequestError(f"{key} must be a number" + ("" if minimum is None else f" of at least {minimum}"))
    return float(value)


def text(body, key, default):
    """Return a gphoto2 setting from a request: empty means the camera's current setting."""
    value = body.get(key, default)
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise RequestError(f"{key} must be text or a number")
    return str(value)


def query_time(query):
    value = query.get("time", "").strip()
    return value or None


def parse_time_text(value):
    try:
        return parse_local_time(value)
    except SystemExit as exc:
        raise RequestError(str(exc)) from exc


def query_number(query, key, default):
    value = query.get(key, "").strip()
    if not value:
        return default
    try:
        number = float(value)
    except ValueError:
        number = float("nan")
    if not np.isfinite(number):
        raise RequestError(f"{key} must be a number")
    return number


def flag(query, key):
    return query.get(key, "").lower() in ("1", "true", "yes", "on")


def run_command(function, *args):
    """Run what a command does, turning its exit with a message, such as a failed download, into a refusal"""
    try:
        return function(*args)
    except SystemExit as exc:
        raise RequestError(str(exc), 502) from exc


class Handler(BaseHTTPRequestHandler):
    server_version = "astro"
    protocol_version = "HTTP/1.1"

    @property
    def app(self):
        return self.server.app

    def log_message(self, format, *args):
        # Requests are frequent and routine; errors are reported through the event log instead
        pass

    def send(self, status, body, content_type, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, value):
        self.send(status, json.dumps(value).encode(), "application/json")

    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        queries = {"/api/weather": self.app.weather, "/api/visible": self.app.visible}
        if path in queries:
            query = {key: values[-1] for key, values in parse_qs(url.query).items()}
            try:
                self.send_json(200, queries[path](query))
            except RequestError as exc:
                self.send_json(exc.status, {"error": str(exc)})
        elif path == "/":
            page = resources.files("astro").joinpath("web/index.html").read_bytes()
            self.send(200, page, "text/html; charset=utf-8")
        elif path == "/api/config":
            self.send_json(200, self.app.config())
        elif path == "/api/directories":
            self.send_json(200, self.app.directories())
        elif path == "/api/events":
            self.stream_events()
        elif path.startswith("/api/image/"):
            image = self.app.images.get(path.removeprefix("/api/image/"))
            if image is None:
                self.send_json(404, {"error": "No image yet"})
            else:
                # Each version has its own URL, so browsers may keep it
                self.send(200, image, "image/png", cache=True)
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        routes = {
            "/api/focus/start": self.app.focus_start,
            "/api/focus/auto": self.app.focus_auto,
            "/api/focus/move": self.app.focus_move,
            "/api/center/start": self.app.center_start,
            "/api/capture/start": self.app.capture_start,
            "/api/stack/start": self.app.stack_start,
            "/api/stop": self.app.stop,
            "/api/camera/refresh": self.app.refresh_camera,
        }
        route = routes.get(urlsplit(self.path).path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if route is None:
                raise RequestError("Not found", 404)
            try:
                body = json.loads(raw or b"{}")
            except ValueError as exc:
                raise RequestError("The request body must be JSON") from exc
            if not isinstance(body, dict):
                raise RequestError("The request body must be a JSON object")
            self.send_json(200, route(body))
        except RequestError as exc:
            self.send_json(exc.status, {"error": str(exc)})

    def stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        client, backlog = self.app.hub.subscribe()
        try:
            for event in backlog:
                self.write_event(event)
            while not self.app.closing.is_set():
                try:
                    event = client.get(timeout=KEEPALIVE)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                else:
                    self.write_event(event)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.app.hub.unsubscribe(client)
            self.close_connection = True

    def write_event(self, event):
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)


def address(host):
    """Return the address to open the page at, finding this machine's network address for a wildcard bind."""
    if host not in ("0.0.0.0", "::", ""):
        return host
    try:
        # Connecting a UDP socket sends nothing, but picks the interface that leads to the network
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            return probe.getsockname()[0]
    except OSError:
        return socket.gethostname()


def main():
    parser = argparse.ArgumentParser(
        description="Serve a web app to focus, point and capture from a browser",
        epilog="Settings can also be set in the [server] table of config.toml; the tabs use the settings of the "
               "focus and center commands.",
    )
    parser.add_argument("--host", help="Address to listen on (default: 0.0.0.0, every network interface)")
    parser.add_argument("--port", type=int, help="Port to listen on (default: 8000)")
    parser.add_argument("--debug", action="store_true", help="Show the server's log in the page")
    add_observer_arguments(parser, required=False, note=", for the Center tab")
    args = parser.parse_args()
    # Everything the server and the commands it runs print, from any thread, is dated
    sys.stdout, sys.stderr = Timestamped(sys.stdout), Timestamped(sys.stderr, error=True)
    settings = command_settings(parser, "server", DEFAULTS)
    host = args.host if args.host is not None else settings["host"]
    try:
        port = args.port if args.port is not None else int(settings["port"])
    except ValueError:
        parser.error("port in [server] must be a whole number")
    if not 0 < port < 65536:
        parser.error("port must be from 1 to 65535")
    focus_settings = command_settings(parser, "focus", focus.DEFAULTS)
    center_settings = command_settings(parser, "center", center.DEFAULTS)
    camera = camera_profile(parser)
    observer = None
    if None not in (args.latitude, args.longitude, args.elevation):
        if not (-90 <= args.latitude <= 90 and -180 <= args.longitude <= 180):
            parser.error("Invalid observer latitude or longitude")
        observer = observer_location(args)
    else:
        print("The observer position is not set, so the Center tab is off", flush=True)
    for tool in ("gphoto2", "solve-field"):
        if shutil.which(tool) is None:
            print(f"{tool} not found: the page will load, but commands that need it will fail", flush=True)

    capture_defaults = {kind: capture_command.capture_settings(parser, kind, {}, camera) for kind in KINDS}

    app = App(focus_settings, camera, center_settings=center_settings, observer=observer,
              timezone=camera_timezone(parser), scale=solve_scale(parser), capture_settings=capture_defaults,
              stack_settings=stack.stack_settings(parser), debug=args.debug)
    # From now on, what the server and the commands it runs print also reaches the page's log
    sys.stdout.hub = sys.stderr.hub = app.hub
    try:
        server = Server((host, port), app)
    except OSError as exc:
        raise SystemExit(f"Cannot listen on {host}:{port}: {exc}") from exc
    threading.Thread(target=app.watch_status, daemon=True).start()
    threading.Thread(target=app.watch_system, daemon=True).start()
    threading.Thread(target=app.fill_library, daemon=True).start()
    print(f"Listening on http://{address(host)}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Finishing the current frame and processing captured frames ...", flush=True)
        app.close(timeout=120)
        server.server_close()


if __name__ == "__main__":
    main()
