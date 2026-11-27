import argparse
from collections import deque
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit
import warnings

from matplotlib import image as mpimage, style
from matplotlib.figure import Figure
import numpy as np

from astrophoto import capture as capture_command, center, convert, focus, stack
from astrophoto.capture import DEFAULTS as CAPTURE_DEFAULTS, KINDS, TARGETS, capture_series, series_directory
from astrophoto.lib.camera import capture, display_limits, live_preview, move_focus, stack_display_limits
from astrophoto.lib.catalog import AmbiguousObject
from astrophoto.lib.config import (
    SOLVE, add_observer_arguments, camera_profile, camera_timezone, command_settings, observer_location, solve_scale,
)
from astrophoto.locate import PlateSolveError
from astrophoto.lib.gphoto import ENGLISH, FOCUS_STEPS, camera_lock
from astrophoto.lib.livestack import LiveStack

DEFAULTS = {"host": "0.0.0.0", "port": "8000"}
ZOOMS = (1, 5, 10)
# Seconds between camera and disk status checks
STATUS_INTERVAL = 30
# Seconds between keep-alive comments on an idle event stream
KEEPALIVE = 15
# Events about something that just happened, which a browser connecting later has no use for
TRANSIENT = {"error", "autofocus"}
# Widest image sent to browsers; larger frames are binned down
PREVIEW_WIDTH = 1600
# Seconds between saves of the live stack during a series, as writing a full frame takes a few seconds
SAVE_INTERVAL = 60


class RequestError(ValueError):
    """A request the server refuses, with the HTTP status to answer."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class Stopped(Exception):
    pass


class Hub:
    """Send events to every connected browser, keeping the latest of each kind for browsers that connect later."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()
        self.latest = {}
        self.logs = deque(maxlen=100)

    def publish(self, kind, /, **data):
        event = {"type": kind, "time": time.time(), **data}
        with self.lock:
            if kind == "log":
                self.logs.append(event)
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
        self.publish("log", text=text)

    def subscribe(self):
        client = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.add(client)
            return client, [*self.latest.values(), *self.logs]

    def unsubscribe(self, client):
        with self.lock:
            self.clients.discard(client)


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
    """Run one camera job at a time in the background, stopping it on request."""

    def __init__(self, hub):
        self.hub = hub
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
            self.hub.publish("job", name=name)
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
            self.hub.publish("job", name=None)

    def stop(self):
        with self.lock:
            name = self.name if self.thread is not None and self.thread.is_alive() else None
            self.stop_event.set()
        return name

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


def preview_png(data, low, high):
    """Return an image as a grey PNG between two levels, binned down to at most PREVIEW_WIDTH pixels wide."""
    factor = -(-data.shape[1] // PREVIEW_WIDTH)
    if factor > 1:
        height, width = data.shape[0] // factor * factor, data.shape[1] // factor * factor
        blocks = data[:height, :width].reshape(height // factor, factor, width // factor, factor)
        # Uncovered stack edges are NaN; a block with any data keeps it
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            data = np.nanmean(blocks, axis=(1, 3))
    output = io.BytesIO()
    mpimage.imsave(output, np.nan_to_num(data, nan=low), cmap="gray", vmin=low, vmax=high, format="png")
    return output.getvalue()


class App:
    def __init__(self, focus_settings, camera=None, radius=None, status_interval=STATUS_INTERVAL,
                 center_settings=None, observer=None, timezone=None, scale=None, capture_settings=None, root=None):
        self.hub = Hub()
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
        default_target = "card" if camera is None else camera.capture_target
        self.capture_settings = capture_settings or {
            kind: {**CAPTURE_DEFAULTS, **capture_command.KIND_DEFAULTS.get(kind, {}), "target": default_target}
            for kind in KINDS
        }
        self.root = Path.cwd() if root is None else Path(root)
        # Captured frames wait here to be converted, measured and stacked, so the camera never waits for them
        self.processing = queue.Queue()
        self.series_lock = threading.Lock()
        self.series = None
        self.live = None
        self.saved = 0.0
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
            "kinds": KINDS,
            "targets": list(TARGETS),
            "zooms": ZOOMS,
            "steps": FOCUS_STEPS,
        }

    # Status

    def watch_status(self):
        while not self.closing.is_set():
            self.check_status()
            self.status_wanted.wait(self.status_interval)
            self.status_wanted.clear()

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
        free = shutil.disk_usage(Path.cwd()).free
        self.hub.publish("status", camera=model, battery=battery, disk_free=free, busy=busy)

    # Focus

    def need_live_view(self):
        if self.camera is not None and not self.camera.live_view:
            raise RequestError(f"The {self.camera.name} does not have live view")

    def focus_start(self, body):
        live = bool(body.get("live", True))
        if live:
            self.need_live_view()
        zoom = choice(body, "zoom", ZOOMS, 5)
        iso = text(body, "iso", self.focus_settings["iso"])
        shutterspeed = text(body, "shutterspeed", self.focus_settings["shutterspeed"])
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
            with tempfile.TemporaryDirectory(prefix="astrophoto-focus-") as temporary:
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
        iso = text(body, "iso", self.focus_settings["iso"])
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
        if self.jobs.running() == "autofocus":
            raise RequestError("Autofocus is driving the lens; stop it first", 409)
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
            "shutterspeed": text(body, "shutterspeed", self.center_settings["shutterspeed"]),
            "iso": text(body, "iso", self.center_settings["iso"]),
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
            with tempfile.TemporaryDirectory(prefix="astrophoto-center-") as temporary:
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
            **{key: text(body, key, defaults[key]) or defaults[key] for key in ("iso", "aperture", "shutterspeed")},
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

    def capture(self, stop, directory, kind, settings):
        with self.series_lock:
            self.series = {
                "directory": directory.name, "kind": kind, "planned": settings["frames"],
                "target": settings["target"], "frames": [],
            }
        self.publish_series()
        if settings["target"] == "card":
            self.hub.log("The series is downloaded once every frame is taken, and cannot be stopped before")
        try:
            capture_series(directory, settings, log=self.hub.log, stop=stop,
                           on_frame=lambda path: self.frame_captured(path, directory, kind))
        finally:
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

    def process_frame(self, path, directory, kind, row):
        fits_path = convert.convert(path, self.timezone)
        data, stars, hfr, problem = focus.measure_frame(fits_path, self.radius)
        values = {"stars": len(stars), "background": float(np.nanmedian(data)), "status": "converted"}
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
        live.write(fits_out, png_out)
        self.live = (directory, live, len(live))
        self.saved = time.monotonic()
        self.publish_stack(fits_out.name)

    def publish_stack(self, saved):
        directory, live, count = self.live
        self.hub.publish("stack", directory=directory.name, frames=len(live), image=self.stack_image,
                         saved=saved, saved_frames=count)

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


class Handler(BaseHTTPRequestHandler):
    server_version = "astrophoto"
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
        path = urlsplit(self.path).path
        if path == "/":
            page = resources.files("astrophoto").joinpath("web/index.html").read_bytes()
            self.send(200, page, "text/html; charset=utf-8")
        elif path == "/api/config":
            self.send_json(200, self.app.config())
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
            "/api/stop": self.app.stop,
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


def addresses(host):
    """Return the host names to open the page at, finding this machine's network address for a wildcard bind."""
    if host not in ("0.0.0.0", "::", ""):
        return [host]
    names = []
    try:
        # Connecting a UDP socket sends nothing, but picks the interface that leads to the network
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            names.append(probe.getsockname()[0])
    except OSError:
        pass
    names.append(socket.gethostname())
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Serve a web app to focus, point and capture from a browser",
        epilog="Settings can also be set in the [server] table of config.toml; the tabs use the settings of the "
               "focus and center commands.",
    )
    parser.add_argument("--host", help="Address to listen on (default: 0.0.0.0, every network interface)")
    parser.add_argument("--port", type=int, help="Port to listen on (default: 8000)")
    add_observer_arguments(parser, required=False, note=", for the Center tab")
    args = parser.parse_args()
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
              timezone=camera_timezone(parser), scale=solve_scale(parser), capture_settings=capture_defaults)
    try:
        server = Server((host, port), app)
    except OSError as exc:
        raise SystemExit(f"Cannot listen on {host}:{port}: {exc}") from exc
    threading.Thread(target=app.watch_status, daemon=True).start()
    for name in addresses(host):
        print(f"Open http://{name}:{port}/", flush=True)
    print("There is no password: anyone on this network can use the camera. Ctrl-C to stop", flush=True)
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
