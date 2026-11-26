from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile
import threading

import exifread
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from astrophoto.lib.gphoto import custom_function, drive_options, get_config, run_gphoto2, set_custom_function

@dataclass(frozen=True)
class Camera:
    name: str
    pixel_size: float  # micrometres
    live_view: bool
    # Custom function that sets what the shutter button does, and the values that move autofocus off it
    back_button_focus: tuple | None = None

# Names match the EXIF model that the camera writes into its RAW files
CAMERAS = {camera.name: camera for camera in [
    Camera("Canon EOS 20D", pixel_size=6.4, live_view=False),
    # C.Fn IV-1: AF/AE lock becomes AE lock/AF, and AF/AF lock (no AE lock) becomes AE/AF (no AE lock)
    Camera("Canon EOS 600D", pixel_size=4.3, live_view=True, back_button_focus=(0x701, {0: 1, 2: 3})),
]}

def pixel_scale(camera, focal_length, binning=1):
    """Return the image scale in arcseconds per pixel."""
    return 206.265 * camera.pixel_size * binning / focal_length

def image_optics(path):
    """Return the camera model and focal length in mm recorded in a RAW or FITS image."""
    path = Path(path)
    if path.suffix.lower() in {".fit", ".fits", ".new"}:
        header = fits.getheader(path)
        model, focal = header.get("INSTRUME"), header.get("FOCALLEN")
    else:
        with path.open("rb") as file:
            tags = exifread.process_file(file, details=False)
        model, focal = tags.get("Image Model"), tags.get("EXIF FocalLength")
        model = None if model is None else str(model).strip()
        focal = None if focal is None else float(Fraction(str(focal)))
    focal = float(focal) if isinstance(focal, (int, float)) and focal > 0 else None
    return model, focal

def preview_figure(**kwargs):
    help_text = (
        "Install a GUI toolkit with 'python -m pip install PyQt6' and run from a graphical session\n"
        "If Agg was configured explicitly, use MPLBACKEND=QtAgg\n"
        "Use --no-plot for measurements without a preview window"
    )
    try:
        figure = plt.figure(**kwargs)
    # Each GUI toolkit has its own error for a missing or unreachable display, such as Tk's TclError
    except Exception as exc:
        raise SystemExit(f"Cannot open a preview window: {exc}\n{help_text}") from exc
    if figure.canvas.required_interactive_framework is None:
        plt.close(figure)
        raise SystemExit(f"Matplotlib backend '{plt.get_backend()}' cannot open a preview window\n{help_text}")
    return figure

def ensure_back_button_focus():
    """Stop the shutter button from autofocusing, so captures keep a focus set by driving the lens.

    Return the camera model when its setting was changed, or None when it was already set or is unknown.
    """
    model, functions = get_config("cameramodel", "customfuncex")
    camera = CAMERAS.get(model)
    if camera is None or camera.back_button_focus is None:
        return None
    function, changes = camera.back_button_focus
    current = custom_function(functions, function)
    if current not in changes:
        return None
    run_gphoto2(["gphoto2", "--set-config", f"customfuncex={set_custom_function(functions, function, changes[current])}"])
    return model

def live_preview(directory, zoom=None, drive=0, step=2, iso=None):
    """Grab one live-view frame, optionally zoomed and after moving the focus, and return its path."""
    directory = Path(directory).resolve()
    command = ["gphoto2", "--set-config", "viewfinder=1"]
    # With exposure simulation, the ISO sets how bright the live view looks
    if iso is not None:
        command += ["--set-config", f"iso={iso}"]
    if zoom is not None:
        command += ["--set-config", f"eoszoom={zoom}"]
    command += drive_options(drive, step)
    command += ["--capture-preview", "--force-overwrite", "--filename", "preview.jpg"]
    # gphoto2 prefixes preview names with thumb_, so run it where that name stays inside the directory
    # gphoto2 reports each saved file on stdout, which would clutter the measurements
    run_gphoto2(command, cwd=directory, stdout=subprocess.DEVNULL)
    files = list(directory.glob("*preview.jpg"))
    if len(files) != 1:
        raise RuntimeError("Expected one downloaded live-view frame")
    return files[0]

def split_frames(buffer):
    """Split complete JPEG frames off the front of a motion-JPEG byte stream, returning them and the rest."""
    frames = []
    while True:
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
        if end < 0:
            return frames, buffer[start:] if start >= 0 else b""
        frames.append(buffer[start:end + 2])
        buffer = buffer[end + 2:]

class LiveView:
    """Keep live view running and hand out its latest frame, so the mirror stays up and the exposure settles."""

    def __init__(self, zoom=None, iso=None):
        command = ["gphoto2", "--set-config", "viewfinder=1"]
        if iso is not None:
            command += ["--set-config", f"iso={iso}"]
        if zoom is not None:
            command += ["--set-config", f"eoszoom={zoom}"]
        command += ["--capture-movie", "--stdout"]
        # A file collects gphoto2's messages, since an unread pipe would eventually stall it
        self.errors = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.errors)
        except FileNotFoundError:
            raise SystemExit("gphoto2 not found")
        self.latest, self.count = None, 0
        self.ready = threading.Condition()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        buffer = b""
        while chunk := self.process.stdout.read1(1 << 16):
            frames, buffer = split_frames(buffer + chunk)
            if frames:
                with self.ready:
                    self.latest, self.count = frames[-1], self.count + len(frames)
                    self.ready.notify_all()
        with self.ready:
            self.ready.notify_all()

    def frame(self, directory, seen=0, timeout=10):
        """Save a frame newer than the `seen` count as a JPEG and return its path and the new count."""
        with self.ready:
            self.ready.wait_for(lambda: self.count > seen or not self.reader.is_alive(), timeout)
            if self.count <= seen:
                self.errors.seek(0)
                message = self.errors.read().decode(errors="replace").strip().splitlines()
                raise RuntimeError("Live view stopped" + (f": {message[-1]}" if message else ""))
            latest, count = self.latest, self.count
        path = Path(directory) / "live-view.jpg"
        path.write_bytes(latest)
        return path, count

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.errors.close()

def move_focus(steps, step=2):
    """Move the focus by a signed number of steps without taking a picture."""
    if steps:
        run_gphoto2(["gphoto2", "--set-config", "viewfinder=1", *drive_options(steps, step)], stdout=subprocess.DEVNULL)

def capture(directory, shutterspeed=None, iso=None):
    directory = Path(directory).resolve()
    command = [
        "gphoto2", "--set-config", "syncdatetime=1",
        "--set-config", "imageformat=RAW",
    ]
    if shutterspeed is not None:
        command += ["--set-config", f"shutterspeed={shutterspeed}"]
    if iso is not None:
        command += ["--set-config", f"iso={iso}"]
    command += [
        "--capture-image-and-download",
        "--filename", str(directory / "capture.%C"),
    ]
    run_gphoto2(command)
    files = list(directory.glob("capture.*"))
    if len(files) != 1:
        raise RuntimeError("Expected one downloaded RAW image")
    return files[0]

def read_image(path, bin_raw=True):
    path = Path(path)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        # Live-view frames are already demosaiced 8-bit images, so there is no Bayer mosaic to bin
        rgb = matplotlib.image.imread(path)
        data = rgb.astype(np.float32) if rgb.ndim == 2 else rgb[..., :3].astype(np.float32).mean(axis=2)
        saturated = (rgb >= 250) if rgb.ndim == 2 else (rgb[..., :3] >= 250).any(axis=2)
        return data, saturated, 1
    if path.suffix.lower() in {".fit", ".fits", ".new"}:
        data, header = fits.getdata(path, header=True)
        is_raw = "RAWFILE" in header
        black = header.get("BLACKLVL", 0)
        white = header.get("WHITELEV", np.inf)
    else:
        with rawpy.imread(str(path)) as raw:
            data = raw.raw_image_visible.copy()
            black = np.mean(raw.black_level_per_channel)
            white = raw.white_level
        is_raw = True

    if data.ndim != 2:
        raise ValueError("Expected a 2D RAW or FITS image")
    saturated = data >= white * 0.98
    data = data.astype(np.float32) - black
    scale = 1
    if is_raw and bin_raw:
        h, w = data.shape
        shape = (h // 2, 2, w // 2, 2)
        data = data[:h // 2 * 2, :w // 2 * 2].reshape(shape).mean(axis=(1, 3))
        saturated = saturated[:h // 2 * 2, :w // 2 * 2].reshape(shape).any(axis=(1, 3))
        scale = 2
    return data, saturated, scale

def display_limits(data):
    finite = data[np.isfinite(data)]
    if not finite.size:
        raise ValueError("Image contains no finite pixels")
    low, high = ZScaleInterval().get_limits(finite)
    return low, max(high, low + 1)

def stack_display_limits(data, header=None):
    if header is not None:
        try:
            low, high = float(header["DISPLO"]), float(header["DISPHI"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if np.isfinite([low, high]).all() and high > low:
                return low, high

    # Without saved limits, fall back to the full-frame stretch
    finite = data[np.isfinite(data)]
    if not finite.size:
        raise ValueError("Image contains no finite pixels")
    low, high = np.percentile(finite, [5, 99.7])
    return float(low), max(float(high), float(low) + 1e-6)
