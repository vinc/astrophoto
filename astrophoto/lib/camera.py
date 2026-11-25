from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import subprocess

import exifread
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from astropy.io import fits
from astropy.visualization import ZScaleInterval

@dataclass(frozen=True)
class Camera:
    name: str
    pixel_size: float  # micrometres
    live_view: bool

# Names match the EXIF model that the camera writes into its RAW files
CAMERAS = {camera.name: camera for camera in [
    Camera("Canon EOS 20D", pixel_size=6.4, live_view=False),
    Camera("Canon EOS 600D", pixel_size=4.3, live_view=True),
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
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(f"Cannot open a preview window: {exc}\n{help_text}") from exc
    if figure.canvas.required_interactive_framework is None:
        plt.close(figure)
        raise SystemExit(f"Matplotlib backend '{plt.get_backend()}' cannot open a preview window\n{help_text}")
    return figure

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
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        raise SystemExit("gphoto2 not found")
    files = list(directory.glob("capture.*"))
    if len(files) != 1:
        raise RuntimeError("Expected one downloaded RAW image")
    return files[0]

def read_image(path, bin_raw=True):
    path = Path(path)
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
