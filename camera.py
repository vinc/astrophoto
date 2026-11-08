from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import rawpy
from astropy.io import fits
from astropy.visualization import ZScaleInterval

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

def capture(directory, shutterspeed=None):
    directory = Path(directory).resolve()
    command = [
        "gphoto2", "--set-config", "syncdatetime=1",
        "--set-config", "imageformat=RAW",
    ]
    if shutterspeed is not None:
        command += ["--set-config", f"shutterspeed={shutterspeed}"]
    command += [
        "--capture-image-and-download",
        "--filename", str(directory / "capture.%C"),
    ]
    subprocess.run(command, check=True)
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
