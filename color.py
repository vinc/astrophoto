from pathlib import Path

import numpy as np
import rawpy
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
WHITE_BALANCE_KEYS = ("WBR", "WBG", "WBB")

def raw_color_metadata(raw):
    colours = raw.color_desc.decode("ascii")
    pattern = "".join(colours[int(index)] for index in raw.raw_colors_visible[:2, :2].flat)
    if pattern not in BAYER_PATTERNS:
        return {}
    metadata = {"BAYERPAT": pattern}
    gains = np.array([raw.daylight_whitebalance[colours.index(colour)] for colour in "RGB"], dtype=float)
    if np.isfinite(gains).all() and np.all(gains > 0):
        metadata.update(zip(WHITE_BALANCE_KEYS, gains / gains[1]))
    return metadata

def color_metadata(path, header, pattern=None):
    metadata = {key: header[key] for key in ("BAYERPAT", *WHITE_BALANCE_KEYS) if key in header}
    if "BAYERPAT" not in metadata or not all(key in metadata for key in WHITE_BALANCE_KEYS):
        # Older converted FITS files can recover colour metadata from the original
        original = Path(path).with_name(Path(header.get("RAWFILE", "missing.CR2")).name)
        if original.is_file() and original.suffix.lower() == ".cr2":
            with rawpy.imread(str(original)) as raw:
                metadata = {**raw_color_metadata(raw), **metadata}
    if "BAYERPAT" not in metadata and header.get("INSTRUME") == "Canon EOS 20D":
        # Visible RGGB layout and LibRaw daylight coefficients for legacy 20D FITS
        metadata["BAYERPAT"] = "RGGB"
        for key, value in zip(WHITE_BALANCE_KEYS, [2.09863424 / 0.93013895, 1, 1.18007076 / 0.93013895]):
            metadata.setdefault(key, value)
    if pattern is not None:
        metadata["BAYERPAT"] = pattern
    if metadata.get("BAYERPAT") not in BAYER_PATTERNS:
        raise ValueError(f"{path}: missing or invalid BAYERPAT; reconvert the RAW or use --bayer-pattern")
    for key in WHITE_BALANCE_KEYS:
        metadata.setdefault(key, 1.0)
        if not np.isfinite(metadata[key]) or metadata[key] <= 0:
            raise ValueError(f"{path}: invalid daylight white balance {key}")
    return metadata

def bin_rgb(data, pattern):
    if pattern not in BAYER_PATTERNS:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    h, w = data.shape
    data = data[:h // 2 * 2, :w // 2 * 2]
    channels = []
    for colour in "RGB":
        planes = [data[index // 2::2, index % 2::2]
                  for index, value in enumerate(pattern) if value == colour]
        channels.append(np.mean(planes, axis=0))
    return np.stack(channels, axis=-1).astype(np.float32)

def luminance(rgb):
    # Match the existing average of one red, two green, and one blue sample
    return (rgb[..., 0] + 2 * rgb[..., 1] + rgb[..., 2]) / 4

def rgb_display_header(rgb, metadata):
    valid = np.isfinite(rgb).all(axis=-1)
    if not valid.any():
        raise ValueError("RGB image contains no finite pixels")
    step = max(1, int(np.sqrt(rgb.shape[0] * rgb.shape[1] / 100000)))
    sample = rgb[::step, ::step]
    sample = sample[np.isfinite(sample).all(axis=-1)]
    if not len(sample):
        sample = rgb[valid]
    black = np.array([sigma_clipped_stats(sample[:, channel])[1] for channel in range(3)])
    gains = np.array([metadata[key] for key in WHITE_BALANCE_KEYS])
    light = luminance((sample - black) * gains)
    _, _, noise = sigma_clipped_stats(light)
    high = max(float(np.percentile(light, 99.9)), 15 * float(noise), 1e-6)
    header = fits.Header({"COLORSPC": "RGB", "RGBHIGH": high, "RGBSOFT": 3.0})
    for index, (key, gain, low) in enumerate(zip(WHITE_BALANCE_KEYS, gains, black), 1):
        header[key] = (float(gain), "Daylight gain used only for RGB previews")
        header[f"RGBLO{index}"] = (float(low), "Background level used only for RGB previews")
    return header

def render_rgb(rgb, header):
    black = np.array([header[f"RGBLO{index}"] for index in range(1, 4)])
    gains = np.array([header[key] for key in WHITE_BALANCE_KEYS])
    valid = np.isfinite(rgb).all(axis=-1)
    balanced = np.maximum((np.where(valid[..., None], rgb, 0) - black) * gains, 0)
    scaled = balanced / header["RGBHIGH"]
    light = luminance(scaled)
    softness = header["RGBSOFT"]
    stretched = np.arcsinh(softness * light) / np.arcsinh(softness)
    factor = np.divide(stretched, light, out=np.zeros_like(light), where=light > 0)
    result = scaled * factor[..., None]
    # Compress bright pixels together so clipping does not change their hue
    result /= np.maximum(1, result.max(axis=-1))[..., None]
    result[~valid] = 0
    return np.dstack((np.clip(result, 0, 1), valid.astype(float))).astype(np.float32)

def read_rgb(path):
    with fits.open(path) as hdus:
        if "RGB" not in hdus:
            return None, None
        data = hdus["RGB"].data
        if data.ndim != 3 or data.shape[0] != 3 or data.shape[1:] != hdus[0].data.shape:
            raise ValueError(f"{path}: invalid RGB extension dimensions")
        return np.moveaxis(data, 0, -1).astype(np.float32), hdus["RGB"].header.copy()
