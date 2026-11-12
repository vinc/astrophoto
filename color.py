from pathlib import Path

import numpy as np
import rawpy
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import find_objects, gaussian_filter, label

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
        # FITS converted before colour metadata was saved can read it from the CR2
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

def saturation_flags(data, header, pattern):
    shape = (data.shape[0] // 2, data.shape[1] // 2)
    if "WHITELEV" not in header:
        return np.zeros(shape, dtype=np.uint8)
    white = float(header["WHITELEV"])
    if not np.isfinite(white) or white <= 0:
        raise ValueError("Invalid WHITELEV for saturation detection")
    clipped = np.isfinite(data) & (data >= 0.98 * white)
    channels = bin_rgb(clipped, pattern) > 0
    # One byte per output pixel: R=1, either green=2, B=4
    return np.sum(channels.astype(np.uint8) * [1, 2, 4], axis=-1).astype(np.uint8)

def luminance(rgb):
    # Same weights as the grayscale 2x2 binning
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
    header["RGBNOISE"] = (float(noise), "Sky noise for the RGB rendering")
    header["RGBKNEE"] = (0.8, "Highlight shoulder start for the RGB rendering")
    header["SATFEATH"] = (1.0, "Highlight feather sigma in binned pixels")
    for index, (key, gain, low) in enumerate(zip(WHITE_BALANCE_KEYS, gains, black), 1):
        header[key] = (float(gain), "Daylight gain for the RGB rendering")
        header[f"RGBLO{index}"] = (float(low), "Background level for the RGB rendering")
    return header

def feather_saturation(saturation, sigma=1.0):
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Invalid saturation feather width")
    if sigma == 0:
        return saturation
    regions, count = label(saturation > 0, structure=np.ones((3, 3)))
    result = np.zeros(saturation.shape, dtype=np.float32)
    radius = int(np.ceil(4 * sigma))
    for index, bounds in enumerate(find_objects(regions, max_label=count), 1):
        rows = slice(max(0, bounds[0].start - radius), min(saturation.shape[0], bounds[0].stop + radius))
        columns = slice(max(0, bounds[1].start - radius), min(saturation.shape[1], bounds[1].stop + radius))
        area = (rows, columns)
        component = np.where(regions[area] == index, saturation[area], 0).astype(np.float32)
        feathered = gaussian_filter(component, sigma=sigma, mode="constant", cval=0)
        # Retain each highlight's peak correction without keeping a flat mask edge
        feathered *= component.max() / feathered.max()
        np.maximum(result[area], feathered, out=result[area])
    return np.clip(result, 0, 1)

def render_rgb(rgb, header, saturation=None):
    black = np.array([header[f"RGBLO{index}"] for index in range(1, 4)])
    gains = np.array([header[key] for key in WHITE_BALANCE_KEYS])
    valid = np.isfinite(rgb).all(axis=-1)
    balanced = (np.where(valid[..., None], rgb, 0) - black) * gains
    noise = float(header.get("RGBNOISE", 0))
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("Invalid RGB sky noise")
    if noise > 0:
        light = luminance(balanced)[..., None]
        chroma = balanced - light
        # Only sky-level pixels feed the colour blur, so stars don't tint the sky or fade
        faint = np.clip(1 - light / (10 * noise), 0, 1)
        blurred = gaussian_filter(chroma * faint, sigma=(1.5, 1.5, 0))
        weight = gaussian_filter(faint, sigma=(1.5, 1.5, 0))
        blurred = np.divide(blurred, weight, out=chroma.copy(), where=weight > 1e-3)
        # The pedestal stops sky noise being half clipped, which turns it red after white balance
        balanced = light + faint * blurred + (1 - faint) * chroma + noise
    balanced = np.maximum(balanced, 0)
    scaled = balanced / header["RGBHIGH"]
    light = luminance(scaled)
    softness = header["RGBSOFT"]
    stretched = np.arcsinh(softness * light) / np.arcsinh(softness)
    factor = np.divide(stretched, light, out=np.zeros_like(light), where=light > 0)
    result = scaled * factor[..., None]
    # Per-channel shoulder: overexposed channels roll off towards white instead of clipping hue
    knee = float(header.get("RGBKNEE", 0.8))
    if not np.isfinite(knee) or not 0 <= knee < 1:
        raise ValueError("Invalid RGB highlight knee")
    shoulder = knee - (1 - knee) * np.expm1(-np.maximum(result - knee, 0) / (1 - knee))
    result = np.where(result <= knee, result, shoulder)
    if saturation is not None:
        if saturation.shape != rgb.shape[:2] or not np.isfinite(saturation).all() or np.any((saturation < 0) | (saturation > 1)):
            raise ValueError("Invalid RGB saturation fraction")
        saturation = feather_saturation(saturation, header.get("SATFEATH", 1.0))
        # Clipped photosites lost their colour, so render them grey
        neutral = result.max(axis=-1, keepdims=True)
        result += saturation[..., None] * (neutral - result)
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

def read_saturation(path):
    with fits.open(path) as hdus:
        if "SATURATED" not in hdus:
            return None
        data = hdus["SATURATED"].data
        if data.shape != hdus[0].data.shape:
            raise ValueError(f"{path}: invalid saturation extension dimensions")
        return data.astype(np.float32)
