"""A library of master darks, so a session without its own dark frames can still be calibrated."""
from datetime import datetime, timezone
import hashlib
import re

import numpy as np
from astropy.io import fits

from astro.lib.config import data_path

# A dark matches lights from the same camera, at the same ISO and exposure time
KEYS = ("INSTRUME", "ISO", "EXPTIME")
# Dark current roughly doubles every 5 to 6 degrees, so darks further from the lights than this subtract poorly
TOLERANCE = 5


def median_temperature(paths):
    """Return the median sensor temperature recorded in FITS frames, or None when none records one."""
    temperatures = []
    for path in paths:
        try:
            value = fits.getheader(path).get("CCD-TEMP")
        except OSError:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            temperatures.append(float(value))
    return float(np.median(temperatures)) if temperatures else None


def temperature_warning(dark, lights):
    """Return a warning when darks were taken much warmer or colder than the lights, or None."""
    if dark is None or lights is None or abs(dark - lights) <= TOLERANCE:
        return None
    return (f"Warning: the darks were taken at {dark:g} °C and the lights at {lights:g} °C; "
            "darks within a few degrees of the lights subtract better")


def library_path():
    return data_path() / "darks"


def sources_digest(paths):
    """Identify a set of dark frames by their contents, so the same set is not saved twice.

    Names and sizes are not enough: frames of one camera have the same size, and names repeat across sessions.
    """
    digests = []
    for path in paths:
        with open(path, "rb") as file:
            digests.append(hashlib.file_digest(file, "sha256").hexdigest())
    # The same frames, listed in any order, are the same set
    return hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest()[:16]


def describe(metadata):
    """Name the camera, ISO and exposure time that a master dark matches, as far as they are known."""
    text = f"{metadata.get('INSTRUME', 'an unknown camera')} at ISO {metadata.get('ISO', '?')}"
    exposure = metadata.get("EXPTIME")
    if isinstance(exposure, (int, float)):
        text += f" and {exposure:g} s"
    return text


def save(dark, metadata, paths):
    """Add a master dark to the library and return its path, or None when it lacks what matching needs.

    A master made from the same frames as one already in the library is not saved again.
    """
    if any(key not in metadata for key in KEYS):
        return None
    digest = sources_digest(paths)
    directory = library_path()
    directory.mkdir(parents=True, exist_ok=True)
    for existing in directory.glob("*.fit"):
        try:
            if fits.getheader(existing).get("DARKSRC") == digest:
                return existing
        except OSError:
            continue
    dates = []
    for path in paths:
        try:
            dates.append(fits.getheader(path)["DATE-OBS"])
        except (OSError, KeyError):
            pass
    taken = min(dates) if dates else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    header = fits.Header()
    for key in KEYS:
        header[key] = metadata[key]
    header["DATE-OBS"] = (taken, "First dark exposure start (UTC)")
    header["NDARKS"] = (len(paths), "Number of median-combined dark frames")
    temperature = median_temperature(paths)
    if temperature is not None:
        header["CCD-TEMP"] = (temperature, "Median sensor temperature (C) of the darks")
    header["DARKSRC"] = (digest, "Digest of the dark frame names and sizes")
    for path in paths:
        header.add_history(f"Dark {path.parent.name}/{path.name}")
    camera = re.sub(r"[^A-Za-z0-9]+", "", str(metadata["INSTRUME"])) or "camera"
    stamp = re.sub(r"[^0-9]", "", taken)[:14]
    name = f"{camera}-iso{metadata['ISO']}-{metadata['EXPTIME']:g}s-{stamp}.fit".replace("/", "_")
    path = directory / name
    fits.writeto(path, np.asarray(dark, dtype=np.float32), header, overwrite=True)
    return path


def find(header, temperature=None):
    """Return the library dark matching an image header, or None.

    Among matches, the one closest to the given sensor temperature wins when temperatures are known, then the
    newest.
    """
    directory = library_path()
    if not directory.is_dir() or any(key not in header for key in KEYS):
        return None
    matches = []
    for path in directory.glob("*.fit"):
        try:
            dark = fits.getheader(path)
        except OSError:
            continue
        if all(dark.get(key) == header[key] for key in KEYS):
            recorded = dark.get("CCD-TEMP")
            known = temperature is not None and isinstance(recorded, (int, float))
            distance = abs(recorded - temperature) if known else float("inf")
            # Smallest distance first, then latest date
            matches.append((-distance, str(dark.get("DATE-OBS", "")), path))
    return max(matches)[2] if matches else None


def load(path):
    """Return a library dark's data and the metadata that lights are checked against."""
    data, header = fits.getdata(path, header=True)
    return data.astype(np.float32), {key: header[key] for key in KEYS if key in header}, header
