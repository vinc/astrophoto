"""The library of a data directory: library.toml describes each series from its frames' EXIF.

A series directory is named <date>-<time>-<kind>, and its section in library.toml records the camera, lens and
exposure settings of its frames. A dark or flat series that holds a master, dark.fit or flat.fit, can calibrate the
lights of later sessions taken with the same settings.
"""
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import statistics
import tomllib

from astropy.io import fits

from astro.convert import camera_temperature, exif_value, read_tags

NAME = "library.toml"
# Series directories as capture names them, with a number added for a second series in the same minute
SERIES = re.compile(r"\d{8}-\d{4}-(light|dark|flat)(-\d+)?")
# The master a dark or flat series keeps, beside its frames
MASTERS = {"dark": "dark.fit", "flat": "flat.fit"}
# A dark only calibrates lights taken with the same sensor settings, and a flat lights taken through the same optics
DARK_KEYS = ("camera", "iso", "shutter")
FLAT_KEYS = ("camera", "scope", "focal", "aperture")


class LibraryError(ValueError):
    pass


def kind(name):
    """Return the kind of frames of a series directory name, or None for any other name."""
    match = SERIES.fullmatch(name)
    return match.group(1) if match else None


def is_frame(path):
    """Tell whether a FITS file in a series directory is a frame, rather than a master or a stack."""
    return path.suffix == ".fit" and path.name not in MASTERS.values() and not path.name.endswith("-stack.fit")


def raw_files(directory):
    return sorted(path for path in Path(directory).iterdir() if path.suffix.lower() == ".cr2")


def master_path(root, name):
    return Path(root) / name / MASTERS[kind(name)]


def shutter_label(seconds):
    """Name an exposure time as a camera does: 1/500 below a second, 5 or 0.3 otherwise."""
    seconds = Fraction(seconds).limit_denominator(100000)
    if 0 < seconds < 1 and (1 / seconds).denominator == 1:
        return f"1/{1 / seconds}"
    return f"{float(seconds):g}"


def seconds(label):
    """Return an exposure time label as a number of seconds, or None when it is not one."""
    try:
        return Fraction(str(label))
    except (ValueError, ZeroDivisionError):
        return None


def number(value):
    """Return a whole number as an int, so library.toml says 28 rather than 28.0."""
    value = float(value)
    return int(value) if value.is_integer() else round(value, 3)


def describe(directory):
    """Describe a series from its frames: the RAW files' EXIF, or else the FITS frames' headers.

    Return None for a directory with no frames yet.
    """
    directory = Path(directory)
    raws = raw_files(directory)
    if raws:
        tags = read_tags(raws[0])
        exposure = exif_value(tags, "EXIF ExposureTime")
        values = {
            "camera": exif_value(tags, "Image Model"),
            "scope": exif_value(tags, "EXIF LensModel"),
            "focal": exif_value(tags, "EXIF FocalLength"),
            "shutter": None if exposure is None else shutter_label(Fraction(exposure)),
            "aperture": exif_value(tags, "EXIF FNumber"),
            "iso": exif_value(tags, "EXIF ISOSpeedRatings"),
        }
        temperatures = [camera_temperature(tags)] + [camera_temperature(read_tags(path)) for path in raws[1:]]
    else:
        frames = sorted(path for path in directory.glob("*.fit") if is_frame(path))
        if not frames:
            return None
        headers = [fits.getheader(path) for path in frames]
        header = headers[0]
        exposure = header.get("EXPTIME")
        values = {
            "camera": header.get("INSTRUME"),
            "scope": header.get("OBJECTIV"),
            "focal": header.get("FOCALLEN"),
            "shutter": None if exposure is None else shutter_label(exposure),
            "aperture": header.get("FNUMBER"),
            "iso": header.get("ISO"),
        }
        temperatures = [header.get("CCD-TEMP") for header in headers]
    section = {}
    for key, value in values.items():
        if value is None or str(value).strip() == "":
            continue
        if key in ("focal", "aperture", "iso"):
            try:
                value = number(Fraction(str(value)))
            except (ValueError, ZeroDivisionError):
                continue
        else:
            value = str(value).strip()
        section[key] = value
    temperatures = [value for value in temperatures if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if temperatures:
        section["temperature"] = number(statistics.median(temperatures))
    return section


def load(root):
    path = Path(root) / NAME
    try:
        with path.open("rb") as file:
            library = tomllib.load(file)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LibraryError(f"Could not read {path}: {exc}") from exc
    return {name: section for name, section in library.items() if isinstance(section, dict)}


def dumps(library):
    """Write the library as TOML, one section per series in order, each holding text and numbers."""
    lines = []
    for name in sorted(library):
        lines.append(f"[{name}]")
        for key, value in library[name].items():
            # A JSON string is a valid TOML basic string
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, str) else repr(value)
            lines.append(f"{key} = {text}")
        lines.append("")
    return "\n".join(lines)


def save(root, library):
    path = Path(root) / NAME
    temporary = path.with_name(f".{NAME}.tmp")
    temporary.write_text(dumps(library))
    temporary.replace(path)


def record(root, name):
    """Describe one series again from its frames and save its section, as when the series ends."""
    library = load(root)
    section = describe(Path(root) / name)
    if section is not None and library.get(name) != section:
        library[name] = section
        save(root, library)
    return library


def update(root, skip=()):
    """Add a section for each series directory that has frames but no section yet, and return the library.

    A series named in skip, still being captured, waits until it ends.
    """
    root = Path(root)
    library = load(root)
    added = False
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if directory.name in library or directory.name in skip or kind(directory.name) is None or not directory.is_dir():
            continue
        section = describe(directory)
        if section is not None:
            library[directory.name] = section
            added = True
    if added:
        save(root, library)
    return library


def _distance(first, second):
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (first, second)):
        return math.inf
    return abs(first - second)


def find(root, library, kind_name, light):
    """Return the name of the dark or flat series whose master suits the lights described by a section, or None.

    Darks need the same camera, ISO and shutter speed, and the closest temperature wins, then the newest.
    Flats need the same camera, lens, focal length and aperture, and the newest wins.
    """
    keys = DARK_KEYS if kind_name == "dark" else FLAT_KEYS
    matches = []
    for name, section in library.items():
        if kind(name) != kind_name or not master_path(root, name).is_file():
            continue
        if any((seconds(section.get(key)) != seconds(light.get(key))) if key == "shutter"
               else section.get(key) != light.get(key) for key in keys):
            continue
        distance = _distance(section.get("temperature"), light.get("temperature")) if kind_name == "dark" else 0
        # Names start with the date and time, so the greatest name is the newest series
        matches.append((-distance, name))
    return max(matches)[1] if matches else None
