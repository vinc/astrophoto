import argparse
import math
import sys
import tomllib
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astrophoto.lib.config import SOLVE, config_path, load_config, update_config


def number(low=-math.inf, high=math.inf, positive=False):
    def parse(text):
        value = float(text)
        if not math.isfinite(value) or not low <= value <= high or (positive and value <= 0):
            bounds = "positive" if positive else f"between {low:g} and {high:g}" if math.isfinite(low) else "finite"
            raise argparse.ArgumentTypeError(f"must be {bounds}")
        return int(value) if value.is_integer() else value
    return parse


def positive_integer(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def timezone(text):
    try:
        ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError):
        raise argparse.ArgumentTypeError(f"unknown timezone {text}")
    return text


def camera_setting(text):
    if text == "help":
        raise argparse.ArgumentTypeError("run 'astrophoto capture' with this option to list the camera's choices")
    # gphoto2 choices are strings, but plain numbers read better in the file
    for kind in (int, float):
        try:
            value = kind(text)
        except ValueError:
            continue
        if math.isfinite(value) and str(value) == text:
            return value
    return text


OPTIONS = {
    None: {
        "latitude": (number(-90, 90), "Observer latitude in degrees"),
        "longitude": (number(-180, 180), "Observer longitude in degrees"),
        "elevation": (number(), "Observer elevation in meters"),
        "timezone": (timezone, "Camera timezone, such as Europe/Paris (default: the system timezone)"),
    },
    "capture": {
        "frames": (positive_integer, "Number of frames per series"),
        "iso": (camera_setting, "ISO setting passed to gphoto2"),
        "aperture": (camera_setting, "Aperture setting passed to gphoto2"),
        "shutterspeed": (camera_setting, "Shutter speed setting passed to gphoto2"),
    },
    "solve": {
        "scale_low": (number(positive=True), "Lower bound of the plate-solving scale in arcsec per pixel"),
        "scale_high": (number(positive=True), "Upper bound of the plate-solving scale in arcsec per pixel"),
    },
}


def main():
    path = config_path()
    parser = argparse.ArgumentParser(
        description=f"Create or update the config file, or print it without options ({path})",
    )
    for table, options in OPTIONS.items():
        group = parser if table is None else parser.add_argument_group(f"[{table}]")
        for key, (kind, help) in options.items():
            group.add_argument(f"--{key.replace('_', '-')}", dest=f"{table}.{key}", metavar=key.upper(), type=kind, help=help)
    args = vars(parser.parse_args())

    try:
        text = path.read_text()
        config = tomllib.loads(text)
    except FileNotFoundError:
        text, config = "", {}
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        parser.error(f"Could not read {path}: {error}")

    updates = {}
    for table, options in OPTIONS.items():
        values = {key: args[f"{table}.{key}"] for key in options if args[f"{table}.{key}"] is not None}
        if values:
            updates[table] = values
    if not updates:
        if not text:
            raise SystemExit(f"No config file at {path}; see 'astrophoto config --help' to create one")
        sys.stdout.write(text)
        return

    solve = config.get("solve", {})
    if "solve" in updates and isinstance(solve, dict):
        scale = {**SOLVE, **solve, **updates["solve"]}
        if not scale["scale_low"] < scale["scale_high"]:
            parser.error("the plate-solving scale must satisfy scale_low < scale_high")

    try:
        text = update_config(text, updates)
    except ValueError as error:
        raise SystemExit(f"Could not update {path}: {error}; edit it by hand")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
