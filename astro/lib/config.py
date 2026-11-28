import os
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astro.lib.localtime import local_timezone

OBSERVER = {"latitude": "degrees", "longitude": "degrees", "elevation": "meters"}
# Only used when an image does not record its focal length and pixel size; 50 degrees covers a 28 mm lens
SOLVE = {"scale_low": 3.0, "scale_high": 50.0}


def config_path():
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "astro" / "config.toml"


def data_path():
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "astro"


def load_config(path=None):
    path = config_path() if path is None else path
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except FileNotFoundError:
        return {}


def observer_location(args):
    # JPL Horizons takes the elevation in km
    return {"lon": args.longitude, "lat": args.latitude, "elevation": args.elevation / 1000}


def read_config(parser):
    path = config_path()
    try:
        return load_config(path), path
    except (OSError, tomllib.TOMLDecodeError) as error:
        parser.error(f"Could not read {path}: {error}")


def add_observer_arguments(parser, required=True, note=""):
    config, path = read_config(parser)
    for name, unit in OBSERVER.items():
        value = config.get(name)
        # bool is a subclass of int, so `latitude = true` would otherwise pass as 1
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            parser.error(f"{name} in {path} must be a number")
        default = None if value is None else float(value)
        help = f"Observer {name} in {unit}{note}"
        if default is not None:
            help += f" (default: {default:g} from config)"
        parser.add_argument(f"--{name}", type=float, default=default, required=required and default is None, help=help)


def camera_timezone(parser):
    # The camera clock is synced from this computer at capture time
    config, path = read_config(parser)
    name = config.get("timezone")
    if name is None:
        return local_timezone()
    if not isinstance(name, str):
        parser.error(f"timezone in {path} must be a string")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        parser.error(f"Unknown timezone {name} in {path}")


def solve_scale(parser):
    config, path = read_config(parser)
    table = config.get("solve", {})
    if not isinstance(table, dict):
        parser.error(f"[solve] in {path} must be a table")
    scale = dict(SOLVE)
    for key, value in table.items():
        if key not in SOLVE:
            parser.error(f"Unknown setting {key} in [solve] of {path}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            parser.error(f"{key} in [solve] of {path} must be a number")
        scale[key] = float(value)
    if not 0 < scale["scale_low"] < scale["scale_high"]:
        parser.error(f"[solve] in {path} must satisfy 0 < scale_low < scale_high")
    return scale["scale_low"], scale["scale_high"]


def camera_profile(parser):
    """Return the camera named by the top-level `camera` key, or None when unset."""
    from astro.lib.camera import CAMERAS

    config, path = read_config(parser)
    name = config.get("camera")
    if name is None:
        return None
    if not isinstance(name, str):
        parser.error(f"camera in {path} must be a string")
    if name not in CAMERAS:
        parser.error(f"Unknown camera {name} in {path}; choose one of: {', '.join(CAMERAS)}")
    return CAMERAS[name]


def command_settings(parser, command, defaults):
    """Return a command's settings from its config table layered over the defaults."""
    config, path = read_config(parser)
    table = config.get(command, {})
    if not isinstance(table, dict):
        parser.error(f"[{command}] in {path} must be a table")
    settings = dict(defaults)
    for key, value in table.items():
        if key not in defaults:
            parser.error(f"Unknown setting {key} in [{command}] of {path}")
        # gphoto2 takes every setting as text, such as "1/4" or "3200"
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            parser.error(f"{key} in [{command}] of {path} must be a number or a string")
        settings[key] = str(value)
    return settings
