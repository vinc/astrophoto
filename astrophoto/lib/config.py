import os
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astrophoto.lib.localtime import local_timezone

OBSERVER = {"latitude": "degrees", "longitude": "degrees", "elevation": "meters"}
SOLVE = {"scale_low": 3.0, "scale_high": 20.0}


def config_path():
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "astrophoto" / "config.toml"


def data_path():
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "astrophoto"


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
