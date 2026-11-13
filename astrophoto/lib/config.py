import os
from pathlib import Path
import tomllib

OBSERVER = {"latitude": "degrees", "longitude": "degrees", "elevation": "meters"}


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


def add_observer_arguments(parser, required=True, note=""):
    path = config_path()
    try:
        config = load_config(path)
    except (OSError, tomllib.TOMLDecodeError) as error:
        parser.error(f"Could not read {path}: {error}")
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
