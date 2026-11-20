import json
import os
from pathlib import Path
import re
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


HEADER = re.compile(r"\s*\[(\[?)\s*([^\[\]]+?)\s*\]\]?\s*(#.*)?$")


def toml_value(value):
    if isinstance(value, str):
        # JSON escapes are valid in TOML basic strings
        return json.dumps(value)
    return repr(value)


def _section(lines, table):
    """Return the range of lines holding the keys of a table, or None for the top level."""
    start = 0 if table is None else None
    for index, line in enumerate(lines):
        match = HEADER.match(line)
        if not match:
            continue
        if start is not None:
            return start, index
        if not match[1] and match[2] == table:
            start = index + 1
    return None if start is None else (start, len(lines))


def _set_key(lines, table, key, value):
    line = f"{key} = {toml_value(value)}"
    section = _section(lines, table)
    if section is None:
        while lines and not lines[-1].strip():
            lines = lines[:-1]
        return [*lines, *([""] if lines else []), f"[{table}]", line]
    start, end = section
    pattern = re.compile(rf"\s*{re.escape(key)}\s*=")
    for index in range(start, end):
        if pattern.match(lines[index]):
            return [*lines[:index], line, *lines[index + 1:]]
    while end > start and not lines[end - 1].strip():
        end -= 1
    after = [""] if end < len(lines) and HEADER.match(lines[end]) else []
    return [*lines[:end], line, *after, *lines[end:]]


def update_config(text, updates):
    """Set keys in a TOML document while keeping its comments and layout.

    `updates` maps a table name, or None for the top level, to the keys to set.
    """
    lines = text.splitlines()
    for table, values in updates.items():
        for key, value in values.items():
            lines = _set_key(lines, table, key, value)
            # Dotted keys and inline tables are not edited, so check the value landed where expected
            try:
                config = tomllib.loads("\n".join(lines))
            except tomllib.TOMLDecodeError:
                config = {}
            found = config if table is None else config.get(table)
            if not isinstance(found, dict) or found.get(key) != value:
                name = key if table is None else f"{key} in [{table}]"
                raise ValueError(f"{name} is set in an unexpected way")
    return "\n".join(lines) + "\n"


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
