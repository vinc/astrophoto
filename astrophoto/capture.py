import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess

from astrophoto.lib.config import read_config

KINDS = ("light", "dark", "flat")
DEFAULTS = {"frames": 20, "iso": "800", "aperture": "4.0", "shutterspeed": "5"}
CAMERA_SETTINGS = ("iso", "aperture", "shutterspeed")
# Flats are shot on a lightbox, far brighter than the sky
KIND_DEFAULTS = {"flat": {"shutterspeed": "50"}}


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _config_settings(parser, table, path, name):
    if not isinstance(table, dict):
        parser.error(f"[{name}] in {path} must be a table")
    settings = {}
    for key, value in table.items():
        if key in KINDS:
            continue
        if key not in DEFAULTS:
            parser.error(f"Unknown setting {key} in [{name}] of {path}")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            parser.error(f"{key} in [{name}] of {path} must be a number or a string")
        if key == "frames":
            if not isinstance(value, int) or value < 1:
                parser.error(f"frames in [{name}] of {path} must be an integer of at least 1")
            settings[key] = value
        else:
            settings[key] = str(value)
    return settings


def capture_settings(parser, kind, options):
    config, path = read_config(parser)
    table = config.get("capture", {})
    base = _config_settings(parser, table, path, "capture")
    overrides = _config_settings(parser, table.get(kind, {}), path, f"capture.{kind}")
    settings = {}
    for layer in (DEFAULTS, base, KIND_DEFAULTS.get(kind, {}), overrides, options):
        settings.update(layer)
    return settings


def run(command):
    result = subprocess.run(command)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Capture light, dark, or flat frames with gphoto2",
        epilog="Settings can also be set in the [capture] table of config.toml, "
               "and for one kind of frame in [capture.light], [capture.dark], or [capture.flat].",
    )
    parser.add_argument("kind", nargs="?", default="light", choices=KINDS, help="Kind of frames (default: light)")
    parser.add_argument("--frames", type=positive_integer, help="Number of frames (default: 20)")
    parser.add_argument("--iso", help="ISO setting passed to gphoto2, or 'help' to list the camera's choices (default: 800)")
    parser.add_argument("--aperture", help="Aperture setting passed to gphoto2, or 'help' to list the camera's choices (default: 4.0)")
    parser.add_argument(
        "--shutterspeed",
        help="Shutter speed setting passed to gphoto2, or 'help' to list the camera's choices (default: 5, or 50 for flat frames)",
    )
    args = parser.parse_args()
    queries = [name for name in CAMERA_SETTINGS if getattr(args, name) == "help"]
    if queries:
        if shutil.which("gphoto2") is None:
            raise SystemExit("gphoto2 not found")
        for name in queries:
            run(["gphoto2", "--get-config", name])
        return
    options = {key: getattr(args, key) for key in DEFAULTS if getattr(args, key) is not None}
    settings = capture_settings(parser, args.kind, options)
    if shutil.which("gphoto2") is None:
        raise SystemExit("gphoto2 not found")

    directory = Path(f"{datetime.now():%Y%m%d-%H%M}-{args.kind}")
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Capturing pictures in {directory} ...", flush=True)
    run([
        "gphoto2",
        "--set-config", "syncdatetime=1",
        "--set-config", "imageformat=RAW",
        "--set-config", f"iso={settings['iso']}",
        "--set-config", f"aperture={settings['aperture']}",
        "--set-config", f"shutterspeed={settings['shutterspeed']}",
        "--capture-image",
        "--interval", "1",
        "--frames", str(settings["frames"]),
    ])

    print(f"Downloading pictures to {directory} ...", flush=True)
    run([
        "gphoto2", "--folder", "/", "--recurse",
        "--filename", f"{directory}/%Y%m%d-%H%M%S.%C",
        "--get-all-files",
    ])

    print("Deleting downloaded pictures from the camera ...", flush=True)
    run(["gphoto2", "--folder", "/DCIM/100CANON", "--delete-file", f"1..{settings['frames']}"])


if __name__ == "__main__":
    main()
