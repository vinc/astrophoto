import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess

from astrophoto.lib.gphoto import camera_lock, run_gphoto2
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


class CaptureError(RuntimeError):
    pass


def files_on_card():
    """Return the number of files in each camera folder that holds any."""
    # Quiet mode lists one path per file, without the translated folder headers
    result = run_gphoto2(["gphoto2", "--quiet", "--list-files"], stdout=subprocess.PIPE)
    folders = {}
    for line in filter(None, result.stdout.splitlines()):
        if not line.startswith("/"):
            raise CaptureError(f"Unexpected gphoto2 file listing: {line}")
        folder = line.rsplit("/", 1)[0] or "/"
        folders[folder] = folders.get(folder, 0) + 1
    return folders


def series_directory(kind, time=None):
    """Return the directory for a new series of frames, named after its start time and kind."""
    return Path(f"{time or datetime.now():%Y%m%d-%H%M}-{kind}")


def capture_series(directory, settings, log=print):
    """Capture a series of RAW frames onto the card, download them to a directory, and delete them from the card.

    Return the downloaded paths. Raise CaptureError, or CalledProcessError when gphoto2 fails.
    """
    directory = Path(directory)
    # The whole series must not be interleaved with other camera commands
    with camera_lock:
        # The pictures are deleted from the card by number once downloaded
        count = sum(files_on_card().values())
        if count:
            raise CaptureError(f"The camera card already holds {count} files; download or delete them before capturing")

        directory.mkdir(parents=True, exist_ok=True)
        log(f"Capturing pictures in {directory} ...")
        run_gphoto2([
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

        log(f"Downloading pictures to {directory} ...")
        before = set(directory.iterdir())
        run_gphoto2([
            "gphoto2", "--folder", "/", "--recurse",
            "--filename", f"{directory}/%Y%m%d-%H%M%S.%C",
            "--get-all-files",
        ])

        # gphoto2 exits successfully even when a picture is missing or overwritten locally
        frames = settings["frames"]
        downloaded = sorted(path for path in directory.iterdir() if path not in before and path.is_file())
        empty = sum(1 for path in downloaded if path.stat().st_size == 0)
        folders = files_on_card()
        on_card = sum(folders.values())
        if len(downloaded) != frames or empty or on_card != frames:
            details = f", {empty} of them empty" if empty else ""
            raise CaptureError(
                f"Expected {frames} pictures, found {on_card} on the camera and {len(downloaded)} downloaded{details}; "
                "left them all on the camera"
            )

        log("Deleting downloaded pictures from the camera ...")
        # The camera moves on to a new folder, such as 101CANON, when one fills up mid-series
        for folder, count in folders.items():
            run_gphoto2(["gphoto2", "--folder", folder, "--delete-file", f"1-{count}"])
    return downloaded


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
    options = {key: getattr(args, key) for key in DEFAULTS if getattr(args, key) is not None}
    settings = None if queries else capture_settings(parser, args.kind, options)
    if shutil.which("gphoto2") is None:
        raise SystemExit("gphoto2 not found")
    try:
        if queries:
            for name in queries:
                run_gphoto2(["gphoto2", "--get-config", name])
            return
        capture_series(series_directory(args.kind), settings, log=lambda text: print(text, flush=True))
    except CaptureError as exc:
        raise SystemExit(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
