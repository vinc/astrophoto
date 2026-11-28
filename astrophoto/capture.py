import argparse
import contextlib
from datetime import datetime
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading

from astrophoto.lib.gphoto import camera_lock, run_gphoto2
from astrophoto.lib.config import camera_profile, read_config

KINDS = ("light", "dark", "flat")
# A target of None leaves the choice to the camera profile
DEFAULTS = {"frames": 20, "iso": "800", "aperture": "4.0", "shutterspeed": "5", "target": None}
# gphoto2's names for where the camera stores a capture
TARGETS = {"ram": "Internal RAM", "card": "Memory card"}
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
        elif key == "target":
            if value not in TARGETS:
                parser.error(f"target in [{name}] of {path} must be one of: {', '.join(TARGETS)}")
            settings[key] = value
        else:
            settings[key] = str(value)
    return settings


def capture_settings(parser, kind, options, camera=None):
    config, path = read_config(parser)
    table = config.get("capture", {})
    base = _config_settings(parser, table, path, "capture")
    overrides = _config_settings(parser, table.get(kind, {}), path, f"capture.{kind}")
    settings = {}
    for layer in (DEFAULTS, base, KIND_DEFAULTS.get(kind, {}), overrides, options):
        settings.update(layer)
    if settings["target"] is None:
        settings["target"] = "card" if camera is None else camera.capture_target
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


def capture_series(directory, settings, log=print, stop=None, on_frame=None):
    """Capture a series of RAW frames into a directory and return their paths.

    With settings["target"] "ram", each frame is downloaded as it is taken and passed to on_frame(path), and the
    series ends early, between frames, once the stop event is set. With "card", the whole series is shot onto the
    card, then downloaded and deleted from it. Raise CaptureError, or CalledProcessError when gphoto2 fails.
    """
    directory = Path(directory)
    if settings.get("target", "card") == "ram":
        return _capture_to_ram(directory, settings, log, stop, on_frame)
    paths = _capture_to_card(directory, settings, log)
    for path in paths:
        if on_frame is not None:
            on_frame(path)
    return paths


def _camera_settings(settings):
    return [
        "--set-config", "syncdatetime=1",
        "--set-config", f"capturetarget={TARGETS[settings.get('target', 'card')]}",
        "--set-config", "imageformat=RAW",
        "--set-config", f"iso={settings['iso']}",
        "--set-config", f"aperture={settings['aperture']}",
        "--set-config", f"shutterspeed={settings['shutterspeed']}",
    ]


def _capture_to_ram(directory, settings, log, stop, on_frame):
    frames = settings["frames"]
    directory.mkdir(parents=True, exist_ok=True)
    log(f"Capturing pictures in {directory} ...")
    paths = []
    # Each frame lands in a staging directory of its own, so files that others write into the series directory
    # meanwhile, such as the FITS frames the server converts, are never taken for the download
    staging = Path(tempfile.mkdtemp(prefix=".capture-", dir=directory))
    try:
        # One gphoto2 run per frame, so the series can stop between frames: interrupting gphoto2 during an
        # exposure cancels it and leaves the camera unresponsive until it is reconnected
        for index in range(1, frames + 1):
            if stop is not None and stop.is_set():
                log(f"Stopped after {len(paths)} of {frames} pictures")
                break
            try:
                run_gphoto2(
                    ["gphoto2", *_camera_settings(settings), "--capture-image-and-download",
                     "--filename", f"{staging}/%Y%m%d-%H%M%S-{index:03d}.%C"],
                    stdout=subprocess.DEVNULL,
                    # Keep a Ctrl-C in the terminal from reaching gphoto2
                    start_new_session=True,
                )
            except subprocess.CalledProcessError:
                log(f"Stopped after {len(paths)} of {frames} pictures")
                raise
            new = [path for path in staging.iterdir() if path.is_file()]
            if len(new) != 1 or new[0].stat().st_size == 0:
                raise CaptureError(
                    f"Expected one picture for frame {index}, found {len(new)}"
                    f"{' empty' if len(new) == 1 else ''}; stopped after {len(paths)} of {frames} pictures"
                )
            path = new[0].rename(_published(directory, new[0]))
            paths.append(path)
            log(f"Saved {path.name} ({index} of {frames})")
            if on_frame is not None:
                on_frame(path)
    finally:
        _recover(staging, directory, log)
    return paths


def _published(directory, path):
    """Return where a downloaded picture goes in the series directory."""
    # Pictures taken to RAM are named capt0000.cr2, but those from the card IMG_0001.CR2
    return directory / (path.name[:-len(path.suffix)] + path.suffix.upper())


def _recover(staging, directory, log):
    """Keep any picture left in the staging directory by a frame that failed, then remove the directory.

    gphoto2 can fail after saving a frame, while deleting it from the camera's memory for instance, and that
    frame is still a good one.
    """
    for path in sorted(staging.iterdir()):
        if path.is_file() and path.stat().st_size == 0:
            path.unlink()
            continue
        target = _published(directory, path)
        if path.is_file() and not target.exists():
            try:
                path.rename(target)
            except OSError:
                continue
            log(f"Kept {target.name} from a frame that did not finish")
    try:
        staging.rmdir()
    except OSError:
        log(f"Left the files of an unfinished frame in {staging}")


def _capture_to_card(directory, settings, log):
    # The whole series must not be interleaved with other camera commands
    with camera_lock:
        # The pictures are deleted from the card by number once downloaded
        count = sum(files_on_card().values())
        if count:
            raise CaptureError(f"The camera card already holds {count} files; download or delete them before capturing")

        directory.mkdir(parents=True, exist_ok=True)
        log(f"Capturing pictures in {directory} ...")
        run_gphoto2([
            "gphoto2", *_camera_settings(settings),
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


@contextlib.contextmanager
def stop_on_interrupt(stop):
    """Turn the first Ctrl-C into a request to stop after the current frame, and a second into an abort."""
    def interrupt(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        print("Stopping after this picture; press Ctrl-C again to abort it, "
              "which may leave the camera needing to be reconnected", file=sys.stderr, flush=True)

    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


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
    parser.add_argument(
        "--target", choices=TARGETS,
        help="Download each frame as it is taken from the camera's memory (ram), or the whole series at the end "
             "from its card (card) (default: ram for the Canon EOS 600D, card otherwise)",
    )
    args = parser.parse_args()
    queries = [name for name in CAMERA_SETTINGS if getattr(args, name) == "help"]
    options = {key: getattr(args, key) for key in DEFAULTS if getattr(args, key) is not None}
    if shutil.which("gphoto2") is None:
        raise SystemExit("gphoto2 not found")
    settings = None if queries else capture_settings(parser, args.kind, options, camera_profile(parser))
    try:
        if queries:
            for name in queries:
                run_gphoto2(["gphoto2", "--get-config", name])
            return
        stop = threading.Event()
        interrupt = stop_on_interrupt(stop) if settings["target"] == "ram" else contextlib.nullcontext()
        with interrupt:
            capture_series(series_directory(args.kind), settings, log=lambda text: print(text, flush=True), stop=stop)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except CaptureError as exc:
        raise SystemExit(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
