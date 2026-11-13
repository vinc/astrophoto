import importlib
from pathlib import Path
import subprocess
import sys

COMMANDS = {
    "weather": "Hourly weather forecast for an imaging session",
    "visible": "List Solar System bodies above the horizon",
    "focus": "Measure manual focus from stars",
    "center": "Guide manual camera pointing using a plate-solved exposure",
    "capture": "Capture light, dark, or flat frames with gphoto2",
    "convert": "Convert Canon CR2 files to FITS",
    "stack": "Align and stack FITS images",
    "locate": "Mark a Solar System body in full-frame images",
    "crop": "Crop a Solar System body from RAW or FITS images",
    "chart": "Draw a star chart locating plate-solved images",
}


def usage():
    width = max(map(len, COMMANDS))
    lines = ["usage: astrophoto <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {description}" for name, description in COMMANDS.items()]
    lines += ["", "Run 'astrophoto <command> --help' for the options of a command."]
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(usage(), file=sys.stderr)
        return 2
    command, arguments = argv[0], argv[1:]
    if command in ("-h", "--help"):
        print(usage())
        return 0
    if command not in COMMANDS:
        print(f"astrophoto: unknown command '{command}'\n\n{usage()}", file=sys.stderr)
        return 2
    if command == "capture":
        script = Path(__file__).with_name("capture.sh")
        return subprocess.run(["sh", str(script), *arguments]).returncode
    module = importlib.import_module(f"astrophoto.{command}")
    sys.argv = [f"astrophoto {command}", *arguments]
    module.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
