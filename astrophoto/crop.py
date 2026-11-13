import argparse
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import matplotlib

from astropy.io import fits
from astropy.time import Time

from astrophoto.lib.config import add_observer_arguments, observer_location
from astrophoto.locate import get_ephemerides, save_image, solve_image

def crop_image(path, radius, ra, dec, time, query, target_name=None, source_path=None):
    return save_image(
        path, ra, dec, time, query, target_name=target_name,
        radius=radius, source_path=source_path,
    )

def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(
        description="Crop a Solar System body from RAW or FITS images"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Saved RAW or FITS images, including stacks",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Solar System body name or designation",
    )
    add_observer_arguments(parser)
    parser.add_argument(
        "--radius",
        type=int,
        default=100,
        help="Crop radius in pixels (default: 100)",
    )
    parser.add_argument("--timezone", default="Europe/Paris", help="Camera clock timezone (default: Europe/Paris)")
    parser.add_argument("--scale-low", type=float, default=3, help="Minimum image width in degrees (default: 3)")
    parser.add_argument("--scale-high", type=float, default=20, help="Maximum image width in degrees (default: 20)")
    parser.add_argument("--cpu-limit", type=int, default=60, help="Plate-solving CPU limit in seconds (default: 60)")
    args = parser.parse_args()
    if not (-90 <= args.latitude <= 90 and -180 <= args.longitude <= 180):
        parser.error("Invalid observer latitude or longitude")
    if not (0 < args.scale_low < args.scale_high):
        parser.error("Scale limits must satisfy 0 < --scale-low < --scale-high")
    if args.radius <= 0 or args.cpu_limit <= 0:
        parser.error("--radius and --cpu-limit must be positive")

    observations = []
    failed = False
    timezone = ZoneInfo(args.timezone)

    with tempfile.TemporaryDirectory(prefix="astrophoto-crop-") as temporary:
        for index, path in enumerate(args.files):
            try:
                directory = Path(temporary) / str(index)
                directory.mkdir()
                solved = solve_image(
                    path, directory, timezone, args.scale_low, args.scale_high,
                    cpu_limit=args.cpu_limit, keep_solution=True,
                )
                header = fits.getheader(solved)
                time = Time(header["DATE-AVG"], format="isot", scale="utc")
                observations.append((path, solved, time))
            except Exception as exc:
                print(f"{path}: ERROR: {exc}")
                failed = True

    if not observations:
        raise SystemExit("No valid observations")

    observer = observer_location(args)

    # Horizons returns discrete epochs in chronological order
    observations.sort(key=lambda observation: observation[2].jd)
    epochs = [time.jd for _, _, time in observations]

    result = get_ephemerides(args.query, observer, epochs)
    if len(result) != len(observations):
        raise SystemExit("Horizons returned a different number of epochs than requested")

    for (path, solved, time), position in zip(observations, result):
        try:
            crop_image(
                solved,
                args.radius,
                float(position["RA"]),
                float(position["DEC"]),
                time,
                args.query,
                target_name=str(position["targetname"]),
                source_path=path,
            )
        except Exception as exc:
            print(f"{path}: ERROR: {exc}")
            failed = True

    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
