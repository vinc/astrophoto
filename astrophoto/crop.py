import argparse
from pathlib import Path
import tempfile

import matplotlib

from astropy.io import fits
from astropy.time import Time

from astrophoto.lib.config import add_observer_arguments, camera_timezone, observer_location, solve_scale
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
    args = parser.parse_args()
    if not (-90 <= args.latitude <= 90 and -180 <= args.longitude <= 180):
        parser.error("Invalid observer latitude or longitude")
    scale_low, scale_high = solve_scale(parser)
    if args.radius <= 0:
        parser.error("--radius must be positive")

    observations = []
    failed = False
    timezone = camera_timezone(parser)

    with tempfile.TemporaryDirectory(prefix="astrophoto-crop-") as temporary:
        for index, path in enumerate(args.files):
            try:
                directory = Path(temporary) / str(index)
                directory.mkdir()
                solved = solve_image(
                    path, directory, timezone, scale_low, scale_high,
                    keep_solution=True,
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
