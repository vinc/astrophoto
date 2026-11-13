import argparse
import sys

import requests

from astrophoto.lib.config import data_path

INDEXES = "https://data.astrometry.net/4100"
# Star, deep-sky object and constellation data, from d3-celestial (BSD-3-Clause)
CELESTIAL = "https://raw.githubusercontent.com/ofrohn/d3-celestial/7e720a3de062059d4c5400a379146a601d9010e0/data"
URLS = [f"{INDEXES}/index-41{number:02d}.fits" for number in range(7, 20)] + [
    f"{CELESTIAL}/{name}" for name in (
        "stars.6.json", "constellations.json", "constellations.lines.json",
        "starnames.json", "messier.json", "dsos.14.json", "dsonames.json",
    )
]


def download(session, url, path, progress):
    # A partial name keeps an interrupted download from looking complete
    partial = path.with_name(f"{path.name}.part")
    total = 0
    try:
        with session.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            with partial.open("wb") as file:
                for chunk in response.iter_content(1 << 20):
                    file.write(chunk)
                    if progress and total:
                        # Content-Length counts compressed bytes, as received, not the decoded chunks
                        print(f"\r{path.name} {100 * response.raw.tell() // total}%", end="", flush=True)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
        if progress and total:
            print()


def main():
    parser = argparse.ArgumentParser(description="Download plate-solving indexes, star chart and object catalog data")
    parser.parse_args()
    directory = data_path()
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Downloading data to {directory} ...")
    progress = sys.stdout.isatty()
    with requests.Session() as session:
        for url in URLS:
            path = directory / url.rsplit("/", 1)[1]
            if path.exists():
                continue
            if not progress:
                print(path.name)
            try:
                download(session, url, path, progress)
            except (requests.RequestException, OSError) as error:
                raise SystemExit(f"Could not download {url}: {error}")


if __name__ == "__main__":
    main()
