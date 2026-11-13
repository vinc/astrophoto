import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import locale
import os
from pathlib import Path
import sys

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import EarthLocation

from astrophoto.lib.config import add_observer_arguments
from astrophoto.lib.localtime import parse_local_time
from astrophoto.lib.night import MAX_HOURS, NIGHT, first_night, sun_altitude

API = "https://api.open-meteo.com/v1/forecast"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "astrophoto" / "weather"
MAX_AGE = timedelta(minutes=15)  # Models update a few times a day, so a short wait loses nothing
VARIABLES = [
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "precipitation", "wind_speed_10m", "wind_gusts_10m", "temperature_2m", "dew_point_2m",
]

LABEL = 12
CELL = 4  # Three characters and a space

CLOUDY = 50  # Total cloud cover in percent above which gaps are too short for a series
HAZY = 20
WINDY = 6  # Gusts in m/s that shake a tripod at long focal lengths
DEW_SPREAD = 2  # The lens cools below the air on clear nights, so dew forms early
COLORS = {"red": "\033[31m", "yellow": "\033[33m", "green": "\033[32m"}
RESET = "\033[0m"


def parse_time(text, now=None):
    """Start of the hour in UTC"""
    if text:
        time = parse_local_time(text, now)
    else:
        time = datetime.now(timezone.utc)
    return time.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def download(params):
    try:
        response = requests.get(API, timeout=60, params=params)
    except requests.RequestException as exc:
        raise SystemExit(f"Open-Meteo request failed: {exc}")
    if not response.ok:
        try:
            reason = response.json().get("reason")
        except ValueError:
            reason = None
        raise SystemExit(f"Open-Meteo request failed: {reason or f'HTTP {response.status_code}'}")
    return response.json()


def fresh(path):
    age = datetime.now().timestamp() - path.stat().st_mtime
    return age < MAX_AGE.total_seconds()


def cached_download(params, update=False):
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"{key}.json"
    if not update and path.exists() and fresh(path):
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    data = download(params)
    CACHE.mkdir(parents=True, exist_ok=True)
    for old in CACHE.glob("*.json"):
        if not fresh(old):
            old.unlink(missing_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
    return data


def fetch_forecast(latitude, longitude, elevation, start, hours, model, update=False):
    end = start + timedelta(hours=hours - 1)
    hourly = cached_download({
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation * 1000,
        "hourly": ",".join(VARIABLES),
        "models": model,
        "wind_speed_unit": "ms",
        "start_hour": start.strftime("%Y-%m-%dT%H:%M"),
        "end_hour": end.strftime("%Y-%m-%dT%H:%M"),
    }, update)["hourly"]
    times = [datetime.fromisoformat(time).replace(tzinfo=timezone.utc) for time in hourly["time"]]
    values = {
        name: np.array([np.nan if value is None else value for value in hourly.get(name, [None] * len(times))], dtype=float)
        for name in VARIABLES
    }
    return times, values


def integer(value):
    return "-" if np.isnan(value) else str(int(round(value)))


def rain(value):
    if np.isnan(value):
        return "-"
    # Even a trace of rain must not read as none, and round(0.5) is 0
    return ">0" if 0 < value < 0.5 else str(max(1, round(value)) if value > 0 else 0)


def date_line(times):
    line = "Date"
    for i, time in enumerate(times):
        if i == 0 or time.hour == 0:
            offset = LABEL + i * CELL + 1
            line = line[:offset].ljust(offset) + f"{time:%a %d}"
    return line


def ratings(values):
    """Red, yellow, or green for each hour, or None with missing data"""
    # Rounded like the table, so that colors agree with the numbers shown
    cloud, gust = np.round(values["cloud_cover"]), np.round(values["wind_gusts_10m"])
    spread = np.round(values["temperature_2m"]) - np.round(values["dew_point_2m"])
    wet = values["precipitation"] > 0
    missing = np.isnan(cloud) | np.isnan(gust) | np.isnan(spread) | np.isnan(values["precipitation"])
    return [
        None if missing[i] else
        "red" if cloud[i] > CLOUDY else
        "yellow" if cloud[i] > HAZY or wet[i] or gust[i] >= WINDY or spread[i] <= DEW_SPREAD else
        "green"
        for i in range(len(cloud))
    ]


def use_color():
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def row(label, cells):
    return (f"{label:<{LABEL}}" + "".join(f"{cell} " for cell in cells)).rstrip()


def table(times, values, color=False):
    hours = [f"{time.hour:>{CELL - 1}}" for time in times]
    if color:
        hours = [f"{COLORS[name]}{hour}{RESET}" if name else hour for hour, name in zip(hours, ratings(values))]
    groups = [
        [
            ("Cloud (%)", [integer(value) for value in values["cloud_cover"]]),
            ("High (%)", [integer(value) for value in values["cloud_cover_high"]]),
            ("Mid (%)", [integer(value) for value in values["cloud_cover_mid"]]),
            ("Low (%)", [integer(value) for value in values["cloud_cover_low"]]),
        ],
        [
            ("Wind (m/s)", [integer(value) for value in values["wind_speed_10m"]]),
            ("Gust (m/s)", [integer(value) for value in values["wind_gusts_10m"]]),
        ],
        [
            ("Temp (°C)", [integer(value) for value in values["temperature_2m"]]),
            ("Dew (°C)", [integer(value) for value in values["dew_point_2m"]]),
        ],
        [("Rain (mm)", [rain(value) for value in values["precipitation"]])],
    ]
    separator = "-" * (LABEL + CELL * len(times) - 1)
    lines = [date_line(times), row("Time (h)", hours)]
    for group in groups:
        lines.append(separator)
        lines += [row(label, (f"{cell:>{CELL - 1}}" for cell in cells)) for label, cells in group]
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Hourly weather forecast for an imaging session, from Open-Meteo"
    )
    add_observer_arguments(parser)
    parser.add_argument("--time", help='Local time in ISO format or words like "tomorrow night" (default: now)')
    parser.add_argument("--update", action="store_true",
                        help=f"Download a fresh forecast even if one less than {MAX_AGE.total_seconds() / 60:.0f} minutes old is cached")
    parser.add_argument("--model", default="meteofrance_seamless",
                        help="Open-Meteo weather model (default: meteofrance_seamless, AROME then ARPEGE)")
    args = parser.parse_args()

    start = parse_time(args.time)
    location = EarthLocation.from_geodetic(
        args.longitude * u.deg, args.latitude * u.deg, args.elevation * u.km,
    )
    hours = [start + timedelta(hours=hour) for hour in range(MAX_HOURS)]
    night = first_night(sun_altitude(hours, location))
    if night.start == night.stop:
        raise SystemExit(f"The Sun stays above {NIGHT}° in the next {MAX_HOURS} hours")
    times, values = fetch_forecast(
        args.latitude, args.longitude, args.elevation, hours[night.start], night.stop - night.start,
        args.model, args.update,
    )
    if not times:
        raise SystemExit(f"No forecast for this night from the {args.model} model")

    try:
        locale.setlocale(locale.LC_TIME, "")
    except locale.Error:
        pass
    for line in table([time.astimezone() for time in times], values, use_color()):
        print(line)


if __name__ == "__main__":
    main()
