#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta, timezone
import locale
import os
import sys

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import EarthLocation

from localtime import parse_local_time
from night import MAX_HOURS, NIGHT, sun_altitude

API = "https://api.open-meteo.com/v1/forecast"
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


def fetch_forecast(latitude, longitude, elevation, start, hours, model):
    end = start + timedelta(hours=hours - 1)
    response = requests.get(API, timeout=60, params={
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation * 1000,
        "hourly": ",".join(VARIABLES),
        "models": model,
        "wind_speed_unit": "ms",
        "start_hour": start.strftime("%Y-%m-%dT%H:%M"),
        "end_hour": end.strftime("%Y-%m-%dT%H:%M"),
    })
    if not response.ok:
        try:
            reason = response.json().get("reason")
        except ValueError:
            reason = None
        raise SystemExit(f"Open-Meteo request failed: {reason or f'HTTP {response.status_code}'}")
    hourly = response.json()["hourly"]
    times = [datetime.fromisoformat(time).replace(tzinfo=timezone.utc) for time in hourly["time"]]
    values = {
        name: np.array([np.nan if value is None else value for value in hourly.get(name, [None] * len(times))], dtype=float)
        for name in VARIABLES
    }
    return times, values


def first_night(sun):
    """Slice of the first consecutive hours with the Sun below the night altitude"""
    dark = np.flatnonzero(sun < NIGHT)
    if not len(dark):
        return slice(0, 0)
    gaps = np.flatnonzero(np.diff(dark) > 1)
    last = dark[gaps[0]] if len(gaps) else dark[-1]
    return slice(int(dark[0]), int(last) + 1)


def integer(value):
    return "-" if np.isnan(value) else str(int(round(value)))


def rain(value):
    if np.isnan(value):
        return "-"
    # Even a trace of rain must not read as none, and round(0.5) is 0
    return ">0" if 0 < value < 0.5 else str(max(1, round(value)) if value > 0 else 0)


def date_line(times):
    line = list(f"{'Date':<{LABEL}}" + " " * (CELL * len(times) + 6))
    for i, time in enumerate(times):
        if i == 0 or time.hour == 0:
            text = f"{time:%a %d}"
            offset = LABEL + i * CELL + 1
            line[offset:offset + len(text)] = text
    return "".join(line).rstrip()


def ratings(values):
    """Red, yellow, or green for each hour, or None with missing data"""
    # Rounded like the table, so that colors agree with the numbers shown
    cloud, gust = np.round(values["cloud_cover"]), np.round(values["wind_gusts_10m"])
    spread = np.round(values["temperature_2m"]) - np.round(values["dew_point_2m"])
    rain = values["precipitation"]
    result = []
    for i in range(len(cloud)):
        if np.isnan(cloud[i]):
            result.append(None)
        elif cloud[i] > CLOUDY:
            result.append("red")
        elif cloud[i] > HAZY or rain[i] > 0 or gust[i] >= WINDY or spread[i] <= DEW_SPREAD:
            result.append("yellow")
        else:
            result.append("green")
    return result


def use_color():
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def table(times, values, color=False):
    rows = [
        ("Time (h)", [str(time.hour) for time in times]),
        ("Cloud (%)", [integer(value) for value in values["cloud_cover"]]),
        ("High (%)", [integer(value) for value in values["cloud_cover_high"]]),
        ("Mid (%)", [integer(value) for value in values["cloud_cover_mid"]]),
        ("Low (%)", [integer(value) for value in values["cloud_cover_low"]]),
        ("Rain (mm)", [rain(value) for value in values["precipitation"]]),
        ("Wind (m/s)", [integer(value) for value in values["wind_speed_10m"]]),
        ("Gust (m/s)", [integer(value) for value in values["wind_gusts_10m"]]),
        ("Temp (°C)", [integer(value) for value in values["temperature_2m"]]),
        ("Dew (°C)", [integer(value) for value in values["dew_point_2m"]]),
    ]
    colors = ratings(values) if color else [None] * len(times)
    lines = [date_line(times)]
    for label, cells in rows:
        cells = [f"{cell:>{CELL - 1}}" for cell in cells]
        if label == "Time (h)":
            cells = [f"{COLORS[name]}{cell}{RESET}" if name else cell for cell, name in zip(cells, colors)]
        lines.append((f"{label:<{LABEL}}" + "".join(cell + " " for cell in cells)).rstrip())
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Hourly weather forecast for an imaging session, from Open-Meteo"
    )
    parser.add_argument("--latitude", type=float, required=True, help="Observer latitude in degrees")
    parser.add_argument("--longitude", type=float, required=True, help="Observer longitude in degrees")
    parser.add_argument("--elevation", type=float, required=True, help="Observer elevation in km")
    parser.add_argument("--time", help='Local time in ISO format or words like "tomorrow night" (default: now)')
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
        args.latitude, args.longitude, args.elevation, hours[night.start], night.stop - night.start, args.model,
    )

    try:
        locale.setlocale(locale.LC_TIME, "")
    except locale.Error:
        pass
    for line in table([time.astimezone() for time in times], values, use_color()):
        print(line)
    if not times:
        print("  (no forecast for this period)")


if __name__ == "__main__":
    main()
