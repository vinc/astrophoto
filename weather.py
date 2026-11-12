#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time
from astropy.utils import iers

API = "https://api.open-meteo.com/v1/forecast"
VARIABLES = [
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m",
    "temperature_2m", "dew_point_2m",
]
COMPASS = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()

# On clear nights the lens radiates heat away and ends up a few degrees below
# the air, so dew forms well before the air itself reaches its dew point
DEW_SPREAD = 2
DEW_RISK_SPREAD = 4

LABEL = 12
CELL = 4  # Three characters and a space
BLOCK = 24  # Hours per block to fit a terminal


def parse_time(text):
    time = datetime.fromisoformat(text) if text else datetime.now(timezone.utc)
    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    return time.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def fetch_forecast(latitude, longitude, elevation, start, hours, model):
    end = start + timedelta(hours=hours - 1)
    response = requests.get(API, timeout=60, params={
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation * 1000,
        "hourly": ",".join(VARIABLES),
        "models": model,
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


def sun_and_moon(times, location):
    """Sun and Moon altitudes (degrees) and Moon illuminated fraction"""
    time = Time(times)
    # Bundled Earth rotation data is plenty for this, even when slightly out of date
    with iers.conf.set_temp("auto_download", False), \
            iers.conf.set_temp("auto_max_age", None), \
            iers.conf.set_temp("iers_degraded_accuracy", "warn"):
        frame = AltAz(obstime=time, location=location)
        sun = get_body("sun", time, location)
        moon = get_body("moon", time, location)
        elongation = sun.separation(moon).deg
        return (
            sun.transform_to(frame).alt.deg,
            moon.transform_to(frame).alt.deg,
            (1 - np.cos(np.radians(elongation))) / 2,
        )


def dew_risk(temperature, dew_point):
    spread = temperature - dew_point
    if np.isnan(spread) or spread > DEW_RISK_SPREAD:
        return ""
    kind = "ice" if dew_point <= 0 else "dew"
    return kind if spread <= DEW_SPREAD else kind[0] + "?"


def integer(value):
    return "-" if np.isnan(value) else str(int(round(value)))


def decimal(value):
    """One decimal when it fits in a cell"""
    if np.isnan(value):
        return "-"
    value = round(value, 1)
    return "0" if value == 0 else f"{value:.1f}" if 0 < value < 10 else str(int(round(value)))


def compass(direction):
    return "-" if np.isnan(direction) else COMPASS[int((direction + 11.25) // 22.5) % 16]


def date_line(times):
    line = list(f"{'Date':<{LABEL}}" + " " * (CELL * len(times) + 6))
    for i, time in enumerate(times):
        if i == 0 or time.hour == 0:
            text = f"{time:%a %d}"
            offset = LABEL + i * CELL + 1
            line[offset:offset + len(text)] = text
    return "".join(line).rstrip()


def table(times, values, sun, moon, illumination, max_cloud):
    temperature, dew_point = values["temperature_2m"], values["dew_point_2m"]
    clear = (sun < -12) & (values["cloud_cover"] <= max_cloud) & ~(values["precipitation"] > 0)
    up = moon > 0
    rows = [
        ("Time (h)", [str(time.hour) for time in times]),
        ("Clear", ["*" if hour else "" for hour in clear]),
        ("Sun (°)", [integer(altitude) for altitude in sun]),
        ("Moon (°)", [integer(altitude) if above else "" for altitude, above in zip(moon, up)]),
        ("Moon (%)", [integer(lit * 100) if above else "" for lit, above in zip(illumination, up)]),
        ("Cloud (%)", [integer(value) for value in values["cloud_cover"]]),
        ("High (%)", [integer(value) for value in values["cloud_cover_high"]]),
        ("Mid (%)", [integer(value) for value in values["cloud_cover_mid"]]),
        ("Low (%)", [integer(value) for value in values["cloud_cover_low"]]),
        ("Rain (mm)", [decimal(value) for value in values["precipitation"]]),
        ("Wind (km/h)", [integer(value) for value in values["wind_speed_10m"]]),
        ("Gust (km/h)", [integer(value) for value in values["wind_gusts_10m"]]),
        ("Wind dir", [compass(value) for value in values["wind_direction_10m"]]),
        ("Temp (°C)", [integer(value) for value in temperature]),
        ("Dew pt (°C)", [integer(value) for value in dew_point]),
        ("Spread (°C)", [decimal(value) for value in temperature - dew_point]),
        ("Dew risk", [dew_risk(*pair) for pair in zip(temperature, dew_point)]),
    ]
    lines = []
    for first in range(0, len(times), BLOCK):
        block = slice(first, first + BLOCK)
        if lines:
            lines.append("")
        lines.append(date_line(times[block]))
        for label, cells in rows:
            lines.append((f"{label:<{LABEL}}" + "".join(f"{cell:>{CELL - 1}} " for cell in cells[block])).rstrip())
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Hourly weather forecast for an imaging session, from Open-Meteo"
    )
    parser.add_argument("--latitude", type=float, required=True, help="Observer latitude in degrees")
    parser.add_argument("--longitude", type=float, required=True, help="Observer longitude in degrees")
    parser.add_argument("--elevation", type=float, required=True, help="Observer elevation in km")
    parser.add_argument("--time", help="UTC start time in ISO format (default: now)")
    parser.add_argument("--hours", type=int, default=24, help="Number of hours to forecast (default: 24)")
    parser.add_argument("--model", default="meteofrance_seamless",
                        help="Open-Meteo weather model (default: meteofrance_seamless, AROME then ARPEGE)")
    parser.add_argument("--max-cloud", type=float, default=20,
                        help="Highest total cloud cover in percent for a clear hour (default: 20)")
    args = parser.parse_args()
    if args.hours < 1:
        parser.error("--hours must be at least 1")

    start = parse_time(args.time)
    times, values = fetch_forecast(
        args.latitude, args.longitude, args.elevation, start, args.hours, args.model,
    )
    location = EarthLocation.from_geodetic(
        args.longitude * u.deg, args.latitude * u.deg, args.elevation * u.km,
    )
    sun, moon, illumination = sun_and_moon(times, location)

    print(f"Forecast for {args.latitude:g}, {args.longitude:g} at {args.elevation * 1000:.0f} m "
          f"from the {args.model} model (Open-Meteo)")
    print(f"Times in UTC, * marks hours with the Sun below -12°, at most {args.max_cloud:g}% cloud, and no rain\n")
    for line in table(times, values, sun, moon, illumination, args.max_cloud):
        print(line)
    if not times:
        print("  (no forecast for this period)")


if __name__ == "__main__":
    main()
