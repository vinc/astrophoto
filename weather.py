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


def sky_state(sun_altitude):
    return (
        "day" if sun_altitude > -0.8 else
        "civil" if sun_altitude > -6 else
        "naut" if sun_altitude > -12 else
        "astro" if sun_altitude > -18 else "dark"
    )


def dew_risk(temperature, dew_point):
    spread = temperature - dew_point
    if np.isnan(spread) or spread > DEW_RISK_SPREAD:
        return ""
    kind = "frost" if dew_point <= 0 else "dew"
    return kind if spread <= DEW_SPREAD else kind + "?"


def number(value, width, decimals=0):
    return f"{'-':>{width}}" if np.isnan(value) else f"{value:{width}.{decimals}f}"


def rows(times, values, sun, moon, illumination, max_cloud):
    for i, time in enumerate(times):
        cloud = values["cloud_cover"][i]
        rain = values["precipitation"][i]
        clear = sun[i] < -12 and cloud <= max_cloud and not rain > 0
        direction = values["wind_direction_10m"][i]
        compass = "-" if np.isnan(direction) else COMPASS[int((direction + 11.25) // 22.5) % 16]
        moon_text = f"{moon[i]:+3.0f} {illumination[i] * 100:3.0f}%" if moon[i] > 0 else "-"
        temperature, dew_point = values["temperature_2m"][i], values["dew_point_2m"][i]
        yield (
            f"{'*' if clear else ' '} {time:%m-%d %H:%M}  {sky_state(sun[i]):<5}  {moon_text:>8}  "
            f"{number(cloud, 5)}  {number(values['cloud_cover_low'][i], 3)}  "
            f"{number(values['cloud_cover_mid'][i], 3)}  {number(values['cloud_cover_high'][i], 4)}  "
            f"{number(rain, 4, 1)}  {number(values['wind_speed_10m'][i], 4)}  "
            f"{number(values['wind_gusts_10m'][i], 4)}  {compass:<3}  "
            f"{number(temperature, 5, 1)}  {number(dew_point, 5, 1)}  "
            f"{number(temperature - dew_point, 6, 1)}  {dew_risk(temperature, dew_point)}"
        ).rstrip()


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
    print(f"Times in UTC, * marks dark hours with at most {args.max_cloud:g}% cloud and no rain\n")
    print(f"  {'Time':<11}  {'Sky':<5}  {'Moon':>8}  {'Cloud':>5}  {'Low':>3}  {'Mid':>3}  {'High':>4}  "
          f"{'Rain':>4}  {'Wind':>4}  {'Gust':>4}  {'Dir':<3}  {'Temp':>5}  {'Dew':>5}  {'Spread':>6}")
    print(f"  {'':<11}  {'':<5}  {'alt ill':>8}  {'%':>5}  {'%':>3}  {'%':>3}  {'%':>4}  "
          f"{'mm':>4}  {'km/h':>4}  {'km/h':>4}  {'':<3}  {'°C':>5}  {'°C':>5}  {'°C':>6}")
    for row in rows(times, values, sun, moon, illumination, args.max_cloud):
        print(row)
    if not times:
        print("  (no forecast for this period)")


if __name__ == "__main__":
    main()
