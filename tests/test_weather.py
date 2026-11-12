import contextlib
from datetime import datetime, timedelta, timezone
import io
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation

import weather

ARGS = ["weather.py", "--latitude", "51.18", "--longitude", "-1.83", "--elevation", "0.10"]

def response(hourly, status=200):
    mock = Mock(ok=status == 200, status_code=status)
    mock.json.return_value = {"hourly": hourly} if status == 200 else {"error": True, "reason": "Bad model"}
    return mock

def forecast(params, **overrides):
    """Fake Open-Meteo hours between the requested start and end, with values overridden from the start"""
    start = datetime.fromisoformat(params["start_hour"])
    count = int((datetime.fromisoformat(params["end_hour"]) - start).total_seconds() // 3600) + 1
    hourly = {"time": [(start + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M") for hour in range(count)]}
    defaults = {
        "cloud_cover": 5, "cloud_cover_low": 0, "cloud_cover_mid": 5, "cloud_cover_high": 10,
        "precipitation": 0.0, "wind_speed_10m": 2.2, "wind_gusts_10m": 5.6,
        "temperature_2m": 4.0, "dew_point_2m": -3.0,
    }
    for name in weather.VARIABLES:
        given = overrides.get(name, [])[:count]
        hourly[name] = given + [defaults[name]] * (count - len(given))
    return response(hourly)

def cells(output, label):
    line = next(line for line in output.splitlines() if line.startswith(label))
    count = len(next(line for line in output.splitlines() if line.startswith("Time (h)"))[weather.LABEL:].split())
    line = line.ljust(weather.LABEL + weather.CELL * count)
    return [line[weather.LABEL + i * weather.CELL:weather.LABEL + (i + 1) * weather.CELL].strip() for i in range(count)]

def run(arguments, **overrides):
    output = io.StringIO()
    with patch.object(sys, "argv", ARGS + arguments), \
            patch.object(weather.requests, "get", side_effect=lambda *_, params, **__: forecast(params, **overrides)) as get, \
            contextlib.redirect_stdout(output):
        weather.main()
    return output.getvalue(), get.call_args.kwargs["params"]

class RequestTest(unittest.TestCase):
    def test_parameters_use_metres_utc_hours_and_model(self):
        _, params = run(["--time", "2026-12-21T21:49", "--model", "icon_seamless"])
        self.assertEqual(params["elevation"], 100)
        self.assertEqual(params["start_hour"], "2026-12-21T21:00")
        self.assertEqual(params["end_hour"], "2026-12-22T08:00")
        self.assertEqual(params["models"], "icon_seamless")
        self.assertEqual(params["wind_speed_unit"], "ms")
        self.assertEqual(params["hourly"].split(","), weather.VARIABLES)

    def test_time_with_offset_is_converted_to_utc(self):
        self.assertEqual(
            weather.parse_time("2026-12-21T22:30+01:00"),
            datetime(2026, 12, 21, 21, tzinfo=timezone.utc),
        )

    def test_api_error_reason_is_reported(self):
        with patch.object(weather.requests, "get", return_value=response({}, status=400)):
            with self.assertRaisesRegex(SystemExit, "Bad model"):
                weather.fetch_forecast(51.18, -1.83, 0.1, datetime(2026, 12, 21, tzinfo=timezone.utc), 2, "nope")

class TableTest(unittest.TestCase):
    def test_only_dark_clear_dry_hours_are_marked(self):
        output, _ = run(["--time", "2026-12-21T19:00"],
                        cloud_cover=[0, 0, 80, 0], precipitation=[0, 0, 0, 0.5])
        self.assertEqual(cells(output, "Time (h)")[:4], ["19", "20", "21", "22"])
        self.assertEqual(cells(output, "Clear")[:4], ["*", "*", "", ""])
        self.assertEqual(cells(output, "Rain (mm)")[:4], ["0", "0", "0", "1"])
        self.assertEqual(cells(output, "Wind (m/s)")[:4], ["2"] * 4)
        self.assertIn("Mon 21", output)
        self.assertIn("Tue 22", output)

    def test_table_ends_at_the_next_sunrise(self):
        for start, first, count in [("2026-12-21T12:00", "12", 21), ("2026-12-22T02:00", "2", 7)]:
            with self.subTest(start=start):
                output, _ = run(["--time", start])
                hours = cells(output, "Time (h)")
                self.assertEqual((hours[0], hours[-1], len(hours)), (first, "8", count))

    def test_night_length_without_sunrise_keeps_every_hour(self):
        self.assertEqual(weather.night_length(np.array([5, -1, -10, -1, 3, 10])), 4)
        self.assertEqual(weather.night_length(np.array([-10, -5, -2])), 3)
        self.assertEqual(weather.night_length(np.array([10, 20, 10])), 3)

    def test_missing_values_print_as_dashes(self):
        output, _ = run(["--time", "2026-12-21T22:00"], cloud_cover=[None], dew_point_2m=[None])
        self.assertEqual(cells(output, "Cloud (%)")[0], "-")
        self.assertEqual(cells(output, "Dew (°C)")[0], "-")
        self.assertEqual(cells(output, "Clear")[:2], ["", "*"])

    def test_values_are_rounded_but_rain_is_never_rounded_to_none(self):
        self.assertEqual([weather.integer(value) for value in (-0.4, 2.5, 99.6, np.nan)], ["0", "2", "100", "-"])
        self.assertEqual([weather.rain(value) for value in (0, 0.1, 0.4, 0.5, 1.1, 1.5, 2.5, np.nan)],
                         ["0", ">0", ">0", "1", "1", "2", "2", "-"])

class SkyTest(unittest.TestCase):
    def test_winter_solstice_at_stonehenge(self):
        location = EarthLocation.from_geodetic(-1.83 * u.deg, 51.18 * u.deg, 0.1 * u.km)
        times = [datetime(2026, 12, 21, 12, tzinfo=timezone.utc), datetime(2026, 12, 21, 22, tzinfo=timezone.utc)]
        sun, moon, illumination = weather.sun_and_moon(times, location)
        self.assertGreater(sun[0], 10)
        self.assertLess(sun[1], -18)
        self.assertTrue(np.all((illumination >= 0) & (illumination <= 1)))

if __name__ == "__main__":
    unittest.main()
