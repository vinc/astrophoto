import contextlib
from datetime import datetime, timezone
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

def forecast(times, **overrides):
    hourly = {"time": times}
    defaults = {
        "cloud_cover": 5, "cloud_cover_low": 0, "cloud_cover_mid": 5, "cloud_cover_high": 10,
        "precipitation": 0.0, "wind_speed_10m": 8, "wind_gusts_10m": 20,
        "temperature_2m": 4.0, "dew_point_2m": -3.0,
    }
    for name in weather.VARIABLES:
        value = overrides.get(name, defaults[name])
        hourly[name] = value if isinstance(value, list) else [value] * len(times)
    return hourly

def cells(output, label, count):
    line = next(line for line in output.splitlines() if line.startswith(label))
    line = line.ljust(weather.LABEL + weather.CELL * count)
    return [line[weather.LABEL + i * weather.CELL:weather.LABEL + (i + 1) * weather.CELL].strip() for i in range(count)]

def run(arguments, hourly):
    output = io.StringIO()
    with patch.object(sys, "argv", ARGS + arguments), \
            patch.object(weather.requests, "get", return_value=response(hourly)) as get, \
            contextlib.redirect_stdout(output):
        weather.main()
    return output.getvalue(), get

class RequestTest(unittest.TestCase):
    def test_parameters_use_metres_utc_hours_and_model(self):
        times = ["2026-12-21T21:00", "2026-12-21T22:00"]
        _, get = run(["--time", "2026-12-21T21:49", "--hours", "2", "--model", "icon_seamless"], forecast(times))
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["elevation"], 100)
        self.assertEqual(params["start_hour"], "2026-12-21T21:00")
        self.assertEqual(params["end_hour"], "2026-12-21T22:00")
        self.assertEqual(params["models"], "icon_seamless")
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
        times = ["2026-12-21T12:00", "2026-12-21T22:00", "2026-12-21T23:00", "2026-12-22T00:00"]
        output, _ = run(["--time", "2026-12-21T12:00", "--hours", "4"], forecast(
            times, cloud_cover=[0, 0, 80, 0], precipitation=[0, 0, 0, 0.4],
        ))
        self.assertEqual(cells(output, "Time (h)", 4), ["12", "22", "23", "0"])
        self.assertEqual(cells(output, "Clear", 4), ["", "*", "", ""])
        self.assertEqual(cells(output, "Rain (mm)", 4), ["0", "0", "0", "0.4"])
        self.assertIn("Mon 21", output)
        self.assertIn("Tue 22", output)

    def test_long_forecasts_are_split_into_blocks(self):
        times = [f"2026-12-{21 + hour // 24}T{hour % 24:02d}:00" for hour in range(30)]
        output, _ = run(["--time", "2026-12-21T00:00", "--hours", "30"], forecast(times))
        rows = [line for line in output.splitlines() if line.startswith("Time (h)")]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1].split()[2:], [str(hour) for hour in range(6)])

    def test_missing_values_print_as_dashes(self):
        output, _ = run(["--time", "2026-12-21T22:00", "--hours", "1"],
                        forecast(["2026-12-21T22:00"], cloud_cover=[None], dew_point_2m=[None]))
        self.assertEqual(cells(output, "Cloud (%)", 1), ["-"])
        self.assertEqual(cells(output, "Dew (°C)", 1), ["-"])
        self.assertEqual(cells(output, "Clear", 1), [""])

    def test_values_fit_in_a_cell(self):
        self.assertEqual([weather.decimal(value) for value in (0.04, 0.4, 9.94, 12.6, -1.2)], ["0", "0.4", "9.9", "13", "-1"])
        self.assertEqual([weather.integer(value) for value in (-0.4, 99.6, np.nan)], ["0", "100", "-"])

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
