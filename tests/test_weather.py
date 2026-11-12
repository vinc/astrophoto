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
        "precipitation": 0.0, "wind_speed_10m": 8, "wind_gusts_10m": 20, "wind_direction_10m": 250,
        "temperature_2m": 4.0, "dew_point_2m": -3.0,
    }
    for name in weather.VARIABLES:
        value = overrides.get(name, defaults[name])
        hourly[name] = value if isinstance(value, list) else [value] * len(times)
    return hourly

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
        rows = output.splitlines()[-4:]
        self.assertEqual([row[0] for row in rows], [" ", "*", " ", " "])
        self.assertIn("day", rows[0])
        self.assertIn("dark", rows[1])
        self.assertIn("WSW", rows[1])

    def test_missing_values_print_as_dashes(self):
        output, _ = run(["--time", "2026-12-21T22:00", "--hours", "1"],
                        forecast(["2026-12-21T22:00"], cloud_cover=[None], dew_point_2m=[None]))
        row = output.splitlines()[-1]
        self.assertEqual(row[0], " ")
        self.assertIn(" - ", row)

    def test_dew_and_frost_risk(self):
        self.assertEqual(weather.dew_risk(6.0, 1.0), "")
        self.assertEqual(weather.dew_risk(6.0, 2.5), "dew?")
        self.assertEqual(weather.dew_risk(6.0, 4.5), "dew")
        self.assertEqual(weather.dew_risk(1.0, -0.5), "frost")
        self.assertEqual(weather.dew_risk(np.nan, 1.0), "")

class SkyTest(unittest.TestCase):
    def test_winter_solstice_at_stonehenge(self):
        location = EarthLocation.from_geodetic(-1.83 * u.deg, 51.18 * u.deg, 0.1 * u.km)
        times = [datetime(2026, 12, 21, 12, tzinfo=timezone.utc), datetime(2026, 12, 21, 22, tzinfo=timezone.utc)]
        sun, moon, illumination = weather.sun_and_moon(times, location)
        self.assertGreater(sun[0], 10)
        self.assertLess(sun[1], -18)
        self.assertEqual([weather.sky_state(altitude) for altitude in sun], ["day", "dark"])
        self.assertTrue(np.all((illumination >= 0) & (illumination <= 1)))

if __name__ == "__main__":
    unittest.main()
