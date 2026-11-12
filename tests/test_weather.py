import contextlib
from datetime import datetime, timedelta, timezone
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import weather
from tests.test_localtime import local_time_zone

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

def run(arguments, zone="UTC", **overrides):
    output = io.StringIO()
    with tempfile.TemporaryDirectory() as cache, patch.object(weather, "CACHE", Path(cache)), \
            local_time_zone(zone), patch.object(sys, "argv", ARGS + arguments), \
            patch.object(weather.requests, "get", side_effect=lambda *_, params, **__: forecast(params, **overrides)) as get, \
            contextlib.redirect_stdout(output):
        weather.main()
    return output.getvalue(), get.call_args.kwargs["params"]

class RequestTest(unittest.TestCase):
    def test_parameters_use_metres_utc_hours_and_model(self):
        _, params = run(["--time", "2026-12-21T21:49", "--model", "icon_seamless"])
        self.assertEqual(params["elevation"], 100)
        self.assertEqual(params["start_hour"], "2026-12-21T21:00")
        self.assertEqual(params["end_hour"], "2026-12-22T06:00")
        self.assertEqual(params["models"], "icon_seamless")
        self.assertEqual(params["wind_speed_unit"], "ms")
        self.assertEqual(params["hourly"].split(","), weather.VARIABLES)

    def test_time_with_offset_is_converted_to_utc(self):
        self.assertEqual(
            weather.parse_time("2026-12-21T22:30+01:00"),
            datetime(2026, 12, 21, 21, tzinfo=timezone.utc),
        )

    def test_start_is_in_utc_to_the_hour(self):
        now = datetime(2026, 12, 21, 18, 45)  # A Monday
        with local_time_zone("Europe/Paris"):
            for text, expected in [
                ("tomorrow", datetime(2026, 12, 22, 17)),
                ("saturday", datetime(2026, 12, 26, 11)),
                ("2026-12-24", datetime(2026, 12, 24, 11)),
            ]:
                with self.subTest(text=text):
                    self.assertEqual(weather.parse_time(text, now), expected.replace(tzinfo=timezone.utc))

    def test_network_error_is_reported(self):
        error = weather.requests.ConnectionError("offline")
        with patch.object(weather.requests, "get", side_effect=error):
            with self.assertRaisesRegex(SystemExit, "Open-Meteo request failed: offline"):
                weather.download({})

    def test_api_error_reason_is_reported(self):
        with tempfile.TemporaryDirectory() as cache, patch.object(weather, "CACHE", Path(cache)), \
                patch.object(weather.requests, "get", return_value=response({}, status=400)):
            with self.assertRaisesRegex(SystemExit, "Bad model"):
                weather.fetch_forecast(51.18, -1.83, 0.1, datetime(2026, 12, 21, tzinfo=timezone.utc), 2, "nope")

class CacheTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "weather"
        cache_patch = patch.object(weather, "CACHE", self.cache)
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.params = {"latitude": 51.18, "start_hour": "2026-12-21T18:00"}

    def download(self, params, update=False, status=200):
        with patch.object(weather.requests, "get", return_value=response({"time": []}, status=status)) as get:
            data = weather.cached_download(params, update)
        return data, get.call_count

    def age(self, minutes):
        for path in self.cache.glob("*.json"):
            when = datetime.now().timestamp() - minutes * 60
            os.utime(path, (when, when))

    def test_repeat_request_reads_the_cache(self):
        self.assertEqual(self.download(self.params), ({"hourly": {"time": []}}, 1))
        self.assertEqual(self.download(self.params), ({"hourly": {"time": []}}, 0))

    def test_other_parameters_are_downloaded(self):
        self.download(self.params)
        self.assertEqual(self.download(dict(self.params, latitude=48.0))[1], 1)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 2)

    def test_expired_or_forced_requests_are_downloaded_again(self):
        self.download(self.params)
        self.assertEqual(self.download(self.params, update=True)[1], 1)
        self.age(16)
        self.assertEqual(self.download(self.params)[1], 1)
        self.assertEqual(self.download(self.params)[1], 0)

    def test_errors_are_not_cached(self):
        with self.assertRaises(SystemExit):
            self.download(self.params, status=500)
        self.assertEqual(list(self.cache.glob("*.json")) if self.cache.exists() else [], [])

    def test_expired_files_are_removed(self):
        self.download(self.params)
        old = next(self.cache.glob("*.json"))
        self.age(60)
        self.download(dict(self.params, latitude=48.0))
        self.assertFalse(old.exists())

    def test_corrupt_cache_is_downloaded_again(self):
        self.download(self.params)
        next(self.cache.glob("*.json")).write_text("{")
        self.assertEqual(self.download(self.params), ({"hourly": {"time": []}}, 1))

class TableTest(unittest.TestCase):
    def test_rows_and_dates(self):
        output, _ = run(["--time", "2026-12-21T19:00"], cloud_cover=[0, 80], precipitation=[0, 0.5])
        self.assertEqual(cells(output, "Time (h)")[:2], ["19", "20"])
        self.assertEqual(cells(output, "Cloud (%)")[:2], ["0", "80"])
        self.assertEqual(cells(output, "Rain (mm)")[:2], ["0", "1"])
        self.assertEqual(cells(output, "Wind (m/s)")[:2], ["2", "2"])
        lines = output.splitlines()
        separator = "-" * len(lines[1])
        self.assertEqual([line if line == separator else line.split()[0] for line in lines],
                         ["Date", "Time", separator, "Cloud", "High", "Mid", "Low", separator, "Wind", "Gust",
                          separator, "Temp", "Dew", separator, "Rain"])
        self.assertIn("Mon 21", output)
        self.assertIn("Tue 22", output)

    def test_only_hours_of_the_next_night_are_shown(self):
        for start, hours in [
            ("2026-12-21T12:00", ["18", "6"]),
            ("2026-12-22T02:00", ["2", "6"]),
            ("2026-12-22T07:00", ["18", "6"]),
        ]:
            with self.subTest(start=start):
                output, _ = run(["--time", start])
                time = cells(output, "Time (h)")
                self.assertEqual([time[0], time[-1]], hours)

    def test_times_are_shown_in_the_local_time_zone(self):
        output, params = run(["--time", "2026-12-21T12:00"], zone="Europe/Paris")
        self.assertEqual(params["start_hour"], "2026-12-21T18:00")
        hours = cells(output, "Time (h)")
        self.assertEqual([hours[0], hours[-1]], ["19", "7"])

    def test_night_beyond_the_forecast_is_reported(self):
        with patch.object(weather, "fetch_forecast", return_value=([], {})):
            with self.assertRaisesRegex(SystemExit, "No forecast for this night"):
                run(["--time", "2026-12-21T19:00"])

    def test_no_night_is_reported(self):
        with patch.object(weather, "sun_altitude", return_value=np.full(weather.MAX_HOURS, -5.0)):
            with self.assertRaisesRegex(SystemExit, "stays above"):
                run(["--time", "2026-06-21T12:00"])

    def test_ratings(self):
        values = {name: np.full(7, value) for name, value in [
            ("cloud_cover", 10.0), ("precipitation", 0.0), ("wind_gusts_10m", 3.0),
            ("temperature_2m", 5.0), ("dew_point_2m", 0.0),
        ]}
        values["cloud_cover"][1:3] = [60, 30]
        values["precipitation"][3] = 0.1
        values["wind_gusts_10m"][4] = 5.6
        values["temperature_2m"][5], values["dew_point_2m"][5] = 4.6, 2.6
        values["cloud_cover"][6] = np.nan
        self.assertEqual(weather.ratings(values), ["green", "red", "yellow", "yellow", "yellow", "yellow", None])

    def test_only_the_time_row_is_colored(self):
        times = [datetime(2026, 12, 21, hour, tzinfo=timezone.utc) for hour in (20, 21)]
        values = {name: np.array([5.0, 90.0]) if name == "cloud_cover" else np.zeros(2) for name in weather.VARIABLES}
        values["dew_point_2m"] = np.full(2, -5.0)
        lines = weather.table(times, values, color=True)
        self.assertEqual(lines[1], "Time (h)    \033[32m 20\033[0m \033[31m 21\033[0m")
        self.assertNotIn("\033", "".join(lines[2:]))
        self.assertNotIn("\033", "".join(weather.table(times, values)))

    def test_no_color_when_piped_or_disabled(self):
        with patch.dict(os.environ), patch.object(sys.stdout, "isatty", return_value=True):
            os.environ.pop("NO_COLOR", None)
            self.assertTrue(weather.use_color())
            with patch.dict(os.environ, {"NO_COLOR": "1"}):
                self.assertFalse(weather.use_color())
        with patch.object(sys.stdout, "isatty", return_value=False):
            self.assertFalse(weather.use_color())

    def test_missing_values_print_as_dashes(self):
        output, _ = run(["--time", "2026-12-21T22:00"], cloud_cover=[None], dew_point_2m=[None])
        self.assertEqual(cells(output, "Cloud (%)")[0], "-")
        self.assertEqual(cells(output, "Dew (°C)")[0], "-")

    def test_values_are_rounded_but_rain_is_never_rounded_to_none(self):
        self.assertEqual([weather.integer(value) for value in (-0.4, 2.5, 99.6, np.nan)], ["0", "2", "100", "-"])
        self.assertEqual([weather.rain(value) for value in (0, 0.1, 0.4, 0.5, 1.1, 1.5, 2.5, np.nan)],
                         ["0", ">0", ">0", "1", "1", "2", "2", "-"])

if __name__ == "__main__":
    unittest.main()
