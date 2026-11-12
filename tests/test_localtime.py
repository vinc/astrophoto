import contextlib
from datetime import datetime, timezone
import os
import time
import unittest
from unittest.mock import patch

from lib.localtime import parse_local_time

@contextlib.contextmanager
def local_time_zone(name):
    try:
        with patch.dict(os.environ, {"TZ": name}):
            time.tzset()
            yield
    finally:
        time.tzset()

def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)

class ParseLocalTimeTest(unittest.TestCase):
    def test_iso_time_without_offset_is_local(self):
        with local_time_zone("Europe/Paris"):
            self.assertEqual(parse_local_time("2026-12-21T22:30"), utc(2026, 12, 21, 21, 30))
        with local_time_zone("Pacific/Auckland"):
            self.assertEqual(parse_local_time("2026-12-21T22:30"), utc(2026, 12, 21, 9, 30))

    def test_iso_time_with_offset_is_kept(self):
        with local_time_zone("Pacific/Auckland"):
            self.assertEqual(parse_local_time("2026-12-21T22:30+01:00"), utc(2026, 12, 21, 21, 30))
            self.assertEqual(parse_local_time("2026-12-21T21:30Z"), utc(2026, 12, 21, 21, 30))

    def test_relative_times_are_local_and_in_the_future(self):
        now = datetime(2026, 12, 21, 18, 45)  # A Monday
        with local_time_zone("Europe/Paris"):
            for text, expected in [
                ("tonight", utc(2026, 12, 21, 19)),
                ("tomorrow night", utc(2026, 12, 22, 19)),
                ("saturday", utc(2026, 12, 26, 11)),
                ("2026-12-24", utc(2026, 12, 24, 11)),
            ]:
                with self.subTest(text=text):
                    self.assertEqual(parse_local_time(text, now), expected)

    def test_bare_date_is_noon_but_explicit_midnight_is_kept(self):
        now = datetime(2026, 12, 21, 18, 45)
        with local_time_zone("UTC"):
            for text, expected in [
                ("2026-12-24", utc(2026, 12, 24, 12)),
                ("2026-12-24T00:00", utc(2026, 12, 24)),
                ("2026-12-24 00:00", utc(2026, 12, 24)),
                ("saturday 00:00", utc(2026, 12, 26)),
                ("midnight", utc(2026, 12, 22)),
                ("tomorrow", utc(2026, 12, 22, 18, 45)),
            ]:
                with self.subTest(text=text):
                    self.assertEqual(parse_local_time(text, now), expected)

    def test_unknown_time_is_reported(self):
        with self.assertRaisesRegex(SystemExit, "Could not understand the time: soonish"):
            parse_local_time("soonish")

if __name__ == "__main__":
    unittest.main()
