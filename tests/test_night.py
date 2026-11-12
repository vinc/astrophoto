from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation

import night

STONEHENGE = EarthLocation.from_geodetic(-1.83 * u.deg, 51.18 * u.deg, 0.1 * u.km)

def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)

class NightTest(unittest.TestCase):
    def test_sun_altitude_on_the_winter_solstice(self):
        sun = night.sun_altitude([utc(2026, 12, 21, 12), utc(2026, 12, 21, 22)], STONEHENGE)
        self.assertGreater(sun[0], 10)
        self.assertLess(sun[1], -18)

    def test_dark_time_is_kept(self):
        self.assertEqual(night.night_start(utc(2026, 12, 21, 20, 49), STONEHENGE), utc(2026, 12, 21, 20, 49))

    def test_daytime_moves_to_the_start_of_the_next_night(self):
        for time in (utc(2026, 12, 21, 12), utc(2026, 12, 22, 8)):
            with self.subTest(time=time):
                start = night.night_start(time, STONEHENGE)
                self.assertTrue(timedelta(0) < start - time < timedelta(hours=24))
                self.assertEqual(start.hour, 17)
                altitude = night.sun_altitude([start - timedelta(minutes=5), start], STONEHENGE)
                self.assertGreaterEqual(altitude[0], night.NIGHT)
                self.assertLess(altitude[1], night.NIGHT)

    def test_time_is_kept_without_a_night(self):
        with patch.object(night, "sun_altitude", side_effect=lambda times, _: np.full(len(times), -5.0)):
            self.assertEqual(night.night_start(utc(2026, 6, 21, 12), STONEHENGE), utc(2026, 6, 21, 12))

if __name__ == "__main__":
    unittest.main()
