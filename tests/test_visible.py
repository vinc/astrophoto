import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import requests
from astropy.time import Time

import visible
from tests.test_localtime import local_time_zone

ARGS = ["visible.py", "--latitude", "51.18", "--longitude", "-1.83", "--elevation", "0.10"]

def cache_data(downloaded):
    return {"downloaded": downloaded, "asteroids": [], "comets": []}

class OrbitTest(unittest.TestCase):
    def test_circular_orbit_moves_a_quarter_turn(self):
        one = np.array([1.0])
        zero = np.array([0.0])
        period = 2 * np.pi / visible.K
        position = visible.heliocentric(one, zero, zero, zero, zero, zero, period / 4)
        ecliptic = np.array([
            position[0, 0],
            position[1, 0] * np.cos(visible.OBLIQUITY) + position[2, 0] * np.sin(visible.OBLIQUITY),
            -position[1, 0] * np.sin(visible.OBLIQUITY) + position[2, 0] * np.cos(visible.OBLIQUITY),
        ])
        np.testing.assert_allclose(ecliptic, [0, 1, 0], atol=1e-9)

    def test_conics_agree_near_parabolic(self):
        q = np.full(3, 1.2)
        e = np.array([1 - 1e-6, 1.0, 1 + 1e-6])
        angles = np.radians(np.full(3, 30.0))
        for dt in (-120.0, 15.0, 400.0):
            position = visible.heliocentric(q, e, np.zeros(3), angles, angles, angles, dt)
            np.testing.assert_allclose(position[:, 0], position[:, 1], atol=1e-4)
            np.testing.assert_allclose(position[:, 2], position[:, 1], atol=1e-4)

    def test_perihelion_distance(self):
        q = np.array([0.5, 2.0, 1.0, 3.0])
        e = np.array([0.2, 0.9, 1.0, 2.5])
        zero = np.zeros(4)
        position = visible.heliocentric(q, e, zero, zero, zero, zero, 0.0)
        np.testing.assert_allclose(np.linalg.norm(position, axis=0), q)

class MagnitudeTest(unittest.TestCase):
    def test_hg_magnitude_at_unit_distances_is_h(self):
        self.assertAlmostEqual(float(visible.hg_magnitude(3.2, 0.32, 1, 1, 0)), 3.2)

    def test_hg_magnitude_fades_with_phase(self):
        self.assertGreater(visible.hg_magnitude(5, 0.15, 2, 1, 20), visible.hg_magnitude(5, 0.15, 2, 1, 5))

    def test_jupiter_at_opposition(self):
        magnitude = visible.planet_magnitude("jupiter", 5.2, 4.2, 0, None, None)
        self.assertAlmostEqual(magnitude, -9.395 + 5 * np.log10(5.2 * 4.2))

    def test_saturn_rings_brighten_when_tilted(self):
        edge_on = visible.planet_magnitude("saturn", 9.5, 8.5, 1, np.array([1.0, 0, 0]), None)
        pole = np.radians(visible.POLES["saturn"])
        toward_pole = np.array([
            np.cos(pole[1]) * np.cos(pole[0]), np.cos(pole[1]) * np.sin(pole[0]), np.sin(pole[1]),
        ])
        tilted = visible.planet_magnitude("saturn", 9.5, 8.5, 1, toward_pole, None)
        self.assertLess(tilted, edge_on)

class CacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name) / "minor-bodies.json"
        patcher = patch.object(visible, "CACHE", self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temporary.cleanup)

    def test_fresh_cache_is_used_without_downloading(self):
        self.cache.write_text(json.dumps(cache_data(Time.now().isot)))
        with patch.object(visible, "download_minor_bodies") as download:
            data = visible.load_minor_bodies(update=False)
        download.assert_not_called()
        self.assertEqual(data["asteroids"], [])

    def test_stale_cache_is_used_when_offline(self):
        old = (Time.now() - 40).isot
        self.cache.write_text(json.dumps(cache_data(old)))
        error = requests.ConnectionError("offline")
        with patch.object(visible, "download_minor_bodies", side_effect=error), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            data = visible.load_minor_bodies(update=False)
        self.assertEqual(data["downloaded"], old)
        self.assertIn("Could not update", stderr.getvalue())

    def test_missing_cache_and_offline_exits(self):
        error = requests.ConnectionError("offline")
        with patch.object(visible, "download_minor_bodies", side_effect=error), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            visible.load_minor_bodies(update=False)

    def test_download_is_saved(self):
        fresh = cache_data(Time.now().isot)
        with patch.object(visible, "download_minor_bodies", return_value=fresh), \
                contextlib.redirect_stderr(io.StringIO()):
            visible.load_minor_bodies(update=True)
        self.assertEqual(json.loads(self.cache.read_text()), fresh)

class MainTest(unittest.TestCase):
    def test_lists_bright_planets_sorted_by_magnitude(self):
        data = cache_data(Time.now().isot)
        argv = ARGS + ["--time", "2026-11-07T21:11", "--limit", "8"]
        with patch.object(visible, "load_minor_bodies", return_value=data), \
                patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            visible.main()
        rows = [line for line in stdout.getvalue().splitlines() if line.strip().endswith(("Saturn", "Uranus", "Neptune"))]
        self.assertEqual([row.split()[-1] for row in rows], ["Saturn", "Uranus", "Neptune"])
        magnitudes = [float(row.split()[0]) for row in rows]
        self.assertEqual(magnitudes, sorted(magnitudes))

    def test_time_is_local(self):
        data = cache_data(Time.now().isot)
        argv = ARGS + ["--time", "2026-11-07T21:11"]
        with local_time_zone("Europe/Paris"), \
                patch.object(visible, "load_minor_bodies", return_value=data), \
                patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            visible.main()
        self.assertTrue(stdout.getvalue().startswith("2026-11-07 21:11 CET, Sun at"))

    def test_daytime_moves_to_the_start_of_the_night(self):
        data = cache_data(Time.now().isot)
        argv = ARGS + ["--time", "2026-12-21T14:00"]
        with local_time_zone("Europe/Paris"), \
                patch.object(visible, "load_minor_bodies", return_value=data), \
                patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            visible.main()
        self.assertTrue(stdout.getvalue().startswith("2026-12-21 18:25 CET, Sun at -12 degrees"))

if __name__ == "__main__":
    unittest.main()
