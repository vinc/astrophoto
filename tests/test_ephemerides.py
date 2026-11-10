import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astropy import units as u
from astropy.table import Table

import locate

class EphemerisCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "horizons"
        self.cache_patch = patch.object(locate, "EPHEMERIS_CACHE", self.cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.observer = {"lon": 0.0, "lat": 0.0, "elevation": 0.0}

    def positions(self, query, observer, epochs):
        return Table({
            "datetime_jd": epochs,
            "RA": [epoch - 2450000 for epoch in epochs] * u.deg,
            "DEC": [20.0] * len(epochs) * u.deg,
            "targetname": [query] * len(epochs),
        })

    def test_repeat_lookup_reads_disk_without_querying_jpl(self):
        epochs = [2450001.0, 2450002.0]
        with patch.object(locate, "_query_ephemerides", side_effect=self.positions) as lookup:
            first = locate.get_ephemerides("Vesta", self.observer, epochs)
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(len(list(self.cache.glob("*.ecsv"))), 2)
        with patch.object(locate, "_query_ephemerides", side_effect=AssertionError("Network used")):
            second = locate.get_ephemerides("Vesta", self.observer, epochs)
        self.assertEqual(list(second["RA"]), list(first["RA"]))
        self.assertEqual(second["RA"].unit, u.deg)
        self.assertEqual(list(second["targetname"]), ["Vesta", "Vesta"])

    def test_overlapping_lists_only_query_missing_epochs_and_keep_order_and_duplicates(self):
        with patch.object(locate, "_query_ephemerides", side_effect=self.positions) as lookup:
            locate.get_ephemerides("Vesta", self.observer, [2450002.0])
            result = locate.get_ephemerides("Vesta", self.observer,
                                           [2450003.0, 2450002.0, 2450001.0, 2450002.0])
        self.assertEqual(lookup.call_args.args[2], [2450001.0, 2450003.0])
        self.assertEqual(list(result["RA"]), [3.0, 2.0, 1.0, 2.0])

    def test_target_observer_and_exact_time_are_part_of_the_cache_key(self):
        with patch.object(locate, "_query_ephemerides", side_effect=self.positions) as lookup:
            locate.get_ephemerides("Vesta", self.observer, [2450001.0])
            locate.get_ephemerides("Vesta", dict(reversed(list(self.observer.items()))), [2450001])
            self.assertEqual(lookup.call_count, 1)
            locate.get_ephemerides("Ceres", self.observer, [2450001.0])
            for key in self.observer:
                locate.get_ephemerides("Vesta", dict(self.observer, **{key: 1.0}), [2450001.0])
            locate.get_ephemerides("Vesta", dict(self.observer, body=301), [2450001.0])
            locate.get_ephemerides("Vesta", self.observer, [2450001.000001])
        self.assertEqual(lookup.call_count, 7)

    def test_corrupt_cache_is_replaced(self):
        with patch.object(locate, "_query_ephemerides", side_effect=self.positions) as lookup:
            locate.get_ephemerides("Vesta", self.observer, [2450001.0])
            path = next(self.cache.glob("*.ecsv"))
            path.write_text("incomplete cache")
            result = locate.get_ephemerides("Vesta", self.observer, [2450001.0])
        self.assertEqual(lookup.call_count, 2)
        self.assertEqual(list(result["RA"]), [1.0])
        self.assertEqual(Table.read(path, format="ascii.ecsv")["RA"][0], 1.0)

    def test_failed_or_incomplete_queries_are_not_cached(self):
        for response in [ValueError("Unknown target"), Table({"RA": []})]:
            with self.subTest(response=response), \
                    patch.object(locate, "_query_ephemerides", side_effect=[response]), \
                    self.assertRaises(ValueError):
                locate.get_ephemerides("unknown", self.observer, [2450001.0])
            self.assertFalse(self.cache.exists())

    def test_unwritable_cache_does_not_discard_a_successful_lookup(self):
        self.cache.write_text("a file blocks the directory")
        with patch.object(locate, "_query_ephemerides", side_effect=self.positions), \
                self.assertWarnsRegex(RuntimeWarning, "Could not cache"):
            result = locate.get_ephemerides("Vesta", self.observer, [2450001.0])
        self.assertEqual(list(result["RA"]), [1.0])

    def test_ambiguous_name_resolution_is_cached_and_does_not_mutate_observer(self):
        ambiguity = ValueError("Ambiguous target name\n  4  Vesta  A807 FA\n  99  Vesta Other  Other")
        def horizons(**kwargs):
            kwargs["location"]["lat"] = 90
            return client
        with patch.object(locate, "Horizons", side_effect=horizons) as constructor, \
                contextlib.redirect_stdout(io.StringIO()):
            client = constructor.return_value
            client.ephemerides.side_effect = [ambiguity, self.positions("Vesta", self.observer, [2450001.0])]
            locate.get_ephemerides("Vesta", self.observer, [2450001.0])
            locate.get_ephemerides("Vesta", self.observer, [2450001.0])
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual(constructor.call_args.kwargs["id"], "4")
        self.assertEqual(self.observer["lat"], 0.0)

if __name__ == "__main__":
    unittest.main()
