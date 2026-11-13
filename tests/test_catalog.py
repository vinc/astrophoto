import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astrophoto.lib import catalog


def point(ra, dec):
    return {"type": "Point", "coordinates": [ra, dec]}


MESSIER = {"features": [
    {"id": "M31", "properties": {"name": "M31", "desig": "NGC 224", "alt": "Andromeda", "dim": "190x60"},
     "geometry": point(10.6751, 41.2667)},
    {"id": "M8", "properties": {"name": "M8", "desig": "NGC 6523", "alt": "Lagoon Nebula", "dim": "90x40"},
     "geometry": point(-89.1, -24.38)},
]}
DSOS = {"features": [
    {"id": "NGC 224", "properties": {"desig": "M 31", "dim": "189.1x61.7"}, "geometry": point(10.6847, 41.26875)},
    {"id": "NGC 7000", "properties": {"desig": "NGC 7000", "dim": "120"}, "geometry": point(-45.3, 44.33)},
    {"id": "IC 1396", "properties": {"desig": "IC 1396", "dim": ""}, "geometry": point(-35.3, 57.5)},
    {"id": "SNR G 024.7+00.6", "properties": {"desig": "SNR G 024.7+00.6", "dim": "30x15"}, "geometry": point(-81.308, -7.097)},
    {"id": "SNR G 024.7-00.6", "properties": {"desig": "SNR G 024.7-00.6", "dim": "15"}, "geometry": point(-80.321, -7.533)},
    {"id": "Sh2 275", "properties": {"desig": "Sh2 275", "dim": "80"}, "geometry": point(97.917, 4.963)},
]}
DSONAMES = {
    "NGC 224": {"name": "Andromeda Galaxy"},
    "NGC 7000": {"name": "North America Nebula"},
    "PGC 2666": {"name": "Andromeda II"},
}
STARNAMES = {
    "91262": {"name": "Vega", "hd": "HD 172167"},
    "98066": {"name": "Terebellum", "hd": ""},
    "98162": {"name": "Terebellum", "hd": ""},
}
STARS = {"features": [
    {"id": 91262, "properties": {"mag": 0.03}, "geometry": point(-80.77, 38.78)},
    {"id": 98066, "properties": {"mag": 4.7}, "geometry": point(-60.0, -26.3)},
    {"id": 98162, "properties": {"mag": 4.5}, "geometry": point(-60.3, -27.7)},
    {"id": 88, "properties": {"mag": 5.7}, "geometry": point(0.27, -48.8)},
]}


def index():
    return catalog.build_index(MESSIER, DSOS, DSONAMES, STARNAMES, STARS)


class ResolveTests(unittest.TestCase):
    def test_spellings_of_one_object_resolve_to_it(self):
        objects = index()
        for query in ["m31", "M31", "M 31", "m-31", "ngc-224", "NGC224", "andromeda", "Andromeda Galaxy"]:
            with self.subTest(query=query):
                obj = catalog.resolve(query, objects)
                self.assertEqual(obj.label, "M31 (Andromeda Galaxy)")
                self.assertEqual(obj.designation, "M31")
                self.assertAlmostEqual(obj.ra, 10.6751)
                self.assertAlmostEqual(obj.size, 190 / 60)

    def test_deep_sky_objects_beyond_messier(self):
        obj = catalog.resolve("north america nebula", index())
        self.assertEqual(obj.label, "NGC 7000 (North America Nebula)")
        self.assertAlmostEqual(obj.ra, 314.7)
        self.assertAlmostEqual(obj.size, 2)
        obj = catalog.resolve("ic1396", index())
        self.assertEqual(obj.label, "IC 1396")
        self.assertIsNone(obj.size)

    def test_signed_coordinates_in_designations_are_kept_apart(self):
        objects = index()
        self.assertAlmostEqual(catalog.resolve("SNR G 024.7+00.6", objects).ra, 278.692)
        self.assertAlmostEqual(catalog.resolve("snr g 024.7-00.6", objects).ra, 279.679)
        with self.assertRaisesRegex(catalog.AmbiguousObject, r"SNR G 024.7\+00.6, SNR G 024.7-00.6"):
            catalog.resolve("SNR G 024.7 00.6", objects)

    def test_hyphens_between_catalog_and_number_are_dropped(self):
        self.assertEqual(catalog.resolve("sh2-275", index()).designation, "Sh2 275")

    def test_stars_by_name_and_catalog_number(self):
        for query in ["vega", "HIP 91262", "hd172167", "HD 172167"]:
            with self.subTest(query=query):
                obj = catalog.resolve(query, index())
                self.assertEqual(obj.label, "Vega")
                self.assertAlmostEqual(obj.ra, 279.23)
        self.assertEqual(catalog.resolve("hip 88", index()).label, "HIP 88")

    def test_unknown_names_fall_back_to_horizons(self):
        for query in ["vesta", "4", "andromeda ii", "andro", "C/2023 A3"]:
            with self.subTest(query=query):
                self.assertIsNone(catalog.resolve(query, index()))

    def test_shared_names_are_listed_rather_than_guessed(self):
        with self.assertRaises(catalog.AmbiguousObject) as raised:
            catalog.resolve("terebellum", index())
        self.assertEqual(str(raised.exception),
                         "terebellum matches several objects: Terebellum (HIP 98066), Terebellum (HIP 98162)")

    def test_catalogs_are_read_from_the_data_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "astrophoto"
            data.mkdir()
            for name, content in zip(catalog.FILES, [MESSIER, DSOS, DSONAMES, STARNAMES, STARS]):
                (data / name).write_text(json.dumps(content))
            with patch.dict(os.environ, {"XDG_DATA_HOME": temporary}), patch.object(catalog, "_index", None):
                self.assertEqual(catalog.resolve("m31").designation, "M31")
                (data / "messier.json").unlink()
                catalog._index = None
                with self.assertRaisesRegex(SystemExit, "astrophoto download"):
                    catalog.resolve("m31")


if __name__ == "__main__":
    unittest.main()
