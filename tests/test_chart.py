import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

from astrophoto import chart

def make_header(ra=56.75, dec=24.1, angle=15, scale=0.00123, size=(3504, 2336)):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(size[0] + 1) / 2, (size[1] + 1) / 2]
    a = np.radians(angle)
    wcs.wcs.cd = scale * np.array([[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]])
    header = wcs.to_header(relax=True)
    header["NAXIS"] = 2
    header["NAXIS1"], header["NAXIS2"] = size
    return header

def write_catalog(directory, stars):
    def collection(features):
        return {"type": "FeatureCollection", "features": features}
    # d3-celestial stores right ascension as a longitude in [-180, 180]
    def lon(ra):
        return (ra + 180) % 360 - 180
    (directory / "stars.6.json").write_text(json.dumps(collection([
        {"type": "Feature", "id": i, "properties": {"mag": mag},
         "geometry": {"type": "Point", "coordinates": [lon(ra), dec]}}
        for i, (ra, dec, mag) in enumerate(stars)
    ])))
    (directory / "constellations.lines.json").write_text(json.dumps(collection([
        {"type": "Feature", "id": "Tau", "geometry": {"type": "MultiLineString",
         "coordinates": [[[lon(60), 20], [lon(65), 18]], [[lon(240), -20], [lon(245), -18]]]}},
    ])))
    (directory / "constellations.json").write_text(json.dumps(collection([
        {"type": "Feature", "id": "Tau", "properties": {"name": "Taurus"},
         "geometry": {"type": "Point", "coordinates": [lon(65), 16]}},
        {"type": "Feature", "id": "Sco", "properties": {"name": "Scorpius"},
         "geometry": {"type": "Point", "coordinates": [lon(250), -30]}},
    ])))

class ChartTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        write_catalog(self.path, [(56.75, 24.1, 2.9), (60, 20, 5.5), (240, -20, 1.0), (57, 25, 6.5)])
        catalog = patch.object(chart, "CATALOG", self.path)
        catalog.start()
        self.addCleanup(catalog.stop)

    def render(self, outlines, **kwargs):
        figures = []
        with patch.object(chart.plt, "close", side_effect=figures.append):
            chart.draw_chart(outlines, self.path / "chart.png", **kwargs)
        self.addCleanup(lambda: [matplotlib.pyplot.close(figure) for figure in figures])
        return figures[0].axes[0]

    def test_footprint_follows_rotated_image_edges(self):
        header = make_header()
        ra, dec = chart.footprint(header, samples=8)
        self.assertEqual(len(ra), 32)
        corners = WCS(header).pixel_to_world_values([-0.5, 3503.5, 3503.5, -0.5], [-0.5, -0.5, 2335.5, 2335.5])
        np.testing.assert_allclose(ra[[0, 8, 16, 24]], corners[0])
        np.testing.assert_allclose(dec[[0, 8, 16, 24]], corners[1])

    def test_footprint_requires_a_plate_solution(self):
        with self.assertRaises(ValueError):
            chart.footprint(fits.Header({"NAXIS": 2, "NAXIS1": 10, "NAXIS2": 10}))

    def test_solved_headers_are_read_without_warnings(self):
        header = make_header()
        header["CTYPE1"], header["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
        header["A_ORDER"] = header["B_ORDER"] = 2
        header["A_2_0"], header["B_0_2"], header["A_1_1"] = 2e-6, -2e-6, 1e-6
        header["DATE-OBS"], header["DATE-AVG"] = "2026-11-06T19:03:00", "2026-11-06T19:05:00"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            outline = chart.footprint(header)
            self.assertTrue(chart.contains(header, outline, 56.75, 24.1))
            for ra, dec in [(60, 24.1), (150, 12), (236.75, -24.1)]:
                self.assertFalse(chart.contains(header, outline, ra, dec))

    def test_center_survives_right_ascension_wrap(self):
        outline = chart.footprint(make_header(ra=359.5, dec=10, angle=0))
        self.assertTrue(np.any(outline[0] > 359) and np.any(outline[0] < 1))
        ra, dec = chart.chart_center([outline])
        self.assertAlmostEqual(abs((ra + 180) % 360 - 180), 0.5, places=3)
        self.assertAlmostEqual(dec, 10, places=3)

    def test_default_fov_keeps_context_around_a_single_image(self):
        outlines = [chart.footprint(make_header())]
        self.assertEqual(chart.default_fov(outlines, chart.chart_center(outlines)), 30)

    def test_every_image_of_a_night_lands_on_the_chart(self):
        spots = [(300, 30), (16.4, -4.6), (83.8, -1), (150, 12)]
        for count in range(2, len(spots) + 1):
            outlines = [chart.footprint(make_header(ra=ra, dec=dec, scale=0.0049)) for ra, dec in spots[:count]]
            center = chart.chart_center(outlines)
            wcs = chart.chart_wcs(center, chart.default_fov(outlines, center))
            for ra, dec in outlines:
                x, y = wcs.world_to_pixel_values(ra, dec)
                self.assertTrue(np.all(chart.inside(np.asarray(x), np.asarray(y))), spots[:count])
                self.assertTrue(np.all(chart.separation(ra, dec, center) < chart.chart_reach(wcs)))

    def test_chart_draws_outline_on_canvas_with_nearby_stars_only(self):
        outline = chart.footprint(make_header())
        ax = self.render([outline], title="Taurus")
        x, y = ax.lines[-1].get_data()
        self.assertTrue(np.all(chart.inside(np.asarray(x), np.asarray(y))))
        stars = ax.collections[0].get_offsets()
        self.assertEqual(len(stars), 2)
        self.assertIsNone(ax.get_legend())
        self.assertEqual([text.get_text() for text in ax.texts], ["TAURUS"])
        self.assertEqual(ax.get_title(), "Taurus")
        self.assertTrue((self.path / "chart.png").exists())

    def test_brighter_stars_are_bigger_and_drawn_last(self):
        ax = self.render([chart.footprint(make_header())])
        sizes = ax.collections[0].get_sizes()
        self.assertGreater(sizes[-1], sizes[0])

    def test_multiple_images_get_a_legend_and_target_track(self):
        outlines = [chart.footprint(make_header(ra=56.75 + i, dec=24.1)) for i in range(3)]
        labels = [f"night {i}" for i in range(3)]
        track = ("4 Vesta", [57.0, 58.0, 59.0], [24.0, 24.0, 24.0])
        ax = self.render(outlines, labels=labels, track=track)
        self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], labels)
        colours = {line.get_color() for line in ax.lines[-4:-1]}
        self.assertEqual(colours, set(chart.SERIES[:3]))
        self.assertIn("4 Vesta", [text.get_text() for text in ax.texts])

    def test_more_images_than_colours_share_one(self):
        outlines = [chart.footprint(make_header(ra=50 + i, dec=24.1)) for i in range(len(chart.SERIES) + 1)]
        ax = self.render(outlines, labels=[str(i) for i in range(len(outlines))])
        self.assertEqual({line.get_color() for line in ax.lines[-len(outlines):]}, {chart.SERIES[0]})
        self.assertIsNone(ax.get_legend())

    def test_wide_charts_leave_out_constellation_names(self):
        ax = self.render([chart.footprint(make_header())], fov=chart.WIDE + 1)
        self.assertEqual(len(ax.texts), 0)

    def test_constellation_segment_crossing_the_chart_is_kept(self):
        (self.path / "constellations.lines.json").write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "id": "Tau", "geometry": {"type": "MultiLineString",
             "coordinates": [[[16.75, 24.1], [96.75, 24.1]]]}},
        ]}))
        outline = chart.footprint(make_header())
        center = chart.chart_center([outline])
        reach = chart.chart_reach(chart.chart_wcs(center, chart.default_fov([outline], center)))
        self.assertTrue(np.all(chart.separation(np.array([16.75, 96.75]), np.array([24.1, 24.1]), center) > reach))
        ax = self.render([outline])
        figures = [line for line in ax.lines if line.get_color() == chart.FIGURES]
        self.assertEqual(len(figures), 1)
        x, y = figures[0].get_data()
        self.assertTrue(np.any(chart.inside(np.asarray(x), np.asarray(y))))

    def test_opposite_images_share_a_chart(self):
        outlines = [chart.footprint(make_header(ra=ra, dec=0, scale=0.0049)) for ra in (0, 180)]
        center = chart.chart_center(outlines)
        for ra, dec in outlines:
            self.assertLess(chart.separation(ra, dec, center).max(), 100)
        fov = chart.default_fov(outlines, center)
        self.assertTrue(np.isfinite(fov))
        wcs = chart.chart_wcs(center, fov)
        for ra, dec in outlines:
            x, y = wcs.world_to_pixel_values(ra, dec)
            self.assertTrue(np.all(chart.inside(np.asarray(x), np.asarray(y))))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.render(outlines, labels=["a", "b"])

    def test_images_all_over_the_sky_fail_clearly(self):
        spots = [(ra, dec) for ra in range(0, 360, 18) for dec in (-60, -30, 0, 30, 60)]
        with self.assertRaises(SystemExit) as raised:
            chart.chart_center([chart.footprint(make_header(ra=ra, dec=dec)) for ra, dec in spots])
        self.assertIn("chart fewer at a time", str(raised.exception))

    def test_missing_catalog_points_to_download_script(self):
        (self.path / "stars.6.json").unlink()
        with self.assertRaises(SystemExit) as raised:
            self.render([chart.footprint(make_header())])
        self.assertIn("astrophoto download", str(raised.exception))

class DescribeTimesTests(unittest.TestCase):
    paris = chart.ZoneInfo("Europe/Paris")

    def test_times_are_shown_on_the_camera_clock(self):
        self.assertEqual(chart.describe_times([Time("2026-11-06T19:12:00")], self.paris), "2026-11-06 20:12 CET")

    def test_one_night_shows_its_time_span(self):
        times = [Time("2026-11-06T19:12:00"), Time("2026-11-06T22:40:00")]
        self.assertEqual(chart.describe_times(times, self.paris), "2026-11-06 20:12 to 23:40 CET")

    def test_several_nights_show_their_date_span(self):
        times = [Time("2026-11-06T19:12:00"), Time("2026-11-08T23:30:00")]
        self.assertEqual(chart.describe_times(times, self.paris), "2026-11-06 to 2026-11-09")

class DefaultOutputTests(unittest.TestCase):
    def test_single_image_chart_sits_next_to_it(self):
        path = Path("20261106-2010-light") / "20261106-2010-light-stack.fit"
        self.assertEqual(chart.default_output([path]), path.with_name("20261106-2010-light-stack-chart.png"))
        self.assertEqual(chart.default_output([path], "4 Vesta").name, "20261106-2010-light-stack-chart-4-vesta.png")

    def test_several_images_get_the_next_number_for_their_name(self):
        paths = [Path("20261106-2003-light/20261106-2003-light-rgb-stack.fit"), Path("20261108-2110-light/20261108-2110-light-stack.fit")]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertEqual(chart.default_output(paths, directory=directory), directory / "light-rgb-stack-chart-0001.png")
            for name in ["light-rgb-stack-chart-0001.png", "light-rgb-stack-chart-0002.png",
                         "light-rgb-stack-chart-vesta-0005.png", "light-stack-chart-0009.png"]:
                (directory / name).touch()
            self.assertEqual(chart.default_output(paths, directory=directory), directory / "light-rgb-stack-chart-0003.png")
            self.assertEqual(chart.default_output(paths, "Vesta", directory), directory / "light-rgb-stack-chart-vesta-0006.png")
            self.assertEqual(chart.default_output(paths, "pallas", directory), directory / "light-rgb-stack-chart-pallas-0001.png")

class ChartCommandTests(unittest.TestCase):
    def run_main(self, argv, **patches):
        with patch.object(sys, "argv", ["chart.py", *argv]), \
                contextlib.redirect_stdout(io.StringIO()) as stdout, \
                contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(patch.object(chart, name, **value))
            chart.main()
        return stdout.getvalue()

    def test_query_needs_observer(self):
        with patch.object(sys, "argv", ["chart.py", "image.fit", "--query", "vesta"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            chart.main()

    def test_images_are_charted_in_time_order_with_target_positions(self):
        times = ["2026-11-12T21:30:00", "2026-11-11T21:30:00"]
        epochs = [Time(time).jd for time in times]
        calls = {}

        def ephemerides(query, observer, requested):
            calls["epochs"] = requested
            return Table({"RA": [56.0, 57.0], "DEC": [24.0, 24.0], "targetname": ["4 Vesta (A807 FA)"] * 2})

        def draw(outlines, output, **kwargs):
            calls.update(kwargs, outlines=outlines, output=output)

        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for i, time in enumerate(times):
                header = make_header(ra=56 + i, scale=0.14, size=(30, 20))
                header["DATE-AVG"] = time
                path = Path(temporary) / f"{time[:10].replace('-', '')}-2130-light-stack.fit"
                fits.writeto(path, np.zeros((20, 30), dtype=np.float32), header)
                paths.append(str(path))
            cwd = os.getcwd()
            os.chdir(temporary)
            try:
                self.run_main(
                    [*paths, "--query", "Vesta", "--latitude", "51.18", "--longitude", "-1.83", "--elevation", "100"],
                    solve_image={"side_effect": lambda path, *args, **kwargs: path},
                    get_ephemerides={"side_effect": ephemerides},
                    draw_chart={"side_effect": draw},
                )
            finally:
                os.chdir(cwd)

        self.assertEqual(calls["epochs"], sorted(epochs))
        self.assertEqual(calls["output"], Path("light-stack-chart-vesta-0001.png"))
        self.assertAlmostEqual(chart.chart_center(calls["outlines"][:1])[0], 57, places=3)
        name, ra, dec = calls["track"]
        self.assertEqual(name, "4 Vesta (A807 FA)")
        np.testing.assert_array_equal(ra, [56.0, 57.0])
        self.assertEqual(calls["title"], "4 Vesta (A807 FA)\n2026-11-11 to 2026-11-12")
        self.assertEqual(calls["labels"], ["2026-11-11 22:30 CET", "2026-11-12 22:30 CET"])

    def chart_target(self, positions):
        times = ["2026-11-06T19:10:00", "2026-11-08T20:00:00", "2026-11-11T21:30:00"]
        calls = {}

        def ephemerides(query, observer, requested):
            ra, dec = zip(*positions)
            return Table({"RA": ra, "DEC": dec, "targetname": ["4 Vesta (A807 FA)"] * len(ra)})

        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for i, time in enumerate(times):
                header = make_header(ra=20 + 40 * i, dec=0, scale=0.14, size=(30, 20))
                header["DATE-AVG"] = time
                path = Path(temporary) / f"{time[:10].replace('-', '')}-2000-light-stack.fit"
                fits.writeto(path, np.zeros((20, 30), dtype=np.float32), header)
                paths.append(str(path))
            output = self.run_main(
                [*paths, "--query", "vesta", "--latitude", "51.18", "--longitude", "-1.83", "--elevation", "100",
                 "--output", str(Path(temporary) / "chart.png")],
                solve_image={"side_effect": lambda path, *args, **kwargs: path},
                get_ephemerides={"side_effect": ephemerides},
                draw_chart={"side_effect": lambda outlines, output, **kwargs: calls.update(kwargs, outlines=outlines)},
            )
        return calls, output

    def test_query_keeps_only_images_showing_the_target(self):
        calls, output = self.chart_target([(20.5, 0.5), (90, 30), (101, -1)])
        self.assertEqual(len(calls["outlines"]), 2)
        self.assertEqual(calls["labels"], ["2026-11-06 20:10 CET", "2026-11-11 22:30 CET"])
        np.testing.assert_array_equal(calls["track"][1], [20.5, 101])
        self.assertEqual(calls["title"], "4 Vesta (A807 FA)\n2026-11-06 to 2026-11-11")
        self.assertIn("20261108-2000-light-stack.fit: Skipped: 4 Vesta (A807 FA) is outside the image", output)

    def test_query_outside_every_image_fails(self):
        with self.assertRaises(SystemExit) as raised:
            self.chart_target([(90, 30)] * 3)
        self.assertIn("in none of the images", str(raised.exception))

    def test_without_query_title_names_the_constellation(self):
        calls = {}
        with tempfile.TemporaryDirectory() as temporary:
            header = make_header(scale=0.14, size=(30, 20))
            header["DATE-AVG"] = "2026-11-11T21:49:00"
            path = Path(temporary) / "stack.fit"
            fits.writeto(path, np.zeros((20, 30), dtype=np.float32), header)
            self.run_main(
                [str(path), "--fov", "50"],
                solve_image={"side_effect": lambda path, *args, **kwargs: path},
                draw_chart={"side_effect": lambda outlines, output, **kwargs: calls.update(kwargs, output=output)},
            )
        self.assertEqual(calls["title"], "Taurus\n2026-11-11 22:49 CET")
        self.assertEqual(calls["fov"], 50)
        self.assertIsNone(calls["track"])
        self.assertEqual(calls["output"].name, "stack-chart.png")

if __name__ == "__main__":
    unittest.main()
