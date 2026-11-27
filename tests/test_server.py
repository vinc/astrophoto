import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from astropy.io import fits

from astrophoto import center, convert, server
from astrophoto.lib import camera, gphoto
from astrophoto.locate import PlateSolveError
from tests.test_cameras import FakeCamera


class ServerCase(unittest.TestCase):
    """Run the server on a free port, with gphoto2 replaced by a fake EOS 600D."""

    def setUp(self):
        self.fake = FakeCamera(best=1)
        patcher = patch.object(gphoto.subprocess, "run", side_effect=self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start(camera.CAMERAS["Canon EOS 600D"])

    def start(self, profile, observer={"lat": 51.18, "lon": -1.83, "elevation": 0.1}):
        self.app = server.App({"shutterspeed": "1/4", "iso": None}, profile, status_interval=3600,
                              center_settings={"shutterspeed": "2", "iso": "3200"}, observer=observer)
        self.server = server.Server(("127.0.0.1", 0), self.app)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self.stop_server, self.app, self.server)

    def stop_server(self, app, instance):
        app.close(timeout=60)
        instance.shutdown()
        instance.server_close()

    def get(self, path):
        with urlopen(self.base + path, timeout=10) as response:
            return response.status, response.headers.get("Content-Type"), response.read()

    def post(self, path, body=None):
        request = Request(self.base + path, data=json.dumps(body or {}).encode(), method="POST",
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def wait_for(self, condition, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = condition()
            if value:
                return value
            time.sleep(0.05)
        self.fail("Timed out")

    def latest(self, kind):
        return self.app.hub.latest.get(kind)


class ServerTests(ServerCase):
    def test_page_and_config_are_served(self):
        status, kind, page = self.get("/")
        self.assertEqual((status, kind), (200, "text/html; charset=utf-8"))
        self.assertIn(b"<title>Astrophoto</title>", page)
        status, _, body = self.get("/api/config")
        config = json.loads(body)
        self.assertEqual(config["camera"], "Canon EOS 600D")
        self.assertTrue(config["live_view"])
        self.assertEqual(config["focus"]["shutterspeed"], "1/4")
        with self.assertRaises(HTTPError) as error:
            self.get("/nothing")
        self.assertEqual(error.exception.code, 404)

    def test_live_focus_loop_publishes_frames_until_stopped(self):
        status, _ = self.post("/api/focus/start", {"live": True, "zoom": 5, "interval": 0})
        self.assertEqual(status, 200)
        event = self.wait_for(lambda: (self.latest("focus") or {}).get("frame", 0) >= 2 and self.latest("focus"))
        self.assertIn("HFR", event["title"])
        self.assertIsNotNone(event["best"])
        status, kind, image = self.get(event["image"])
        self.assertEqual(kind, "image/png")
        self.assertTrue(image.startswith(b"\x89PNG"))
        # One job at a time
        status, body = self.post("/api/focus/auto")
        self.assertEqual(status, 409)
        self.assertIn("Busy with focus", body["error"])
        status, body = self.post("/api/stop")
        self.assertEqual(body["stopped"], "focus")
        self.app.jobs.join(timeout=30)
        self.assertIsNone(self.latest("job")["name"])
        self.assertTrue(any("--capture-preview" in command and "eoszoom=5" in command for command in self.fake.commands))

    def test_drive_buttons_move_the_lens(self):
        status, _ = self.post("/api/focus/move", {"steps": 2, "size": 3})
        self.assertEqual(status, 200)
        self.assertEqual(self.fake.position, 6)
        status, _ = self.post("/api/focus/move", {"steps": -1, "size": 1})
        self.assertEqual(self.fake.position, 5)
        self.assertIn("Moved focus Near 1 x1", self.app.hub.logs[-1]["text"])

    def test_requests_are_validated(self):
        for path, body, message in [
            ("/api/focus/move", {"steps": 0}, "non-zero"),
            ("/api/focus/move", {"steps": 1, "size": 4}, "size must be one of 1, 2, 3"),
            ("/api/focus/start", {"zoom": 3}, "zoom must be one of 1, 5, 10"),
            ("/api/focus/start", {"interval": -1}, "interval must be a number of at least 0"),
            ("/api/focus/auto", {"span": 1}, "span must be a whole number of at least 2"),
            ("/api/focus/auto", {"samples": True}, "samples must be a whole number"),
            ("/api/nothing", {}, "Not found"),
        ]:
            with self.subTest(path=path, body=body):
                status, result = self.post(path, body)
                self.assertIn(status, (400, 404))
                self.assertIn(message, result["error"])
        request = Request(self.base + "/api/stop", data=b"[1]", method="POST")
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 400)

    def test_autofocus_leaves_the_lens_at_best_focus(self):
        status, _ = self.post("/api/focus/auto", {"span": 4, "step": 1, "samples": 1})
        self.assertEqual(status, 200)
        self.wait_for(lambda: self.app.jobs.running() == "autofocus")
        # The lens belongs to the scan until it ends
        status, body = self.post("/api/focus/move", {"steps": 1, "size": 1})
        self.assertEqual(status, 409)
        self.app.jobs.join(timeout=60)
        self.assertEqual(self.fake.position, 1)
        self.assertTrue(any(event["text"] == "Focused at step +1 of the scan" for event in self.app.hub.logs))
        # A result is only news to browsers connected at the time
        self.assertNotIn("autofocus", self.app.hub.latest)

    def test_a_failed_job_reports_gphoto2_error(self):
        def broken(command, **kwargs):
            raise subprocess.CalledProcessError(1, command, stderr="*** Error: No camera found. ***\n")

        with patch.object(gphoto.subprocess, "run", side_effect=broken):
            self.post("/api/focus/auto", {"span": 2})
            self.app.jobs.join(timeout=30)
        self.assertEqual(self.app.hub.logs[-1]["text"], "Autofocus failed: Error: No camera found.")

    def test_events_stream_replays_the_latest_state(self):
        self.app.hub.publish("status", camera="Canon EOS 600D", battery="75%", busy=False)
        self.app.hub.publish("error", message="old")
        self.app.hub.log("hello")
        with urlopen(self.base + "/api/events", timeout=10) as response:
            self.assertEqual(response.headers.get("Content-Type"), "text/event-stream")
            events = []
            while len(events) < 2:
                line = response.readline()
                if line.startswith(b"data: "):
                    events.append(json.loads(line.removeprefix(b"data: ")))
        self.assertEqual([event["type"] for event in events], ["status", "log"])
        self.assertEqual(events[0]["battery"], "75%")

    def test_status_reads_model_and_battery_but_skips_a_busy_camera(self):
        def summary(command, **kwargs):
            output = "Label: Camera Model\nCurrent: Canon EOS 600D\nEND\nLabel: Battery Level\nCurrent: 75%\nEND\n"
            return subprocess.CompletedProcess(command, 0, output, "")

        with patch.object(gphoto.subprocess, "run", side_effect=summary):
            self.app.check_status()
        status = self.latest("status")
        self.assertEqual((status["camera"], status["battery"], status["busy"]), ("Canon EOS 600D", "75%", False))
        held = threading.Event()
        release = threading.Event()

        def hold():
            with gphoto.camera_lock:
                held.set()
                release.wait(10)

        threading.Thread(target=hold).start()
        held.wait(10)
        try:
            with patch.object(gphoto.subprocess, "run") as run:
                self.app.check_status()
            run.assert_not_called()
        finally:
            release.set()
        status = self.latest("status")
        self.assertEqual((status["camera"], status["busy"]), ("Canon EOS 600D", True))

    def test_computer_readings_are_published(self):
        from astrophoto.lib.system import CpuUsage
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for relative, text in [("proc/stat", "cpu  100 0 100 800 0 0 0 0\n"),
                               ("proc/meminfo", "MemTotal: 1000 kB\nMemAvailable: 600 kB\n"),
                               ("sys/class/thermal/thermal_zone0/type", "acpitz\n"),
                               ("sys/class/thermal/thermal_zone0/temp", "36000\n")]:
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(text)
        cpu = CpuUsage(root)
        (root / "proc/stat").write_text("cpu  150 0 150 900 0 0 0 0\n")
        self.app.check_system(cpu, root)
        event = self.latest("system")
        self.assertEqual((event["cpu"], event["memory"], event["temperature"]), (50, 40, 36))
        self.assertGreater(event["disk_free"], 0)

    def test_live_view_requests_are_refused_without_live_view(self):
        self.start(camera.CAMERAS["Canon EOS 20D"])
        for path in ["/api/focus/auto", "/api/focus/move", "/api/focus/start"]:
            with self.subTest(path=path):
                status, body = self.post(path, {"steps": 1})
                self.assertEqual(status, 400)
                self.assertIn("does not have live view", body["error"])
        status, _ = self.post("/api/focus/start", {"live": False})
        self.assertEqual(status, 200)


class CenterServerTests(ServerCase):
    def solved_frame(self, directory):
        """Write a plate-solved frame centred on RA 10, Dec 20, one hundredth of a degree per pixel."""
        path = Path(directory) / "capture.new"
        header = fits.Header({
            "DATE-AVG": "2026-11-27T21:00:00", "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
            "CRVAL1": 10.0, "CRVAL2": 20.0, "CRPIX1": 100.5, "CRPIX2": 75.5, "CDELT1": -0.01, "CDELT2": 0.01,
        })
        image = np.random.default_rng(1).normal(100, 5, (150, 200)).astype(np.float32)
        fits.writeto(path, image, header)
        return path

    def test_config_reports_center_settings(self):
        _, _, body = self.get("/api/config")
        self.assertEqual(json.loads(body)["center"], {"shutterspeed": "2", "iso": "3200", "observer": True})

    def test_pointing_once_publishes_the_correction_and_image(self):
        solves = []

        def solve(path, directory, *args, **kwargs):
            solves.append(kwargs)
            return self.solved_frame(directory)

        with patch.object(center, "capture", return_value=Path("capture.CR2")) as capture, \
                patch.object(center, "solve_image", side_effect=solve):
            status, body = self.post("/api/center/start", {"ra": 10.3, "dec": 20.0, "loop": False, "iso": "6400"})
            self.assertEqual((status, body["target"]), (200, "RA 10.3, Dec 20"))
            self.app.jobs.join(timeout=60)
        self.assertEqual(capture.call_args.args[1:], ("2", "6400"))
        self.assertIsNone(solves[0]["near"])
        event = self.latest("center")
        self.assertRegex(event["title"], "^Pan (left|right) .* and (raise|lower) ")
        self.assertTrue(event["inside"])
        self.assertTrue(any("Separation from centre" in line for line in event["report"]))
        _, kind, image = self.get(event["image"])
        self.assertTrue(image.startswith(b"\x89PNG"))
        self.assertIsNone(self.app.jobs.running())

    def test_repeat_retries_failures_and_searches_near_the_last_solution(self):
        solves = []
        stop_after = threading.Event()

        def solve(path, directory, *args, **kwargs):
            solves.append(kwargs["near"])
            if len(solves) == 1:
                raise PlateSolveError("No solution")
            if len(solves) == 3:
                stop_after.set()
            return self.solved_frame(directory)

        with patch.object(center, "capture", return_value=Path("capture.CR2")), \
                patch.object(center, "solve_image", side_effect=solve), \
                patch.object(center, "resolve", return_value=None), \
                patch.object(center, "get_ephemerides", side_effect=lambda query, observer, epochs: [
                    {"RA": 10.0, "DEC": 20.0, "targetname": "4 Vesta"} for _ in epochs]):
            self.post("/api/center/start", {"query": "vesta", "loop": True})
            stop_after.wait(60)
            self.post("/api/stop")
            self.app.jobs.join(timeout=60)
        self.assertEqual(solves[:3], [None, None, (10.0, 20.0, center.NEAR_RADIUS)])
        self.assertTrue(any("No solution; retrying" in event["text"] for event in self.app.hub.logs))
        self.assertEqual(self.latest("center")["target"], "4 Vesta")

    def test_a_failed_single_attempt_is_reported(self):
        with patch.object(center, "capture", return_value=Path("capture.CR2")), \
                patch.object(center, "solve_image", side_effect=PlateSolveError("No solution")):
            self.post("/api/center/start", {"ra": 10, "dec": 20, "loop": False})
            self.app.jobs.join(timeout=30)
        self.assertEqual(self.app.hub.logs[-1]["text"], "Center failed: No solution")

    def test_center_requests_are_validated(self):
        with patch.object(center, "resolve", side_effect=center.AmbiguousObject("x matches several objects: A, B")):
            for body, message in [
                ({}, "Give a target name, or both RA and Dec"),
                ({"ra": 10}, "both RA and Dec"),
                ({"ra": 360, "dec": 0}, "RA must be in [0, 360)"),
                ({"ra": "ten", "dec": 0}, "ra must be a number"),
                ({"query": 3}, "query must be text"),
                ({"query": "x"}, "matches several objects"),
            ]:
                with self.subTest(body=body):
                    status, result = self.post("/api/center/start", body)
                    self.assertEqual(status, 400)
                    self.assertIn(message, result["error"])
        self.assertIsNone(self.app.jobs.running())

    def test_center_needs_the_observer_position(self):
        self.start(camera.CAMERAS["Canon EOS 600D"], observer=None)
        status, result = self.post("/api/center/start", {"query": "M31"})
        self.assertEqual(status, 400)
        self.assertIn("Set latitude, longitude and elevation", result["error"])
        _, _, body = self.get("/api/config")
        self.assertFalse(json.loads(body)["center"]["observer"])


class CaptureServerTests(ServerCase):
    def setUp(self):
        super().setUp()
        self.directory = Path(tempfile.mkdtemp(prefix="astrophoto-server-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.start(camera.CAMERAS["Canon EOS 600D"])
        self.app.root = self.directory
        self.captured = threading.Event()
        self.blank = set()
        for target, fake in [(server, "capture_series"), (convert, "convert")]:
            patcher = patch.object(target, fake, side_effect=getattr(self, f"fake_{fake}"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_capture_series(self, directory, settings, log, stop, on_frame):
        """Save a placeholder RAW file per frame, stopping between frames like a RAM series."""
        self.settings = settings
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(1, settings["frames"] + 1):
            if stop.is_set():
                break
            path = directory / f"20261127-210000-{index:03d}.CR2"
            path.write_text("raw")
            paths.append(path)
            on_frame(path)
            if index == 2:
                self.captured.set()
            time.sleep(0.05)
        return paths

    def fake_convert(self, path, tz):
        """Write a star field drifting a few pixels per frame, or empty sky for frames listed in self.blank."""
        from tests.test_livestack import sky
        index = int(path.stem.rsplit("-", 1)[1])
        if index in self.blank:
            image = np.random.default_rng(index).normal(500, 5, (480, 640)).astype(np.float32)
        else:
            image = sky(2 * index, -index, seed=index, shape=(480, 640), stars=60)
        output = path.with_suffix(".fit")
        header = fits.Header({"DATE-OBS": "2026-11-27T21:00:00.000", "DATE-AVG": "2026-11-27T21:00:02.500",
                              "RAWFILE": path.name, "EXPTIME": 5.0, "CCD-TEMP": 20 + index})
        fits.writeto(output, image, header, overwrite=True)
        return output

    def run_series(self, **body):
        status, result = self.post("/api/capture/start", body)
        self.assertEqual(status, 200, result)
        self.app.jobs.join(timeout=60)
        self.wait_for(lambda: self.app.processing.unfinished_tasks == 0 or self.app.processing.empty())
        self.wait_for(lambda: all(row["status"] not in ("captured",) for row in self.latest("series")["frames"]))
        return self.app.root / result["directory"]

    def test_config_reports_capture_settings_for_each_kind(self):
        _, _, body = self.get("/api/config")
        config = json.loads(body)
        self.assertEqual(config["kinds"], ["light", "dark", "flat"])
        self.assertEqual(config["capture"]["light"]["target"], "ram")
        self.assertEqual(config["capture"]["flat"]["shutterspeed"], "50")

    def test_light_frames_are_measured_and_stacked_as_they_arrive(self):
        directory = self.run_series(frames=4, iso="1600", target="ram")
        self.assertEqual(self.settings["iso"], "1600")
        self.assertEqual(self.settings["shutterspeed"], "5")
        series = self.latest("series")
        self.assertEqual((series["kind"], series["planned"]), ("light", 4))
        self.assertEqual([row["status"] for row in series["frames"]], ["stacked"] * 4)
        self.assertTrue(all(row["stars"] > 10 and row["hfr"] > 0 for row in series["frames"]))
        self.assertEqual([row["temperature"] for row in series["frames"]], [21, 22, 23, 24])
        # The field drifts 2 pixels right and 1 down per frame, half that once binned
        self.assertEqual(series["frames"][0]["shift"], [0.0, 0.0])
        self.assertAlmostEqual(series["frames"][3]["shift"][0], -3, delta=0.3)
        stacked = self.wait_for(lambda: (self.latest("stack") or {}).get("saved_frames") == 4 and self.latest("stack"))
        self.assertEqual(stacked["frames"], 4)
        self.assertEqual(stacked["saved"], f"{directory.name}-live-stack.fit")
        with fits.open(directory / stacked["saved"]) as hdus:
            self.assertEqual(hdus[0].header["NCOMBINE"], 4)
        self.assertTrue((directory / f"{directory.name}-live-stack.fit").exists())
        self.assertTrue((directory / f"{directory.name}-live-stack.png").exists())
        for kind in ("stack", "frame"):
            _, content_type, image = self.get(self.latest(kind)["image"])
            self.assertTrue(image.startswith(b"\x89PNG"))

    def test_stop_ends_the_series_after_the_current_frame(self):
        self.post("/api/capture/start", {"frames": 50})
        self.captured.wait(30)
        status, body = self.post("/api/stop")
        self.assertEqual(body["stopped"], "capture")
        self.app.jobs.join(timeout=30)
        self.assertLess(len(self.latest("series")["frames"]), 50)

    def test_a_frame_that_cannot_be_aligned_is_left_out(self):
        self.blank = {2}
        self.run_series(frames=3)
        rows = self.latest("series")["frames"]
        self.assertEqual([row["status"] for row in rows], ["stacked", "not stacked", "stacked"])
        self.assertIn("Could not align", rows[1]["problem"])
        self.assertEqual(self.latest("stack")["frames"], 2)

    def test_dark_frames_are_converted_but_not_stacked(self):
        self.run_series(kind="dark", frames=2)
        self.assertEqual([row["status"] for row in self.latest("series")["frames"]], ["converted"] * 2)
        self.assertIsNone(self.latest("stack"))

    def test_a_new_series_starts_a_new_stack(self):
        # Both start within the same minute
        with patch.object(server, "series_directory", return_value=Path("20261127-2100-light")):
            first = self.run_series(frames=2)
            second = self.run_series(frames=3)
        self.assertEqual((first.name, second.name), ("20261127-2100-light", "20261127-2100-light-2"))
        self.assertEqual((self.latest("stack")["frames"], len(self.latest("series")["frames"])), (3, 3))

    def test_a_frame_that_cannot_be_converted_is_marked_failed(self):
        with patch.object(convert, "convert", side_effect=ValueError("missing exposure timestamp")):
            self.run_series(frames=1)
        row = self.latest("series")["frames"][0]
        self.assertEqual((row["status"], row["problem"]), ("failed", "missing exposure timestamp"))

    def test_capture_requests_are_validated(self):
        for body, message in [
            ({"kind": "bias"}, "kind must be one of light, dark, flat"),
            ({"frames": 0}, "frames must be a whole number of at least 1"),
            ({"target": "sd"}, "target must be one of ram, card"),
        ]:
            with self.subTest(body=body):
                status, result = self.post("/api/capture/start", body)
                self.assertEqual(status, 400)
                self.assertIn(message, result["error"])


class CalibratedStackTests(ServerCase):
    def setUp(self):
        super().setUp()
        from astrophoto.lib import darks
        from tests.test_calibration import raw_frame, series
        self.root = Path(tempfile.mkdtemp(prefix="astrophoto-server-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        patcher = patch.object(darks, "library_path", return_value=self.root / "library")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app.root = self.root
        for name in ("20261127-2100-light", "20261127-2200-light", "20261127-2130-dark", "20261127-2140-flat",
                     "20261127-2150-dark", "notes"):
            (self.root / name).mkdir()
        series(self.root / "20261127-2100-light", count=3)
        for index in range(3):
            raw_frame(self.root / "20261127-2130-dark", f"dark{index}.fit", seed=index, dark=True)
            flat = self.root / "20261127-2140-flat" / f"flat{index}.fit"
            image = np.random.default_rng(index).normal(8000, 20, (240, 320)).astype(np.float32)
            fits.writeto(flat, image, fits.Header({"INSTRUME": "Canon EOS 600D", "ISO": 800, "BLACKLVL": 100.0,
                                                   "WHITELEV": 16000}))

    def stack(self, **body):
        status, result = self.post("/api/stack/start", {"lights": "20261127-2100-light", **body})
        self.assertEqual(status, 200, result)
        self.app.stacking.join(timeout=120)

    def test_series_directories_are_listed_newest_first(self):
        _, _, body = self.get("/api/directories")
        self.assertEqual(json.loads(body), {
            "light": ["20261127-2200-light", "20261127-2100-light"],
            "dark": ["20261127-2150-dark", "20261127-2130-dark"],
            "flat": ["20261127-2140-flat"],
        })

    def test_lights_are_stacked_with_darks_and_flats(self):
        self.stack(darks="20261127-2130-dark", flats="20261127-2140-flat")
        event = self.latest("calibrated")
        self.assertEqual((event["file"], event["darks"], event["flats"]),
                         ("20261127-2100-light-stack.fit", "20261127-2130-dark", "20261127-2140-flat"))
        header = fits.getheader(self.root / "20261127-2100-light" / event["file"])
        self.assertTrue(header["DARKCOR"] and header["FLATCOR"])
        _, _, image = self.get(event["image"])
        self.assertTrue(image.startswith(b"\x89PNG"))
        self.assertIsNone(self.latest("stacking")["name"])
        # The master dark went into the library, for a later session without darks
        self.stack(darks="library")
        self.assertEqual(self.latest("calibrated")["darks"], "library")

    def test_progress_is_published_while_stacking(self):
        events = []
        publish = self.app.hub.publish

        def record(kind, /, **data):
            if kind == "stack_progress":
                events.append(data)
            publish(kind, **data)

        with patch.object(self.app.hub, "publish", side_effect=record):
            self.stack(darks="20261127-2130-dark")
        stages = list(dict.fromkeys(event["stage"] for event in events))
        self.assertEqual(stages, ["Combining darks", "Aligning lights", "Stacking lights", "Saving"])
        self.assertTrue(all(event["directory"] == "20261127-2100-light" for event in events))
        self.assertEqual((events[-1]["done"], events[-1]["total"]), (1, 1))
        self.assertTrue(all(event["elapsed"] >= 0 for event in events))

    def test_deband_defaults_to_the_stack_table_and_can_be_changed(self):
        _, _, body = self.get("/api/config")
        self.assertEqual(json.loads(body)["stack"], {"deband": 1.0})
        self.app.stack_settings = {"deband": 0.5}
        self.stack()
        header = fits.getheader(self.root / "20261127-2100-light" / "20261127-2100-light-stack.fit")
        self.assertEqual(header["DEBAND"], 0.5)
        self.stack(deband=0.8)
        self.assertEqual(self.latest("calibrated")["deband"], 0.8)
        self.assertIn("with deband 0.8", self.app.hub.logs[-2]["text"])
        for value in (1.5, -0.1, "high"):
            with self.subTest(deband=value):
                status, result = self.post("/api/stack/start", {"lights": "20261127-2100-light", "deband": value})
                self.assertEqual(status, 400)
                self.assertIn("deband", result["error"])

    def test_colour_stacks_are_saved_with_an_rgb_prefix(self):
        self.stack(rgb=True)
        event = self.latest("calibrated")
        self.assertEqual((event["file"], event["rgb"]), ("20261127-2100-light-rgb-stack.fit", True))
        _, _, image = self.get(event["image"])
        self.assertTrue(image.startswith(b"\x89PNG"))

    def test_a_missing_library_dark_is_reported(self):
        self.stack(darks="library")
        self.assertIsNone(self.latest("calibrated"))
        self.assertIn("No master dark for Canon EOS 600D at ISO 800 and 5 s", self.app.hub.logs[-1]["text"])

    def test_only_series_in_the_session_can_be_stacked(self):
        for body, message in [
            ({"lights": "../20261127-2100-light"}, "No light series named"),
            ({"lights": "notes"}, "No light series named notes"),
            ({"lights": "20261127-2200-light"}, "No FITS frames in 20261127-2200-light yet"),
            ({"lights": "20261127-2100-light", "darks": "20261127-2150-dark"}, "No FITS frames in 20261127-2150-dark"),
            ({"lights": "20261127-2100-light", "flats": "20261127-2130-dark"}, "No flat series named"),
            ({"lights": 3}, "Choose a light series"),
        ]:
            with self.subTest(body=body):
                status, result = self.post("/api/stack/start", body)
                self.assertEqual(status, 400)
                self.assertIn(message, result["error"])

    def test_stacking_runs_beside_a_camera_job(self):
        self.post("/api/focus/start", {"live": True, "interval": 0.2})
        self.stack()
        self.assertEqual(self.app.jobs.running(), "focus")
        self.assertIsNotNone(self.latest("calibrated"))
        self.post("/api/stop")


class LogTests(unittest.TestCase):
    def test_log_lines_are_printed_with_the_date_and_time(self):
        import contextlib
        import io
        import re
        hub = server.Hub()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            hub.log("Saved frame 1")
        self.assertRegex(output.getvalue(), r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\] Saved frame 1\n$")
        # Browsers get the text and the time separately, and format the time themselves
        self.assertEqual(hub.logs[-1]["text"], "Saved frame 1")
        self.assertAlmostEqual(hub.logs[-1]["time"], time.time(), delta=5)


class AddressTests(unittest.TestCase):
    def test_a_specific_host_is_shown_as_given(self):
        self.assertEqual(server.address("192.168.1.20"), "192.168.1.20")

    def test_a_wildcard_bind_shows_the_network_address(self):
        with patch.object(server.socket, "socket") as probe:
            probe.return_value.__enter__.return_value.getsockname.return_value = ("192.168.1.55", 40000)
            self.assertEqual(server.address("0.0.0.0"), "192.168.1.55")

    def test_without_a_network_the_host_name_is_shown(self):
        with patch.object(server.socket, "socket", side_effect=OSError("Network is unreachable")):
            self.assertEqual(server.address("0.0.0.0"), server.socket.gethostname())


if __name__ == "__main__":
    unittest.main()
