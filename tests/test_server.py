import json
from pathlib import Path
import subprocess
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from astropy.io import fits

from astrophoto import center, server
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
        app.close()
        app.jobs.join(timeout=30)
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
        self.app.hub.publish("status", camera="Canon EOS 600D", battery="75%", disk_free=10**10, busy=False)
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
        self.assertGreater(status["disk_free"], 0)
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


class AddressTests(unittest.TestCase):
    def test_a_specific_host_is_shown_as_given(self):
        self.assertEqual(server.addresses("192.168.1.20"), ["192.168.1.20"])

    def test_a_wildcard_bind_lists_this_machine(self):
        self.assertIn(server.socket.gethostname(), server.addresses("0.0.0.0"))


if __name__ == "__main__":
    unittest.main()
