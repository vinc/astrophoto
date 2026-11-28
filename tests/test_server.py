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

from astro import center, convert, server, visible, weather
from astro.lib import camera, gphoto
from astro.locate import PlateSolveError
from tests.test_cameras import FakeCamera
from tests.test_localtime import local_time_zone
from tests.test_weather import forecast


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
        self.assertIn(b"<title>Astro</title>", page)
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

    def test_log_lines_reach_browsers_only_when_debugging(self):
        _, _, body = self.get("/api/config")
        self.assertFalse(json.loads(body)["debug"])
        self.app.hub.log("hidden")
        client, backlog = self.app.hub.subscribe()
        self.addCleanup(self.app.hub.unsubscribe, client)
        self.app.hub.log("hidden too")
        self.assertNotIn("log", [event["type"] for event in backlog])
        self.assertTrue(client.empty())
        # The server keeps them all the same
        self.assertEqual([event["text"] for event in self.app.hub.logs][-2:], ["hidden", "hidden too"])
        debug = server.App({"shutterspeed": "1/4", "iso": None}, debug=True)
        self.addCleanup(debug.close, 10)
        self.assertTrue(debug.config()["debug"])
        debug.hub.log("shown")
        self.assertEqual([event["text"] for event in debug.hub.subscribe()[1] if event["type"] == "log"], ["shown"])

    def test_drive_buttons_move_the_lens(self):
        status, _ = self.post("/api/focus/move", {"steps": 2, "size": 3})
        self.assertEqual(status, 200)
        self.assertEqual(self.fake.position, 6)
        status, _ = self.post("/api/focus/move", {"steps": -1, "size": 1})
        self.assertEqual(self.fake.position, 5)
        self.assertIn("Moved focus Near 1 x1", self.app.hub.logs[-1]["text"])

    def test_a_job_cannot_start_while_a_lens_move_is_underway(self):
        checked, release = threading.Event(), threading.Event()

        def slow_move(steps, size):
            checked.set()
            release.wait(10)

        with patch.object(server, "move_focus", side_effect=slow_move):
            mover = threading.Thread(target=self.app.focus_move, args=({"steps": 1, "size": 1},))
            mover.start()
            checked.wait(10)
            # Starting a capture now waits for the move rather than getting it partway through
            starter = threading.Thread(target=self.app.jobs.start, args=("capture", lambda stop: stop.wait(5)))
            starter.start()
            starter.join(0.5)
            # Read without the lock, which the move holds
            self.assertTrue(starter.is_alive())
            self.assertIsNone(self.app.jobs.name)
            release.set()
            mover.join(10)
            starter.join(10)
        self.assertEqual(self.app.jobs.running(), "capture")
        self.app.jobs.stop()

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
        self.app.hub.send_logs = True
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
        from astro.lib.system import CpuUsage
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
        """Write a plate-solved frame centered on RA 10, Dec 20, one hundredth of a degree per pixel."""
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
        self.assertTrue(any("Separation from center" in line for line in event["report"]))
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
        self.directory = Path(tempfile.mkdtemp(prefix="astro-server-"))
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

    def test_the_lens_stays_put_during_a_capture(self):
        self.post("/api/capture/start", {"frames": 50})
        self.captured.wait(30)
        status, result = self.post("/api/focus/move", {"steps": 1, "size": 1})
        self.assertEqual(status, 409)
        self.assertIn("only be moved while idle or focusing, not during capture", result["error"])
        self.assertFalse(any(option.startswith("manualfocusdrive") for command in self.fake.commands
                             for option in command))
        self.post("/api/stop")

    def test_a_failed_live_stack_save_does_not_stop_frame_processing(self):
        from astro.lib import livestack
        with patch.object(livestack.LiveStack, "write", side_effect=OSError("No space left on device")):
            self.run_series(frames=2)
        self.assertTrue(any("Could not save the live stack" in event["text"] and "No space left" in event["text"]
                            for event in self.app.hub.logs))
        self.assertEqual([row["status"] for row in self.latest("series")["frames"]], ["stacked", "stacked"])
        # The worker is still there for the next series, and its stack is saved
        directory = self.run_series(frames=2)
        self.wait_for(lambda: (directory / f"{directory.name}-live-stack.fit").exists())
        self.assertEqual([row["status"] for row in self.latest("series")["frames"]], ["stacked", "stacked"])

    # Index 50 is the flat frames' default shutter speed, which gphoto2 takes as an index for want of a label
    SHUTTER_SPEEDS = ["bulb", "30", "15", "8", "5"] + [f"1/{1000 + n}" for n in range(55)]
    SHUTTER_SPEEDS[50] = "1/2000"
    CHOICES = {
        "iso": ["Auto", "100", "400", "800", "1600", "3200"],
        "aperture": ["4", "4.5", "5", "5.6"],
        "shutterspeed": SHUTTER_SPEEDS,
    }

    def answer_with_choices(self, command, **kwargs):
        names = [command[i + 1] for i, option in enumerate(command) if option == "--get-config"]
        output = ""
        for name in names:
            current = {"cameramodel": "Canon EOS 600D", "batterylevel": "75%"}.get(name, "")
            choices = "".join(f"Choice: {index} {label}\n" for index, label in enumerate(self.CHOICES.get(name, [])))
            output += f"Label: {name}\nType: RADIO\nCurrent: {current}\n{choices}END\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    def read_choices(self):
        with patch.object(gphoto.subprocess, "run", side_effect=self.answer_with_choices):
            self.app.check_status()
        return self.latest("choices")

    def test_the_camera_choices_and_defaults_are_published_as_labels(self):
        choices = self.read_choices()
        self.assertEqual(choices["settings"], self.CHOICES)
        self.assertEqual({key: choices["capture"]["light"][key] for key in ("iso", "aperture", "shutterspeed")},
                         {"iso": "800", "aperture": "4", "shutterspeed": "5"})
        self.assertEqual(choices["capture"]["flat"]["shutterspeed"], "1/2000")
        self.assertEqual(choices["capture"]["light"]["frames"], 20)

    def test_a_series_is_taken_with_the_labels_of_the_camera_choices(self):
        self.read_choices()
        self.run_series(frames=1, kind="flat", aperture="5.6")
        self.assertEqual((self.settings["iso"], self.settings["aperture"], self.settings["shutterspeed"]),
                         ("800", "5.6", "1/2000"))
        status, result = self.post("/api/capture/start", {"iso": "12800"})
        self.assertEqual(status, 400)
        self.assertIn("iso 12800 is not one of the camera's choices: Auto, 100, 400", result["error"])

    def test_without_a_camera_settings_are_passed_as_configured(self):
        with patch.object(server, "camera_status", return_value=(None, None)):
            self.app.check_status()
        self.assertEqual(self.latest("choices")["settings"], None)
        self.assertEqual(self.latest("choices")["capture"]["flat"]["shutterspeed"], "50")
        self.run_series(frames=1, kind="flat")
        self.assertEqual(self.settings["shutterspeed"], "50")

    def test_focus_and_center_take_the_camera_choices_or_its_own_setting(self):
        choices = self.read_choices()
        # The focus shutter speed is not offered, so it is sent as configured; the center's 2 is an index
        self.assertEqual(choices["focus"], {"iso": None, "shutterspeed": "1/4"})
        self.assertEqual(choices["center"], {"iso": "3200", "shutterspeed": "15"})
        with patch.object(self.app.jobs, "start") as start, patch.object(self.app, "focus_loop") as loop:
            self.assertEqual(self.post("/api/focus/start", {"live": False, "iso": "1", "shutterspeed": ""})[0], 200)
            start.call_args.args[1](None)
        # An empty setting leaves the camera's own
        self.assertEqual(loop.call_args.args[3:5], ("100", None))
        for path, body in [("/api/focus/start", {"iso": "12800"}), ("/api/focus/auto", {"iso": "12800"}),
                           ("/api/center/start", {"ra": 10, "dec": 20, "shutterspeed": "1/3"})]:
            with self.subTest(path=path):
                status, result = self.post(path, body)
                self.assertEqual(status, 400)
                self.assertIn("is not one of the camera's choices", result["error"])

    def test_a_refresh_reads_the_camera_again(self):
        self.app.status_wanted.clear()
        self.assertEqual(self.post("/api/camera/refresh"), (200, {"ok": True}))
        self.assertTrue(self.app.status_wanted.is_set())

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
        from astro.lib import darks
        from tests.test_calibration import raw_frame, series
        self.root = Path(tempfile.mkdtemp(prefix="astro-server-"))
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

    def test_raw_files_without_fits_frames_are_converted_first(self):
        from tests.test_calibration import raw_frame
        # A series captured with the capture command, never converted, and a dark series converted halfway
        lights, darks = self.root / "20261127-2200-light", self.root / "20261127-2150-dark"
        for index in range(3):
            (lights / f"frame{index}.CR2").write_text("raw")
            (darks / f"dark{index}.CR2").write_text("raw")
        raw_frame(darks, "dark0.fit", seed=0, dark=True)
        converted = []

        def fake_convert(path, tz):
            converted.append(path.name)
            index = int(path.stem[-1])
            if path.stem.startswith("dark"):
                return raw_frame(path.parent, f"{path.stem}.fit", seed=index, dark=True)
            return raw_frame(path.parent, f"{path.stem}.fit", 2 * index, -index, seed=index)

        events = []
        publish = self.app.hub.publish

        def record(kind, /, **data):
            if kind == "stack_progress":
                events.append(data)
            publish(kind, **data)

        with patch.object(convert, "convert", side_effect=fake_convert), \
                patch.object(self.app.hub, "publish", side_effect=record):
            status, result = self.post("/api/stack/start", {"lights": "20261127-2200-light", "darks": "20261127-2150-dark"})
            self.assertEqual(status, 200, result)
            self.app.stacking.join(timeout=120)
        self.assertEqual(sorted(converted), ["dark1.CR2", "dark2.CR2", "frame0.CR2", "frame1.CR2", "frame2.CR2"])
        self.assertEqual(events[0]["stage"], "Converting")
        self.assertEqual(self.latest("calibrated")["file"], "20261127-2200-light-stack.fit")
        header = fits.getheader(lights / "20261127-2200-light-stack.fit")
        self.assertEqual((header["NCOMBINE"], header["NDARKS"]), (3, 3))
        self.assertIn("Converting 5 RAW files to FITS", [event["text"] for event in self.app.hub.logs])

    def test_an_unreadable_raw_file_is_left_out_of_the_stack(self):
        from tests.test_calibration import raw_frame
        lights = self.root / "20261127-2200-light"
        raw_frame(lights, "frame0.fit", seed=0)
        raw_frame(lights, "frame1.fit", 2, -1, seed=1)
        (lights / "frame2.CR2").write_text("incomplete")
        with patch.object(convert, "convert", side_effect=ValueError("frame2.CR2: missing exposure timestamp")):
            self.stack(lights="20261127-2200-light")
        self.assertIn("Could not convert frame2.CR2, leaving it out: frame2.CR2: missing exposure timestamp",
                      [event["text"] for event in self.app.hub.logs])
        self.assertEqual(fits.getheader(lights / "20261127-2200-light-stack.fit")["NCOMBINE"], 2)

    def test_a_series_with_only_unreadable_raw_files_is_reported(self):
        (self.root / "20261127-2200-light" / "frame0.CR2").write_text("incomplete")
        with patch.object(convert, "convert", side_effect=ValueError("missing exposure timestamp")):
            self.stack(lights="20261127-2200-light")
        self.assertIsNone(self.latest("calibrated"))
        self.assertEqual(self.app.hub.logs[-1]["text"], "Stacking failed: No usable frames in 20261127-2200-light")

    def test_a_raw_file_converted_meanwhile_is_not_converted_again(self):
        from tests.test_calibration import raw_frame
        lights = self.root / "20261127-2200-light"
        (lights / "frame0.CR2").write_text("raw")
        # The frame processor converted it between the listing and its turn
        raw_frame(lights, "frame0.fit", seed=0)
        with patch.object(convert, "convert") as conversion:
            self.assertEqual(self.app.convert_raw(lights / "frame0.CR2"), lights / "frame0.fit")
        conversion.assert_not_called()

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

    def test_color_stacks_are_saved_with_an_rgb_prefix(self):
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
            ({"lights": "20261127-2200-light"}, "No frames in 20261127-2200-light yet"),
            ({"lights": "20261127-2100-light", "darks": "20261127-2150-dark"}, "No frames in 20261127-2150-dark yet"),
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
    STAMP = r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\] "

    def test_every_printed_line_starts_with_the_date_and_time(self):
        import io
        output = io.StringIO()
        stream = server.Timestamped(output)
        print("Combined 24 dark frames", file=stream)
        stream.write("Aligned frame1.fit\nAligned ")
        stream.write("frame2.fit\n\n")
        self.assertRegex(output.getvalue(), rf"^{self.STAMP}Combined 24 dark frames\n{self.STAMP}Aligned frame1.fit\n"
                                            rf"{self.STAMP}Aligned frame2.fit\n\n$")
        self.assertEqual(stream.encoding, output.encoding)

    def test_lines_from_threads_printing_at_once_are_not_mixed(self):
        import io
        output = io.StringIO()
        stream = server.Timestamped(output)

        def chatter(name):
            for index in range(200):
                print(f"{name} line {index}", file=stream)

        threads = [threading.Thread(target=chatter, args=(name,)) for name in ("first", "second", "third")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 600)
        for line in lines:
            self.assertRegex(line, rf"^{self.STAMP}(first|second|third) line \d+$")

    def test_printed_lines_reach_browsers_once_a_hub_is_given(self):
        import contextlib
        import io
        hub = server.Hub()
        output, errors = server.Timestamped(io.StringIO()), server.Timestamped(io.StringIO(), error=True)
        print("Before the hub", file=output)
        output.hub = errors.hub = hub
        print("Combined 24 dark frames", file=output)
        # Details that commands indent, such as convert's for each frame, stay in the terminal
        print("frame.CR2 -> frame.fit\n  Exposure: 5.0 s\n", file=output)
        print("*** Error: No camera found ***", file=errors)
        self.assertEqual([(event["text"], event.get("error", False)) for event in hub.logs], [
            ("Combined 24 dark frames", False), ("frame.CR2 -> frame.fit", False),
            ("*** Error: No camera found ***", True),
        ])
        # The server's own messages go out once, through the wrapped output
        with contextlib.redirect_stdout(output):
            hub.log("Saved frame 1")
        self.assertEqual([event["text"] for event in hub.logs].count("Saved frame 1"), 1)

    def test_log_lines_reach_browsers_without_the_stamp(self):
        import contextlib
        import io
        hub = server.Hub()
        with contextlib.redirect_stdout(io.StringIO()):
            hub.log("Saved frame 1")
        self.assertEqual(hub.logs[-1]["text"], "Saved frame 1")
        self.assertAlmostEqual(hub.logs[-1]["time"], time.time(), delta=5)


class SkyTests(ServerCase):
    """The Weather and Visible tabs, with Open-Meteo and JPL replaced by canned answers."""

    def get_json(self, path):
        try:
            with urlopen(self.base + path, timeout=60) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_weather_lists_the_hours_of_the_night_with_ratings(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        answer = lambda *_, params, **__: forecast(params, cloud_cover=[0, 80], wind_gusts_10m=[3, 3])
        with local_time_zone("UTC"), patch.object(weather, "CACHE", Path(cache.name)), \
                patch.object(weather.requests, "get", side_effect=answer) as get:
            status, result = self.get_json("/api/weather?time=2026-12-21T19:00")
        self.assertEqual(status, 200)
        self.assertEqual(get.call_args.kwargs["params"]["elevation"], 100)
        self.assertEqual(get.call_args.kwargs["params"]["start_hour"], "2026-12-21T19:00")
        self.assertEqual(result["model"], weather.DEFAULT_MODEL)
        hours = result["hours"]
        self.assertEqual([hours[0]["hour"], hours[-1]["hour"]], [19, 6])
        self.assertEqual([hour["rating"] for hour in hours[:2]], ["green", "red"])
        self.assertEqual(hours[0]["date"], "Mon 21")
        labels = [[row["label"] for row in group] for group in result["groups"]]
        self.assertEqual(labels[0], ["Cloud (%)", "High (%)", "Mid (%)", "Low (%)"])
        self.assertEqual(result["groups"][0][0]["cells"][:2], ["0", "80"])

    def test_weather_errors_are_reported(self):
        with patch.object(weather, "night_forecast", side_effect=SystemExit("Open-Meteo request failed: Bad model")):
            status, result = self.get_json("/api/weather")
        self.assertEqual(status, 502)
        self.assertEqual(result["error"], "Open-Meteo request failed: Bad model")
        status, result = self.get_json("/api/weather?time=not%20a%20time%20at%20all")
        self.assertEqual(status, 400)
        self.assertIn("Could not understand the time", result["error"])

    def test_visible_lists_bright_planets_brightest_first(self):
        data = {"downloaded": "2026-11-07T00:00:00", "asteroids": [], "comets": []}
        with local_time_zone("Europe/Paris"), patch.object(visible, "load_minor_bodies", return_value=data):
            status, result = self.get_json("/api/visible?time=2026-11-07T21:11&limit=8")
        self.assertEqual(status, 200)
        self.assertEqual(result["label"], "2026-11-07 21:11 CET")
        self.assertEqual(result["sky"], "dark sky")
        self.assertEqual((result["limit"], result["min_altitude"]), (8, 10))
        names = [body["name"] for body in result["bodies"]]
        self.assertEqual([name for name in names if name in ("Saturn", "Uranus", "Neptune")],
                         ["Saturn", "Uranus", "Neptune"])
        magnitudes = [body["mag"] for body in result["bodies"]]
        self.assertEqual(magnitudes, sorted(magnitudes))
        saturn = result["bodies"][names.index("Saturn")]
        self.assertEqual(saturn["compass"], visible.compass(saturn["az"]))
        self.assertIsNone(result["warning"])

    def test_a_failed_orbit_update_is_reported_with_the_cached_report(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        path = Path(cache.name) / "minor-bodies.json"
        path.write_text(json.dumps({"downloaded": "2026-11-07T00:00:00", "asteroids": [], "comets": []}))
        error = visible.requests.ConnectionError("offline")
        with patch.object(visible, "CACHE", path), patch.object(visible, "download_minor_bodies", side_effect=error):
            status, result = self.get_json("/api/visible?time=2026-11-07T21:11&update=1")
        self.assertEqual(status, 200)
        self.assertEqual(result["downloaded"], "2026-11-07T00:00:00")
        self.assertEqual(result["warning"], "Could not update orbital elements: offline")

    def test_visible_requests_are_validated(self):
        for query, message in [("limit=bright", "limit must be a number"),
                               ("min_altitude=95", "min_altitude must be from -90 to 90")]:
            with self.subTest(query=query):
                status, result = self.get_json(f"/api/visible?{query}")
                self.assertEqual(status, 400)
                self.assertIn(message, result["error"])

    def test_weather_and_visible_need_the_observer_position(self):
        self.start(camera.CAMERAS["Canon EOS 600D"], observer=None)
        for path in ("/api/weather", "/api/visible"):
            with self.subTest(path=path):
                status, result = self.get_json(path)
                self.assertEqual(status, 400)
                self.assertIn("Set latitude, longitude and elevation", result["error"])
        _, _, body = self.get("/api/config")
        self.assertFalse(json.loads(body)["observer"])


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
