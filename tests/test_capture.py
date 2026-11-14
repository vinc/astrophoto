import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class CaptureTests(unittest.TestCase):
    def run_capture(self, directory, *arguments, fail_stage="", config=None, camera_files="",
                    captured="", downloaded="", empty="", first_folder=""):
        # Fake camera: --capture-image stores the frame count, which --list-files and
        # --get-all-files then report and download, unless CAPTURED or DOWNLOADED override it;
        # past FIRST_FOLDER frames, the rest are listed in the next folder. Without --quiet,
        # the listing is in French, as gphoto2 translates it
        executable = directory / "bin" / "gphoto2"
        executable.parent.mkdir(exist_ok=True)
        executable.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" CALL "$@" >> "$CAPTURE_ARGUMENTS"\n'
            'for argument in "$@"; do\n'
            '  if [ "$argument" = "$FAIL_STAGE" ]; then exit 1; fi\n'
            '  case "$previous" in --frames) frames=$argument ;; --filename) filename=$argument ;; esac\n'
            '  previous=$argument\n'
            'done\n'
            'case " $* " in\n'
            '  *" --list-files "*)\n'
            '    printf "%b" "$CAMERA_FILES"\n'
            '    if [ -f "$CAMERA_STATE" ]; then\n'
            '      n=$(cat "$CAMERA_STATE"); first=${FIRST_FOLDER:-$n}\n'
            '      for folder in 100CANON 101CANON; do\n'
            '        if [ $folder = 100CANON ]; then count=$first; else count=$((n - first)); fi\n'
            '        [ $count -gt 0 ] || continue\n'
            '        case " $* " in\n'
            '          *" --quiet "*) i=1; while [ $i -le $count ]; do printf "/DCIM/%s/IMG_%04d.CR2\\n" $folder $i; i=$((i + 1)); done ;;\n'
            '          *) printf "Il y a %d fichiers dans le dossier « /DCIM/%s ».\\n" $count $folder\n'
            '             i=1; while [ $i -le $count ]; do printf "#%d     IMG_%04d.CR2  rd  8341 Ko image/x-canon-cr2\\n" $i $i; i=$((i + 1)); done ;;\n'
            '        esac\n'
            '      done\n'
            '    fi ;;\n'
            '  *" --capture-image "*) echo "${CAPTURED:-$frames}" > "$CAMERA_STATE" ;;\n'
            '  *" --get-all-files "*)\n'
            '    i=1; n=${DOWNLOADED:-$(cat "$CAMERA_STATE")}\n'
            '    while [ $i -le $n ]; do\n'
            '      if [ $i -le "${EMPTY:-0}" ]; then : > "${filename%/*}/frame$i.CR2"; else echo raw > "${filename%/*}/frame$i.CR2"; fi\n'
            '      i=$((i + 1))\n'
            '    done ;;\n'
            'esac\n'
        )
        executable.chmod(0o755)
        settings = directory / "config"
        if config is not None:
            (settings / "astrophoto").mkdir(parents=True)
            (settings / "astrophoto" / "config.toml").write_text(config)
        log = directory / "arguments.txt"
        result = subprocess.run(
            [sys.executable, "-m", "astrophoto", "capture", *arguments], cwd=directory,
            env=dict(os.environ, PATH=f"{executable.parent}:{os.environ['PATH']}", PYTHONPATH=str(ROOT),
                     XDG_CONFIG_HOME=str(settings), CAPTURE_ARGUMENTS=str(log), FAIL_STAGE=fail_stage,
                     CAMERA_FILES=camera_files, CAMERA_STATE=str(directory / "camera"),
                     CAPTURED=str(captured), DOWNLOADED=str(downloaded), EMPTY=str(empty),
                     FIRST_FOLDER=str(first_folder)),
            capture_output=True, text=True,
        )
        arguments = [call.splitlines() for call in log.read_text().split("CALL\n")[1:]] if log.exists() else []
        folders = [path for path in directory.iterdir() if path.is_dir() and path.name not in ("bin", "config")]
        return result, arguments, folders

    def test_default_light_dark_and_flat_modes(self):
        for mode, suffix, shutter in [(None, "-light", "5"), ("light", "-light", "5"), ("dark", "-dark", "5"), ("flat", "-flat", "50")]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                result, arguments, folders = self.run_capture(directory, *([mode] if mode else []))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(arguments), 5)
                listing, capture, download, check, cleanup = arguments
                self.assertEqual(listing, ["--quiet", "--list-files"])
                self.assertEqual(check, ["--quiet", "--list-files"])
                self.assertEqual(len(list(folders[0].iterdir())), 20)
                self.assertEqual(len(folders), 1)
                self.assertRegex(folders[0].name, rf"^\d{{8}}-\d{{4}}{suffix}$")
                self.assertIn(f"shutterspeed={shutter}", capture)
                self.assertIn("iso=800", capture)
                self.assertIn("aperture=4.0", capture)
                self.assertIn("imageformat=RAW", capture)
                self.assertFalse(any(argument.startswith("capturetarget=") for argument in capture))
                self.assertIn("--capture-image", capture)
                self.assertNotIn("--capture-image-and-download", capture)
                self.assertEqual(capture[capture.index("--interval") + 1], "1")
                self.assertEqual(capture[capture.index("--frames") + 1], "20")
                self.assertEqual(download, [
                    "--folder", "/", "--recurse", "--filename",
                    f"{folders[0].name}/%Y%m%d-%H%M%S.%C", "--get-all-files",
                ])
                self.assertEqual(cleanup, ["--folder", "/DCIM/100CANON", "--delete-file", "1-20"])

    def test_a_series_split_across_folders_is_deleted_from_both(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, commands, folders = self.run_capture(Path(temporary), first_folder=12)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(folders[0].iterdir())), 20)
            self.assertEqual(commands[-2:], [
                ["--folder", "/DCIM/100CANON", "--delete-file", "1-12"],
                ["--folder", "/DCIM/101CANON", "--delete-file", "1-8"],
            ])

    def test_a_series_in_the_next_folder_is_deleted_from_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, commands, _ = self.run_capture(Path(temporary), first_folder=0)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(commands[-1], ["--folder", "/DCIM/101CANON", "--delete-file", "1-20"])

    def test_failed_capture_or_download_never_deletes_files(self):
        for stage, expected_calls in [("--capture-image", 2), ("--get-all-files", 3)]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                result, commands, folders = self.run_capture(Path(temporary), fail_stage=stage)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(commands), expected_calls)
                self.assertEqual(len(folders), 1)
                self.assertTrue(all("--delete-file" not in command for command in commands))

    def test_failed_deletion_reports_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, commands, _ = self.run_capture(Path(temporary), fail_stage="--delete-file")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(commands), 5)

    def test_incomplete_download_leaves_every_picture_on_the_camera(self):
        for options, message in [
            ({"downloaded": 19}, "Expected 20 pictures, found 20 on the camera and 19 downloaded;"),
            ({"empty": 1}, "found 20 on the camera and 20 downloaded, 1 of them empty;"),
            ({"captured": 19}, "found 19 on the camera and 19 downloaded;"),
        ]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                result, commands, folders = self.run_capture(Path(temporary), **options)
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stderr)
                self.assertIn("left them all on the camera", result.stderr)
                self.assertEqual(commands[-1], ["--quiet", "--list-files"])
                self.assertTrue(all("--delete-file" not in command for command in commands))
                self.assertEqual(len(folders), 1)

    def test_invalid_arguments_do_not_capture_or_create_a_session(self):
        for arguments in [("unknown",), ("dark", "extra"), ("--frames", "0")]:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temporary:
                result, commands, folders = self.run_capture(Path(temporary), *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("usage: astrophoto capture", result.stderr)
                self.assertEqual(commands, [])
                self.assertEqual(folders, [])

    def test_missing_gphoto2_stops_before_creating_a_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = subprocess.run(
                [sys.executable, "-m", "astrophoto", "capture"], cwd=directory,
                env=dict(os.environ, PATH=str(directory), PYTHONPATH=str(ROOT), XDG_CONFIG_HOME=str(directory)),
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("gphoto2 not found", result.stderr)
            self.assertEqual(list(directory.iterdir()), [])

    def test_help_values_list_the_camera_choices_without_capturing(self):
        for arguments, queried in [
            (("--shutterspeed", "help"), ["shutterspeed"]),
            (("flat", "--aperture", "help", "--iso", "help", "--shutterspeed", "1/4000"), ["iso", "aperture"]),
        ]:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temporary:
                result, commands, folders = self.run_capture(Path(temporary), *arguments, config="[capture]\nshutter = 8\n")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(commands, [["--get-config", name] for name in queried])
                self.assertEqual(folders, [])

    def test_failed_choice_listing_reports_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, commands, _ = self.run_capture(Path(temporary), "--iso", "help", fail_stage="--get-config")
            self.assertEqual(result.returncode, 1)
            self.assertEqual(commands, [["--get-config", "iso"]])

    def test_files_already_on_the_card_stop_before_capturing(self):
        listing = "/store_00010001/DCIM/100CANON/IMG_0001.CR2\\n/store_00010001/DCIM/101CANON/IMG_0002.CR2\\n"
        with tempfile.TemporaryDirectory() as temporary:
            result, commands, folders = self.run_capture(Path(temporary), camera_files=listing)
            self.assertEqual(result.returncode, 1)
            self.assertIn("already holds 2 files", result.stderr)
            self.assertEqual(commands, [["--quiet", "--list-files"]])
            self.assertEqual(folders, [])

    def test_unexpected_card_listing_stops_before_capturing(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, commands, folders = self.run_capture(Path(temporary), camera_files="IMG_0001.CR2\\n")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Unexpected gphoto2 file listing: IMG_0001.CR2", result.stderr)
            self.assertEqual(folders, [])

    def test_failed_card_listing_stops_before_capturing(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, commands, folders = self.run_capture(Path(temporary), fail_stage="--list-files")
            self.assertEqual(result.returncode, 1)
            self.assertEqual(commands, [["--quiet", "--list-files"]])
            self.assertEqual(folders, [])

    def test_config_and_options_override_the_defaults(self):
        config = (
            "latitude = 51.18\n"
            "[capture]\nframes = 30\niso = 400\nshutterspeed = 8\n"
            "[capture.flat]\nshutterspeed = \"1/2500\"\n"
        )
        for arguments, expected in [
            ((), {"frames": "30", "iso=": "400", "aperture=": "4.0", "shutterspeed=": "8"}),
            (("dark",), {"frames": "30", "iso=": "400", "shutterspeed=": "8"}),
            (("flat",), {"frames": "30", "iso=": "400", "shutterspeed=": "1/2500"}),
            (("flat", "--shutterspeed", "1/4000", "--frames", "5", "--aperture", "5.6"),
             {"frames": "5", "aperture=": "5.6", "shutterspeed=": "1/4000"}),
        ]:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temporary:
                result, commands, _ = self.run_capture(Path(temporary), *arguments, config=config)
                self.assertEqual(result.returncode, 0, result.stderr)
                _, capture, _, _, cleanup = commands
                for name, value in expected.items():
                    if name == "frames":
                        self.assertEqual(capture[capture.index("--frames") + 1], value)
                        self.assertEqual(cleanup[-1], f"1-{value}")
                    else:
                        self.assertIn(f"{name}{value}", capture)

    def test_flat_defaults_survive_a_capture_shutterspeed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, commands, _ = self.run_capture(Path(temporary), "flat", config="[capture]\nshutterspeed = 8\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("shutterspeed=50", commands[1])

    def test_invalid_config_stops_before_capturing(self):
        for config, message in [
            ("capture = 3\n", "[capture] in .* must be a table"),
            ("[capture]\nshutter = 8\n", "Unknown setting shutter in [capture]"),
            ("[capture]\nframes = 2.5\n", "frames in [capture] of .* must be an integer"),
            ("[capture.flat]\niso = true\n", "iso in [capture.flat] of .* must be a number or a string"),
        ]:
            with self.subTest(config=config), tempfile.TemporaryDirectory() as temporary:
                arguments = ("flat",) if "flat" in config else ()
                result, commands, folders = self.run_capture(Path(temporary), *arguments, config=config)
                self.assertEqual(result.returncode, 2)
                self.assertRegex(result.stderr, message.replace("[", r"\[").replace("]", r"\]"))
                self.assertEqual(commands, [])
                self.assertEqual(folders, [])

if __name__ == "__main__":
    unittest.main()
