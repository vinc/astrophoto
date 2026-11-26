# Astrophoto manual

```
astrophoto <command> [options]
astrophoto <command> --help
```

Commands, in the order of a session:

| Command | Purpose |
|---|---|
| [`download`](#download) | Download plate-solving indexes and catalog data |
| [`weather`](#weather) | Hourly weather forecast for the night |
| [`visible`](#visible) | Solar System bodies above the horizon |
| [`focus`](#focus) | Measure and adjust focus from stars |
| [`center`](#center) | Point the camera at a target |
| [`capture`](#capture) | Capture light, dark, or flat frames |
| [`convert`](#convert) | Convert CR2 files to FITS |
| [`stack`](#stack) | Align and stack FITS images |
| [`locate`](#locate) | Mark a Solar System body in full-frame images |
| [`crop`](#crop) | Crop a Solar System body from images |
| [`chart`](#chart) | Star chart locating plate-solved images |

## Configuration

Settings are read from `~/.config/astrophoto/config.toml` (or
`$XDG_CONFIG_HOME/astrophoto/config.toml`). Command-line options override them.

Top-level keys apply to every command:

| Key | Meaning |
|---|---|
| `latitude`, `longitude` | Observer position in degrees |
| `elevation` | Observer elevation in meters |
| `timezone` | Timezone of the camera clock, e.g. `"Europe/Paris"` (default: the system timezone) |
| `camera` | `"Canon EOS 20D"` or `"Canon EOS 600D"` |

A table named after a command holds that command's settings:

| Table | Keys |
|---|---|
| `[capture]` | `frames`, `iso`, `aperture`, `shutterspeed` |
| `[capture.light]`, `[capture.dark]`, `[capture.flat]` | The same keys, for one kind of frame |
| `[focus]` | `shutterspeed`, `iso` |
| `[center]` | `shutterspeed`, `iso` |
| `[solve]` | `scale_low`, `scale_high`: image width range in degrees for plate solving (default: 3 to 20) |

## Common options

Commands that need the observer position take `--latitude`, `--longitude`
(degrees) and `--elevation` (meters). They are required unless set in the
configuration.

Commands that open a window take `--plot` to open one for saved images,
`--no-plot` to print results only, and `--save-preview FILE` to save the
annotated image as a PNG without a window.

Images can be Canon RAW (CR2) or FITS files, including stacks, unless noted.

---

## download

```
astrophoto download
```

Downloads the astrometry.net index files used for plate solving, the star chart
data, and the star and deep-sky object catalogs into
`~/.local/share/astrophoto`. Run it once after installing.

## weather

```
astrophoto weather [--time TIME] [--update] [--model MODEL]
```

Prints an hourly forecast of cloud cover by layer, wind, temperature, dew point
and rain from Open-Meteo.

`--time TIME`
: Local time in ISO format or words like `"tomorrow night"` (default: now).

`--update`
: Download a fresh forecast even if one less than 15 minutes old is cached.

`--model MODEL`
: Open-Meteo weather model (default: `meteofrance_seamless`).

## visible

```
astrophoto visible [--time TIME] [--limit MAG] [--min-altitude DEG] [--update]
```

Lists planets and asteroids above the horizon, brightest first, with the Sun
and Moon altitude and the Moon phase.

`--time TIME`
: Local time in ISO format or words like `"tonight"` (default: now).

`--limit MAG`
: Faintest magnitude to list (default: 10).

`--min-altitude DEG`
: Lowest altitude in degrees (default: 10).

`--update`
: Download fresh orbital elements from JPL.

## focus

```
astrophoto focus [options] [FILE ...]
```

Measures the half-flux radius (HFR) of stars; a smaller HFR means sharper
focus. Without files, it captures repeatedly so you can adjust the focus
between frames, showing the latest and best HFR and a chart of past readings.

`FILE ...`
: Measure saved images instead of capturing.

`--live`
: Measure zoomed live-view frames instead of RAW captures. Much faster, and
  does not trigger autofocus. Needs a camera with live view.

`--auto`
: Drive the lens through a range of focus positions in live view, fit the
  HFR curve, and leave the lens at its sharpest position. Needs a camera with
  live view and the lens switch on AF. On an EOS 600D, it also sets C.Fn IV-1
  to "AE lock/AF" if needed, so the shutter button no longer autofocuses and
  later captures keep the focus; autofocus by hand then uses the rear ✱ button.

`--zoom {1,5,10}`
: Live-view zoom for `--live` and `--auto` (default: 5).

`--step {1,2,3}`
: Size of each lens step for `--auto`, from smallest to largest (default: 2).

`--span N`
: Steps scanned on each side of the starting position for `--auto`
  (default: 8). Increase it if the rough focus is far off.

`--samples N`
: Live-view frames measured at each step for `--auto` (default: 3). More
  frames give a steadier curve when few stars are visible.

`--shutterspeed SPEED`
: Shutter speed for RAW captures (default: `1/4`).

`--iso ISO`
: ISO for captures; with live view, it sets how bright the preview is.

`--frames N`
: Number of captures; 0 repeats until stopped (default: 0).

`--interval SECONDS`
: Pause between captures (default: 1).

`--radius PIXELS`
: Measuring aperture radius (default: 12, larger for cameras with smaller
  pixels). Increase it if stars are reported as out of range.

Configuration: `[focus]`, `camera`.

## center

```
astrophoto center (--query NAME | --ra DEG --dec DEG) [options] [FILE]
```

Captures an image, plate-solves it, and prints how far to pan and tilt the
camera to bring the target to the centre of the frame. It also warns when the
target is too large for the frame.

`FILE`
: Use a saved image instead of capturing.

`--query NAME`
: Star, deep-sky object, or Solar System body, by name or designation
  (e.g. `vesta`, `M31`, `Betelgeuse`).

`--ra DEG`, `--dec DEG`
: Fixed ICRS coordinates in degrees, instead of `--query`.

`--loop`
: Keep capturing and solving until stopped, to follow the camera as you
  move it.

`--shutterspeed SPEED`, `--iso ISO`
: Capture settings (default: the camera's current settings).

Configuration: `[center]`, `[solve]`, `camera`, `timezone`.

## capture

```
astrophoto capture [light|dark|flat] [--frames N] [--iso ISO] [--aperture F] [--shutterspeed SPEED]
```

Captures a sequence of RAW frames, downloads them to a new
`YYYYMMDD-HHMM-<kind>` directory, and deletes them from the camera once they
are all downloaded. The camera card must be empty before starting.

`light|dark|flat`
: Kind of frames (default: `light`).

`--frames N`
: Number of frames (default: 20).

`--iso ISO`, `--aperture F`, `--shutterspeed SPEED`
: Camera settings, as the camera names them (defaults: `800`, `4.0`, `5`, or
  `50` for flats). Pass `help` to list the camera's choices.

Configuration: `[capture]`, `[capture.light]`, `[capture.dark]`,
`[capture.flat]`.

## convert

```
astrophoto convert FILE ...
```

Converts Canon CR2 files to FITS files next to them, keeping the raw Bayer
data and recording the exposure time, camera and lens settings.

Configuration: `timezone`.

## stack

```
astrophoto stack [options] DIRECTORY
```

Aligns and stacks the FITS files in a directory into
`<prefix>-stack.fit` and `<prefix>-stack.png`, with a coverage map in
`<prefix>-coverage.png`.

`--dark FILE ...`
: Dark frames to subtract.

`--flat FILE ...`
: Flat frames to divide by.

`--method {mean,median}`
: Combination method (default: sigma-clipped mean).

`--sigma N`
: Clipping threshold for mean stacking (default: 3).

`--no-normalize`
: Do not equalize the sky background between frames.

`--min-coverage FRACTION`
: Crop to the area covered by this fraction of frames; 0 keeps the full
  field (default: 0).

`--deband AMOUNT`
: Remove horizontal banding, from 0 to 1 (default: 0). Useful on older Canon
  cameras.

`--rgb`
: Keep colour; output files get an `-rgb` prefix.

`--bayer-pattern {RGGB,BGGR,GRBG,GBRG}`
: Bayer layout for older FITS files, with `--rgb`.

`--output-prefix NAME`
: Output basename (default: the directory name).

## locate

```
astrophoto locate --query NAME FILE ...
```

Plate-solves each image and saves a full-frame PNG marking the Solar System
body, as `<image>-locate-<name>.png`.

`--query NAME`
: Solar System body name or designation.

Configuration: `[solve]`, `timezone`.

## crop

```
astrophoto crop --query NAME [--radius PIXELS] FILE ...
```

Plate-solves each image and saves a crop around the Solar System body, labelled
with its name and the time, as `<image>-crop-<name>.png`.

`--query NAME`
: Solar System body name or designation.

`--radius PIXELS`
: Crop radius (default: 100).

Configuration: `[solve]`, `timezone`.

## chart

```
astrophoto chart [--query NAME] [--fov DEG] [--magnitude MAG] [--output FILE] FILE ...
```

Draws a star chart with the outline of each plate-solved image, optionally
marking a target at each image time.

`--query NAME`
: Star, deep-sky object, or Solar System body to mark. A Solar System body
  needs the observer position.

`--fov DEG`
: Chart width in degrees (default: fit the images, at least 30).

`--magnitude MAG`
: Faintest stars to draw, up to 6 (default: 6).

`--output FILE`
: PNG path (default: next to a single image, or
  `<kind>-chart-NNNN.png` in the current directory).

Configuration: `[solve]`, `timezone`.

## Files

| Path | Contents |
|---|---|
| `~/.config/astrophoto/config.toml` | Configuration |
| `~/.local/share/astrophoto` | Indexes and catalogs from `download` |
| `~/.cache/astrophoto` | Cached forecasts and ephemerides |
| `<image>.new` | Saved plate solution of an image, reused by later commands |
