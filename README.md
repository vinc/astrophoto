# Astrophoto

A collection of vibe coded astrophoto scripts

## Setup

```sh
pip install -r requirements.txt
sh index/download.sh
```

## Usage

Capture a sequence of pictures and convert the CR2 into FITS format:

```sh
sh capture.sh
python convert.py 2026*/*.CR2
```

Crop a major or minor body from the image:

```sh
sh solve.sh 2026*/*.fit
python crop.py 2026*/*.new \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
```

Stack the FITS images in a dir:

```sh
python stack.py 20261221-2149
```

Mark an object in the full image and save a `-locate-<name>.png` beside each input:

```sh
python locate.py 2026*/*-stack.fit \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
```

Images are plate-solved if needed. The title and date match `crop.py`.

Adjust manual focus with repeated short captures and an enlarged star preview:

Preview windows need a GUI toolkit, for example `python -m pip install PyQt6`.

```sh
python focus.py
```

Lower HFR means smaller stars. Keep exposures short enough to avoid star trails.

Measure a saved image without a window, or save an annotated PNG:

```sh
python focus.py image.fit
python focus.py image.fit --save-preview focus.png
```

Centre an object with a full-frame preview and tripod pan/tilt corrections:

```sh
python center.py --query vesta \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --shutterspeed 2 \
    --loop
```

Both helpers also accept a saved RAW or FITS image as a positional argument.
Saved images print results without a window by default. Use `--plot` to open
a window or `--save-preview preview.png` to save an annotated image.
New captures are temporary. Close the window or press Ctrl-C to stop.
Use `--shutterspeed` to change the exposure, and `--ra` with `--dec` in degrees
instead of `--query` for a fixed sky position.
