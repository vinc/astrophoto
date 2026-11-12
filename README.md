# Astrophoto

A personal collection of mostly vibe coded astrophotography scripts.

## Setup

```sh
pip install -r requirements.txt
sh index/download.sh
```

## Usage

Prepare a session by listing what is visible:

```sh
python visible.py \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --time 2026-12-21T21:49
```

Center and focus the camera, capture a sequence of raw pictures (light, dark,
and flat frames), and convert the CR2 files into FITS files:

```sh
python focus.py
python center.py \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
sh capture.sh light
sh capture.sh dark
sh capture.sh flat
python convert.py 2026*/*.CR2
```

Stack the FITS files in a directory with optional dark and flat frames:

```sh
python stack.py 20261221-2149-light \
    --dark 20261221-*-dark/*.fit \
    --flat 20261221-*-flat/*.fit
```

Add `--deband 1.0` to remove horizontal banding on older Canon cameras,
and `--rgb` to render an RGB image.

Mark an object in the full image:

```sh
python locate.py 2026*/*-stack.fit \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
```

Crop a major or minor body from one or more images:

```sh
python crop.py 2026*/*.fit \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
```

Draw a star chart around one or more images with an optional filter query:

```sh
python chart.py 2026*/*-stack.fit \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
```

Images are plate-solved if needed, with solutions saved as `.new` files for
reuse.
