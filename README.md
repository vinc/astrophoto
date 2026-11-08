# Astrophoto

A collection of vibe coded astrophoto scripts

## Setup

```sh
pip install -r requirements.txt
sh index/download.sh
```

## Usage

Center and focus the camera, capture a sequence of raw pictures, and convert
the CR2 files into FITS files:

```sh
python center.py 2026*/*.new \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
python focus.py
sh capture.sh
python convert.py 2026*/*.CR2
```

Stack the FITS files in a dir:

```sh
python stack.py 20261221-2149
```

Mark an object in the full image:

```sh
python locate.py 2026*/*-stack.fit \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
```

Images are plate-solved if needed.

Crop a major or minor body from the image:

```sh
sh solve.sh 2026*/*.fit
python crop.py 2026*/*.new \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
```
