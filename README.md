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
