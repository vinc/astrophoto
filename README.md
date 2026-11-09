# Astrophoto

A collection of mostly vibe coded astrophoto scripts.

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

Center and focus the camera, capture a sequence of raw pictures, and convert
the CR2 files into FITS files:

```sh
python focus.py
python center.py \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
sh capture.sh
python convert.py 2026*/*.CR2
```

Stack the FITS files in a dir:

```sh
python stack.py 20261221-2149
```

Stacking matches each frame's background to the middle frame, then takes a
sigma-clipped mean. The stack FITS includes `COVERAGE` (available frames per
pixel) and `ACCEPTED` (frames remaining after clipping) extensions. A
`-coverage.png` preview shows the input coverage, with brighter areas covered
by more frames. Clipping is disabled where fewer than five frames overlap.

Crop to the largest rectangle covered by every aligned frame:

```sh
python stack.py 20261221-2149 --min-coverage 1
```

The default keeps the full field; `--min-coverage 0.8` requires at least 80%
of the aligned frames. Cropping uses input coverage before outlier rejection.
This normalization matches background offsets, not gradients or transparency.

Compare with the previous median method without overwriting the default stack:

```sh
python stack.py 20261221-2149 \
    --method median --no-normalize --output-prefix median
```

Use `--sigma` to adjust the clipping threshold (default: 3). Strong clipping
can reject moving targets or stellar detail when alignment or star shapes vary;
compare the `COVERAGE` and `ACCEPTED` extensions when evaluating a stack.

Mark an object in the full image:

```sh
python locate.py 2026*/*-stack.fit \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
```

Images are plate-solved if needed, with solutions saved as `.new` files for
reuse.

Crop a major or minor body from the image:

```sh
python crop.py 2026*/*.fit \
    --latitude 51.18 \
    --longitude -1.83 \
    --elevation 0.10 \
    --query vesta
```
