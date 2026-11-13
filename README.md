# Astrophoto

A personal collection of mostly vibe coded astrophotography scripts.

## Setup

```sh
pip install -r requirements.txt
sh index/download.sh
```

## Usage

### Before a session

Prepare a session by checking the weather forecast for the night:

```
$ python weather.py --latitude 51.18 --longitude -1.83 --elevation 0.10
Date         Thu 12  Fri 13
Time (h)     22  23   0   1   2   3   4   5   6   7
---------------------------------------------------
Cloud (%)     0   0   0   0   0   0   0   0   0   0
High (%)      0   0   0   0   0   0   0   0   0   0
Mid (%)       0   0   0   0   0   0   0   0   0   0
Low (%)       0   0   0   0   0   0   0   0   0   0
---------------------------------------------------
Wind (m/s)    4   4   4   4   3   3   2   2   2   2
Gust (m/s)    6   6   6   6   6   5   4   3   3   3
---------------------------------------------------
Temp (°C)     1   1   1   1   1   1   0   0   0  -1
Dew (°C)     -1  -2  -2  -2  -2  -2  -2  -2  -3  -3
---------------------------------------------------
Rain (mm)     0   0   0   0   0   0   0   0   0   0
```

Check what is currently visible:

```
$ python visible.py --latitude 51.18 --longitude -1.83 --elevation 0.10
2026-11-12 22:53 CET, Sun at -49 degrees (dark sky)
Orbital elements downloaded 2026-11-12 (0 days ago)
Bodies above 10 degrees and brighter than magnitude 10:

  Mag   Alt       Az      RA     Dec  Elong  Type      Name
  0.5    39  192 SSW  00h34m    +0.8    138  planet    Saturn
  5.6    44  115 ESE  04h07m   +20.8    166  planet    Uranus
  6.9    34  182 S    01h05m    -4.5    142  asteroid  4 Vesta (A807 FA)
  7.7    36  200 SSW  00h07m    -0.7    131  planet    Neptune
  8.0    14   69 ENE  07h42m   +24.6    117  asteroid  1 Ceres (A801 AA)
  8.7    18  184 S    00h56m   -20.9    129  asteroid  2 Pallas (A802 FA)
  9.8    45  207 SSW  23h57m    +9.1    132  asteroid  192 Nausikaa (A879 DA)
```

Use `--time <datetime>` to choose the date and time of the planned session.

### During a session

Focus and center the camera, capture a sequence of raw pictures (light, dark,
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

### After a session

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
