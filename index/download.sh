#!/bin/sh

CDPATH= cd -- "$(dirname -- "$0")" || exit 1

for i in $(seq -w 07 19); do
  wget -nc https://data.astrometry.net/4100/index-41$i.fits
done

# Star and constellation data for chart.py, from d3-celestial (BSD-3-Clause)
CELESTIAL=https://raw.githubusercontent.com/ofrohn/d3-celestial/7e720a3de062059d4c5400a379146a601d9010e0/data
for name in stars.6.json constellations.json constellations.lines.json; do
  wget -nc "$CELESTIAL/$name"
done
