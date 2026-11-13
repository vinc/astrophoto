#!/bin/sh

dir="${XDG_DATA_HOME:-$HOME/.local/share}/astrophoto"
mkdir -p "$dir" && cd "$dir" || exit 1
echo "Downloading data to $dir ..."

for i in $(seq -w 07 19); do
  wget -nc https://data.astrometry.net/4100/index-41$i.fits || exit 1
done

# Star and constellation data for chart, from d3-celestial (BSD-3-Clause)
CELESTIAL=https://raw.githubusercontent.com/ofrohn/d3-celestial/7e720a3de062059d4c5400a379146a601d9010e0/data
for name in stars.6.json constellations.json constellations.lines.json; do
  wget -nc "$CELESTIAL/$name" || exit 1
done
