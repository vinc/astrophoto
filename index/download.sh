#!/bin/sh

CDPATH= cd -- "$(dirname -- "$0")" || exit 1

for i in $(seq -w 07 19); do
  wget -nc https://data.astrometry.net/4100/index-41$i.fits
done
