#!/bin/sh

for i in $(seq -w 07 19); do
  wget https://data.astrometry.net/4100/index-411$i.fits
done
