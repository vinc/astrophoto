#!/bin/sh

dir=$(date +%Y%m%d-%H%M)
mkdir -p "$dir"
echo "Capturing pictures in $dir ..."

gphoto2 \
  --set-config syncdatetime=1 \
  --set-config imageformat=RAW \
  --set-config iso=800 \
  --set-config aperture=4.0 \
  --set-config shutterspeed=5 \
  --capture-image-and-download \
  --interval 15 \
  --frames 20 \
  --filename "%Y%m%d-%H%M%S.%C"
