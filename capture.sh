#!/bin/sh

dir=$(date +%Y%m%d-%H%M)
shutterspeed=5
frames=20

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [light|dark|flat]" >&2
  exit 1
fi

case "${1-}" in
  ""|light) dir="$dir-light" ;;
  dark) dir="$dir-dark" ;;
  flat)
    dir="$dir-flat"
    shutterspeed=50
    ;;
  *)
    echo "Usage: $0 [light|dark|flat]" >&2
    exit 1
    ;;
esac

mkdir -p "$dir" || exit 1
echo "Capturing pictures in $dir ..."

gphoto2 \
  --set-config syncdatetime=1 \
  --set-config imageformat=RAW \
  --set-config iso=800 \
  --set-config aperture=4.0 \
  --set-config "shutterspeed=$shutterspeed" \
  --capture-image \
  --interval 1 \
  --frames "$frames" || exit 1

echo "Downloading pictures to $dir ..."
gphoto2 \
  --folder / --recurse \
  --filename "$dir/%Y%m%d-%H%M%S.%C" \
  --get-all-files || exit 1

echo "Deleting downloaded pictures from the camera ..."
gphoto2 --folder /DCIM/100CANON --delete-file "1..$frames"
