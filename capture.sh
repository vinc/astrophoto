#!/bin/sh

dir=$(date +%Y%m%d-%H%M)
shutterspeed=5

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [dark|flat]" >&2
  exit 1
fi

case "${1-}" in
  "") dir="$dir-light" ;;
  dark) dir="$dir-dark" ;;
  flat)
    dir="$dir-flat"
    shutterspeed=45
    ;;
  *)
    echo "Usage: $0 [dark|flat]" >&2
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
  --capture-image-and-download \
  --interval 15 \
  --frames 20 \
  --filename "$dir/%Y%m%d-%H%M%S.%C"
