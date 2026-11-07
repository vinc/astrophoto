#!/bin/sh

for file in "$@"; do
  solve-field \
    --backend-config astrometry.cfg \
    --scale-units degwidth \
    --scale-low 16 \
    --scale-high 18 \
    --downsample 2 \
    --no-plots \
    --overwrite \
    "$file" || exit 1
done
