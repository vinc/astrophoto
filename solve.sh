#!/bin/sh

failed=0

for file in "$@"; do
  solve-field \
    --fits-image \
    --backend-config astrometry.cfg \
    --scale-units degwidth \
    --scale-low 16 \
    --scale-high 18 \
    --downsample 2 \
    --no-plots \
    --overwrite \
    "$file" || {
      echo "Failed to solve $file" >&2
      failed=$((failed + 1))
    }
done

if [ "$failed" -gt 0 ]; then
  echo "$failed of $# files failed to solve" >&2
  exit 1
fi
