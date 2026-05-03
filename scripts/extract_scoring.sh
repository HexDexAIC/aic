#!/usr/bin/env bash
# Pull scoring artifacts out of the running aic_eval container.
# After 3 trials complete, scoring.yaml lives in /root/aic_results/ inside the container.

set -e
CONTAINER=${1:-aic_eval_running}
OUT=${2:-$HOME/aic_results}
mkdir -p "$OUT"
echo "Extracting from $CONTAINER -> $OUT"
docker cp "$CONTAINER":/root/aic_results "$OUT"_tmp 2>&1 || {
    echo "Could not docker cp /root/aic_results — maybe not yet written."
    exit 1
}
# Move contents (docker cp creates aic_results subdir).
shopt -s dotglob
mv "$OUT"_tmp/aic_results/* "$OUT"/ 2>/dev/null || true
rm -rf "$OUT"_tmp
echo "Done. Contents:"
ls -la "$OUT"
echo "---"
if [ -f "$OUT/scoring.yaml" ]; then
    echo "scoring.yaml:"
    cat "$OUT/scoring.yaml"
fi
