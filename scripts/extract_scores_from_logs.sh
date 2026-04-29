#!/usr/bin/env bash
# Pull trial scores from the live container's logs (since scoring.yaml only
# appears after the final trial completes).

CONTAINER=${1:-aic_eval_running}
echo "=== Trial scores from $CONTAINER ==="
docker logs "$CONTAINER" 2>&1 \
    | grep -E "Trial '[a-z_0-9]+' completed successfully|total score is:" \
    | sed -E 's/^.*aic_engine\]: //; s/\x1b\[[0-9;]*m//g'
