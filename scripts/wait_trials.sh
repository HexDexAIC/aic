#!/usr/bin/env bash
# Wait for the running aic_eval container's trials to progress, emitting
# milestones to stdout. Polls every 60 sec, exits when all trials done
# or container stops.

PATTERN='Trial 2/3|Trial 3/3|Final Score|Eval Summary|TasksComplete|All trials complete'
PREV=""
while docker ps --filter name=aic_eval_running --format '{{.Names}}' | grep -q aic_eval_running; do
    CUR=$(docker logs aic_eval_running 2>&1 | grep -E "$PATTERN" | tail -3)
    if [ "$CUR" != "$PREV" ]; then
        DIFF=$(comm -13 <(echo "$PREV") <(echo "$CUR"))
        echo "MILESTONE:"
        echo "$DIFF"
        PREV="$CUR"
        if echo "$CUR" | grep -qE 'Final Score|Eval Summary|All trials complete'; then
            echo ALL_DONE
            break
        fi
    fi
    sleep 90
done
echo CONTAINER_STOPPED_OR_DONE
docker logs aic_eval_running 2>&1 | grep -iE 'trial|score|tier|complete' | tail -25
