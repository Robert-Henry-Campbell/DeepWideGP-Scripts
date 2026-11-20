#!/bin/bash
# Restarts the queue's scheduler if it dies (unless the stop file exists); checks every 2 minutes. Detached.
Q=/mnt/sdg/robert/deepwidegp/runs/tweaks/queue
while :; do
    if [ ! -e $Q/stop ] && ! ps -eo args | grep -q "[p]ython3 $Q/scheduler.py\|[p]ython3 scheduler.py"; then
        echo "$(date -u +%FT%TZ) watchdog: scheduler not running; restarted" >> $Q/queue.log
        (cd $Q && setsid nohup python3 scheduler.py > /dev/null 2>&1 < /dev/null &)
    fi
    sleep 120
done
