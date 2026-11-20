#!/usr/bin/env python3
"""A small job queue for the overnight tweak runs (20 Nov 2026). Jobs are bash scripts in jobs/, run in name order
as GPU slots free up; each takes one argument, the GPU index (or "" for a CPU job). Headers a job may carry:
    # slots: N        GPU slots it needs (0: CPU only; a cached frozen-table run 1; a streamed run 3 = a whole GPU)
    # after: GLOB     start only once every job matching GLOB (by name, without .sh) is in done/
gpu_slots.txt lists "gpu capacity" per line and may be edited while the queue runs. A finished job moves to done/ or
failed/ with its exit code in logs/<job>.rc; its output is logs/<job>.log. The queue stops when stop exists.
Detached:  setsid nohup python3 scheduler.py > /dev/null 2>&1 < /dev/null &
"""
import fnmatch
import os
import re
import subprocess
import sys
import time

Q = os.path.dirname(os.path.abspath(__file__))
JOBS, RUNNING, DONE, FAILED, LOGS = (os.path.join(Q, d) for d in ("jobs", "running", "done", "failed", "logs"))


def say(msg):
    with open(os.path.join(Q, "queue.log"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def header(path, key, default):
    with open(path) as f:
        for line in f:
            m = re.match(rf"#\s*{key}:\s*(.+?)\s*$", line)
            if m:
                return m.group(1)
    return default


def capacity():
    cap = {}
    try:
        with open(os.path.join(Q, "gpu_slots.txt")) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    cap[parts[0]] = int(parts[1])
    except OSError:
        pass
    return cap


def pending():
    return sorted(n[:-3] for n in os.listdir(JOBS) if n.endswith(".sh"))


class Adopted:
    """A job started by an earlier scheduler, known by its pid: alive while kill -0 works; its exit code read from the
    last "exit N" line of its log when it is gone (1 if the log has none)."""

    def __init__(self, pid, log):
        self.pid, self.log = int(pid), log

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            pass
        try:
            with open(self.log) as f:
                last = [l for l in f.read().splitlines() if l.startswith("exit ")]
            return int(last[-1].split()[1]) if last else 1
        except (OSError, ValueError, IndexError):
            return 1


def adopt(running):
    """Jobs left in running/ by an earlier scheduler, with their pid files."""
    for n in sorted(os.listdir(RUNNING)):
        if not n.endswith(".sh"):
            continue
        name = n[:-3]
        try:
            with open(os.path.join(RUNNING, name + ".pid")) as f:
                pid, gpu, slots = f.read().split()
            gpu = "" if gpu == "-" else gpu
        except (OSError, ValueError):
            say(f"{name}: in running/ without a pid file; moved to failed/")
            os.replace(os.path.join(RUNNING, n), os.path.join(FAILED, n))
            continue
        running[name] = (Adopted(pid, os.path.join(LOGS, name + ".log")), gpu, int(slots))
        say(f"{name}: adopted (pid {pid}, GPU {gpu or 'cpu'}, slots {slots})")


def main():
    running = {}            # name -> (Popen or Adopted, gpu, slots)
    say(f"scheduler started, pid {os.getpid()}")
    adopt(running)
    while True:
        if os.path.exists(os.path.join(Q, "stop")):
            say("stop file seen; the scheduler exits (running jobs continue)")
            return
        for name, (p, gpu, slots) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            with open(os.path.join(LOGS, name + ".rc"), "w") as f:
                f.write(f"{rc}\n")
            os.replace(os.path.join(RUNNING, name + ".sh"), os.path.join(DONE if rc == 0 else FAILED, name + ".sh"))
            pidfile = os.path.join(RUNNING, name + ".pid")
            if os.path.exists(pidfile):
                os.remove(pidfile)
            say(f"{name}: exit {rc} on GPU {gpu or 'cpu'}")
            del running[name]
        cap = capacity()
        used = {g: 0 for g in cap}
        for _, gpu, slots in running.values():
            if gpu in used:
                used[gpu] += slots
        done = {n[:-3] for n in os.listdir(DONE)}
        failed = {n[:-3] for n in os.listdir(FAILED)}
        held = set()        # a GPU kept draining for a ready job that needs more slots than any GPU has free (no starvation)
        for name in pending():
            path = os.path.join(JOBS, name + ".sh")
            slots = int(header(path, "slots", "1"))
            after = header(path, "after", None)
            if after:
                wanted = [n for n in done | failed | set(running) | set(pending()) if fnmatch.fnmatch(n, after) and n != name]
                if not wanted or any(n not in done for n in wanted):
                    if any(n in failed for n in wanted):
                        os.replace(path, os.path.join(FAILED, name + ".sh"))
                        say(f"{name}: not run, a job it waits for failed")
                    continue
            gpu = ""
            if slots > 0:
                free = [g for g in sorted(cap, key=lambda g: (used[g], g)) if cap[g] - used[g] >= slots and g not in held]
                if not free:
                    if slots > 1 and cap:
                        held.add(max((g for g in cap if cap[g] >= slots), key=lambda g: cap[g] - used[g], default=None))
                        held.discard(None)
                    continue
                gpu = free[0]
                used[gpu] += slots
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=os.environ.get("JOB_THREADS", "3"),
                       MKL_NUM_THREADS=os.environ.get("JOB_THREADS", "3"))
            os.replace(path, os.path.join(RUNNING, name + ".sh"))
            log = open(os.path.join(LOGS, name + ".log"), "a")
            log.write(f"=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} starting on GPU {gpu or 'cpu'}\n")
            log.flush()
            p = subprocess.Popen(["bash", os.path.join(RUNNING, name + ".sh"), gpu], env=env, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            running[name] = (p, gpu, slots)
            with open(os.path.join(RUNNING, name + ".pid"), "w") as f:
                f.write(f"{p.pid} {gpu or '-'} {slots}\n")
            say(f"{name}: started on GPU {gpu or 'cpu'} (slots {slots}), pid {p.pid}")
        time.sleep(20)


if __name__ == "__main__":
    main()
