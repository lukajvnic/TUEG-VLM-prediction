# shorten pending seed-2 jobs to 1.5x the original arm's measured run time plus an hour (every job of the original,
# resumed ones included), never above the limit run.py gave them: short limits backfill sooner and stay clear of the
# Oct 27 outage. A job that still runs out is resumed by auto-resubmit.sh
#   python shorten_limits.py [--apply]
import collections
import subprocess
import sys

PREFIX = "eeg-vlm-vision-freeze-"


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def seconds(limit):
    days, _, clock = limit.rpartition("-")
    parts = [int(x) for x in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


used = collections.Counter()
for line in run("sacct", "-u", "luka", "-S", "2026-10-15", "-X", "-n", "-P", "-o", "JobName,State,ElapsedRaw").splitlines():
    name, state, elapsed = line.split("|")
    if name.startswith(PREFIX) and state.split()[0] in ("COMPLETED", "TIMEOUT"):
        used[name[len(PREFIX):]] += int(elapsed)

for line in run("squeue", "-u", "luka", "-h", "-t", "PD", "-o", "%i|%j|%l").splitlines():
    job, name, limit = line.split("|")
    arm = name[len(PREFIX):]
    if not arm.endswith("-s2") or arm[:-3] not in used:
        continue
    target = int(1.5 * used[arm[:-3]] + 3600)
    if target < seconds(limit):
        minutes = -(-target // 60)
        print(f"{arm}: {limit} -> {minutes // 1440}-{minutes % 1440 // 60:02d}:{minutes % 60:02d}:00 "
              f"(original used {used[arm[:-3]] / 3600:.1f} h)")
        if "--apply" in sys.argv:
            run("scontrol", "update", f"JobId={job}", f"TimeLimit={minutes}")
