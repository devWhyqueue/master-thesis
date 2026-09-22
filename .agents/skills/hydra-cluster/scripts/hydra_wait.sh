#!/usr/bin/env bash
# Block until the given SLURM jobs leave the queue, then report how they ended.
#
# Run from local, streamed over SSH (nothing to scp, nothing left in /tmp):
#   ssh hydra 'bash -l -s -- <jobid> [<jobid>...]' < .claude/skills/hydra-cluster/scripts/hydra_wait.sh
#
# While waiting, cancels jobs stuck on DependencyNeverSatisfied (an afterok
# upstream failed, so they would otherwise pend forever). At the end prints
# sacct per task and the stderr tail of up to 3 failed tasks.
# Exit 0 when every task COMPLETED, 1 otherwise.
set -u
[ $# -gt 0 ] || { echo "usage: hydra_wait.sh <jobid>..." >&2; exit 2; }
ids=$(IFS=,; echo "$*")

while squeue -h -j "$ids" 2>/dev/null | grep -q .; do
  dead=$(squeue -h -j "$ids" -t PD -o '%i %r' | awk '$2 == "DependencyNeverSatisfied" {print $1}')
  if [ -n "$dead" ]; then
    echo "scancel (DependencyNeverSatisfied): $dead"
    scancel $dead
  fi
  sleep 30
done

sacct -j "$ids" -X -o JobID%16,JobName%28,State%24,ExitCode,Elapsed

status=0
shown=0
while IFS='|' read -r jid raw name state err; do
  [ "$state" = COMPLETED ] && continue
  status=1
  [ $shown -ge 3 ] && continue
  case $jid in *_*) arr=${jid%%_*}; task=${jid#*_} ;; *) arr=$jid; task=0 ;; esac
  err=${err//%x/$name}; err=${err//%A/$arr}; err=${err//%a/$task}; err=${err//%j/$raw}
  if [ -f "$err" ]; then
    echo "=== $jid $state: $err"
    tail -n 20 "$err"
    shown=$((shown + 1))
  fi
done < <(sacct -j "$ids" -X -n -P -o JobID,JobIDRaw,JobName,State,StdErr)
exit $status
