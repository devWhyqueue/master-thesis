#!/bin/bash
set -euo pipefail

container="${EXPERIMENT_CONTAINER:-}"
if [ "$container" = "" ] && [ -f environment.sif ]; then
  container="$PWD/environment.sif"
fi

if [ "$container" != "" ]; then
  if [ "${EXPERIMENT_USE_GPU:-0}" = "1" ]; then
    apptainer run --nv -B /home/space:/home/space:ro "$container" python3 "$@"
  else
    apptainer run -B /home/space:/home/space:ro "$container" python3 "$@"
  fi
else
  python3 "$@"
fi
