#!/bin/bash
set -euo pipefail

if [ "${EXPERIMENT_CONTAINER:-}" != "" ]; then
  apptainer run -B /home/space:/home/space:ro "$EXPERIMENT_CONTAINER" python3 "$@"
else
  python3 "$@"
fi

