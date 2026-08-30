#!/bin/bash
# find URLs here: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

if [ -z "$1" ]; then
  echo "Usage: $0 <remote_path> [destination]"
  echo "Example: $0 data/tuh_eeg/TEST datasets/"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DESTINATION="${2:-$PROJECT_ROOT/datasets/}"
mkdir -p "$DESTINATION"

rsync -auvxL -e "ssh -i ~/.ssh/id_ed25519" "nedc-tuh-eeg@www.isip.piconepress.com:$1" "$DESTINATION"
