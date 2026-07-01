#!/bin/bash
# find URLs here: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

if [ -z "$1" ]; then
  echo "Usage: $0 <remote_path>"
  echo "Example: $0 data/tuh_eeg/TEST"
  exit 1
fi

rsync -auvxL -e "ssh -i ~/.ssh/uw/id_ed25519" "nedc-tuh-eeg@www.isip.piconepress.com:$1" ..
