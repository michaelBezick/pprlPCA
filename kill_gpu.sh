#!/bin/bash

nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | sort -u \
  | xargs -r -n1 kill -9
