#!/bin/bash
sinteractive -A kildisha -p training --nodes=1 --gpus-per-node=1 --cpus-per-gpu=1 --mem=15G --time=24:00:00 --qos=training
