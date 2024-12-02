#!/bin/bash

cd /workspace/Arc-Prize-2024-sebhtml

source activate-venv.sh

log_file=$(date --iso-8601=seconds).log
(time python src/main.py) &> /workspace/logs/$log_file
