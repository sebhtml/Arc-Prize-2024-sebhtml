#!/bin/bash

cd /workspace/Arc-Prize-2024-sebhtml

source venv/bin/activate

log_file=$(date --iso-8601=seconds).log
(time python arc_prize_2024_sebhtml.py) &> /workspace/logs/$log_file

