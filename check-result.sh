#!/bin/bash

grep Result $(ls -lhtr /workspace/logs/*|awk '{print $NF}'|tail -n 1)
