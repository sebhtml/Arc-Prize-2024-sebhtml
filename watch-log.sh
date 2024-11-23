#!/bin/bash

watch tail $(ls -lhtr /workspace/logs/*|awk '{print $NF}'|tail -n 1)

