#!/bin/bash

less -R $(ls -lhtr /workspace/logs/*|awk '{print $NF}'|tail -n 1)

