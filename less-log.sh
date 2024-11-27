#!/bin/bash

less $(ls -lhtr /workspace/logs/*|awk '{print $NF}'|tail -n 1)

