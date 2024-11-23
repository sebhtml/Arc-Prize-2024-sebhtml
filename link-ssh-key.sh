#!/bin/bash

cp /workspace/id_ed25519.pub ~/.ssh/id_ed25519.pub
cp /workspace/id_ed25519 ~/.ssh/id_ed25519

chmod 0600 ~/.ssh/id_ed25519.pub
chmod 0600 ~/.ssh/id_ed25519
