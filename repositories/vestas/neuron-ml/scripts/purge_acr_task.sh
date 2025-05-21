#!/bin/bash

# Create a scheduled task to purge tags in a ACR repository for images that have not been updated in the last 30 days
# This is only used for the development environment. For prod, we want to keep all images

# This follows the docs here: https://learn.microsoft.com/da-dk/azure/container-registry/container-registry-auto-purge
# The purge command is a bit tricky to get right, so it's recommended to test it with `--dry-run` first
REGISTRY=lacneurondev.azurecr.io
FILTER_DAYS=14d

PURGE_CMD="acr purge \
    --filter 'prediction-api:.*' \
    --filter 'training-api:.*' \
    --ago $FILTER_DAYS --untagged"
# add `--dry-run` to test the command without deleting anything

echo "Purge command: $PURGE_CMD"
# Run the task directly
# az acr run \
#   --cmd "$PURGE_CMD" \
#   --registry $REGISTRY \
#   /dev/null

# Create a task to run the purge command every week (Sunday, 1am)
az acr task create --name weeklyPurgeTask \
  --cmd "$PURGE_CMD" \
  --schedule "0 1 * * 0" \
  --registry $REGISTRY \
  --context /dev/null