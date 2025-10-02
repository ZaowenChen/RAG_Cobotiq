#!/usr/bin/env bash
set -euo pipefail

MEILI_URL="${MEILI_URL:-http://localhost:7700}"
MEILI_MASTER_KEY="${MEILI_MASTER_KEY:-dev_master_key}"

echo "Creating Meilisearch index 'robot_elements'..."
curl -s -X POST "${MEILI_URL}/indexes" \
  -H "X-Meili-API-Key: ${MEILI_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @configs/meilisearch.create.json >/dev/null || true

echo "Applying Meilisearch settings..."
curl -s -X PATCH "${MEILI_URL}/indexes/robot_elements/settings" \
  -H "X-Meili-API-Key: ${MEILI_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @configs/meilisearch.settings.json >/dev/null

echo "Done."
