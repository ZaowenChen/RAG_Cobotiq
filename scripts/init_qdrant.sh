#!/usr/bin/env bash
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
echo "Creating Qdrant collection 'robot_elements'..."
curl -s -X PUT "${QDRANT_URL}/collections/robot_elements" \
  -H "Content-Type: application/json" \
  --data-binary @configs/qdrant.json >/dev/null || true
echo "Done."
