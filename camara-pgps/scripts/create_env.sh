#!/usr/bin/env bash
# Create or update the camara-pgps conda environment (Linux / macOS).
#
#   ./scripts/create_env.sh

set -euo pipefail
cd "$(dirname "$0")/.."

env_name="camara-pgps"

if conda env list | grep -qE "^\s*${env_name}\s"; then
    echo "Updating existing '${env_name}' environment..."
    conda env update -f environment.yml --prune
else
    echo "Creating '${env_name}' environment..."
    conda env create -f environment.yml
fi

echo
echo "Done. Activate with:  conda activate ${env_name}"
