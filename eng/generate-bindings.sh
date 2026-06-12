#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_PATH="${REPO_ROOT}/tools/JYPPX.BindingGenerator/JYPPX.BindingGenerator.csproj"
dotnet build "${PROJECT_PATH}" -c Debug
dotnet exec "${REPO_ROOT}/tools/JYPPX.BindingGenerator/bin/Debug/net8.0/JYPPX.BindingGenerator.dll" --repo-root "${REPO_ROOT}"
