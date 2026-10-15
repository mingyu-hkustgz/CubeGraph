#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
${CXX:-g++} -std=c++17 -O2 -march=core-avx2 -fopenmp -Iinclude -Ithird_party tests/test_bottom_up.cpp -o "$out/test"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} "$out/test" "$out/data"
