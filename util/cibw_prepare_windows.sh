#!/bin/sh
set -e
cd "${1:-.}"
rm -rf .openblas
mkdir .openblas
curl -sSL -o .openblas/openblas.zip https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34-x64.zip
python -m zipfile -e .openblas/openblas.zip .openblas
rm .openblas/openblas.zip
cp .openblas/lib/libopenblas.lib .openblas/lib/openblas.lib
