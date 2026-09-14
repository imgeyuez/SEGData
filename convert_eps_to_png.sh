#!/bin/bash

# This script converts all EPS files in the current directory to PNG format
# using the 'convert' command from ImageMagick. The output PNG files will have
# the same base name as the input EPS files.

TARGET_DIR="VeraAufgaben"

for eps_file in "$TARGET_DIR"/**/*.eps; do
    if [[ -f "$eps_file" ]]; then
        png_file="${eps_file%.eps}.png"
        convert -density 300 "$eps_file" -trim +repage  -background white -flatten "$png_file"
        echo "Converted $eps_file to $png_file"
    else
        echo "No EPS files found in the current directory."
    fi
done