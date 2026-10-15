#!/usr/bin/env bash
set -e

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

DATA_FILE="notebook/household_power_consumption.txt"

echo "Checking electricity dataset..."
if [ ! -f "$DATA_FILE" ] || [ "$(wc -c < "$DATA_FILE")" -lt 1000000 ]; then
    mkdir -p notebook
    echo "Downloading the real dataset from UCI..."
    curl -fL --retry 3 
      "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip" 
      -o /tmp/electricity_dataset.zip
    python -c "import zipfile; zipfile.ZipFile('/tmp/electricity_dataset.zip').extractall('/tmp/electricity_dataset')"
    find /tmp/electricity_dataset -name "household_power_consumption.txt" -exec cp {} "$DATA_FILE" \;
fi

echo "Dataset size:"
ls -lh "$DATA_FILE"

if [ ! -s "$DATA_FILE" ]; then
    echo "ERROR: Dataset download failed."
    exit 1
fi

echo "Build setup completed."