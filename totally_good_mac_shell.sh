#!/bin/bash

set -e

cd "$(dirname "$0")"

pip install -r requirements.txt

echo "Starting Pipeline on macOS..."

echo "Running Speaker Metrics..."
pushd script/engagement_metrix > /dev/null
python3 speaker_time.py
python3 speaker_time_analyzer.py
python3 speaker_stat_plotter_seaborn.py
popd > /dev/null

echo "Running Context..."
pushd script/context > /dev/null
python3 context_extraction.py
python3 combat.py
popd > /dev/null

echo "Running Combination..."
pushd script/combination > /dev/null
python3 emotion_combination.py
popd > /dev/null

echo "Running Profiling..."
pushd script/profiling > /dev/null
python3 profile_generator.py
popd > /dev/null

echo "Successfully ran all scripts"