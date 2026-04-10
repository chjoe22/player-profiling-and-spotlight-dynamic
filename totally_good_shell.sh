#!/bin/bash

# Stop the script if any command fails
set -e

echo "Starting Sequential Python Pipeline..."

echo "Running Speaker_time..."
cd script/engagement_metrix
python3 speaker_time.py
python3 speaker_time_analyzer.py
python3 speaker_stat_plotter_seaborn.py
cd ../..

echo "Running Context_extraction..."
cd script/context
python3 context_extraction.py
python3 combat.py
cd ../..

echo "Running emotion_combination..."
cd script/combination
python3 emotion_combination.py
cd ../..

echo "Running profile_generator..."
cd script/profiling
python3 profile_generator.py
cd ../..

echo "Successfully ran all scripts"