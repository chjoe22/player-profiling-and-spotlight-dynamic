@echo off
setlocal enabledelayedexpansion

echo Starting Sequential Python Pipeline...

echo Running speaker_time...
cd script/engagement_metrix
python speaker_time.py || goto :error
python speaker_time_analyzer.py || goto :error
python speaker_stat_plotter_seaborn.py || goto :error
cd ../..

echo Running context_extraction...
cd script/context
python context_extraction.py || goto :error
python combat.py || goto :error
cd ../..

echo Running emotion_combination...
cd script/combination
python emotion_combination.py || goto :error
cd ../..

echo Running profile_generator...
cd script/profiling
python profile_generator.py || goto :error
cd ../..

echo Successfully ran all scripts
pause
exit /b 0

:error
echo.
echo Error
echo Script failed
popd
pause
exit /b 0