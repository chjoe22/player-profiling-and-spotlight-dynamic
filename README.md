## How to run
### Initialize Environment
macOS/Linux
```bash
python3 -m venv .venv`
```
Windows
```bash
python3 -m venv .venv`
```

### Activate Environment
Bash
```bash
source .venv/bin/activate
```

### Installation of Libraries
To install required libraries
```bash
pip install -r requirements.txt
```

### Running Scripts
To run the project
```bash
python "file-name.py"
```


## Disclaimer
You most likely need to download `ffmpeg` in order to use and run the python scripts.

Futhermore, run \
 `ffmpeg -i episode1.mp3 -acodec pcm_s16le -ac 1 -ar 16000 episode1.wav`

 This changes the format from MP3 to WAV and 16 kHz


# Project Execution
> Complete the  *How to run* before running any of the following scripts.

## Audio
Running the Audio pipeline requires 2 components.

1. Audio File
2. Corresponding Transcript

The audio file is most likely *.mp3* and we would like the file to be *.wav* as it is easier to read (i believe). Running the previous *ffmpeg* command should suffice.

Also, change the `episode_number` inside `audio_pipeline.py` to reflect the chosen episode, and also the transcript name.

Once the audio file and the corresponding transcript file are in the correct folders, simply run the following command inside the `~/script/pipeline` 

```bash
python3 audio_pipeline.py
``` 

This would there after output 2 files;

1. Emotion results
2. Overlap results

These can be found inside `~/results/audio`.

## Video
Running the Video pipeline requires a few steps before `video_pipeline.py` can be run.

1. Downloading Video into `~/video`.
2. Run `videoSegment.py` inside `~/script/helper`. 
    - use `python3 videoSegment.py`
3. Ensure `cast_embeddings.npz` exist.
    - If not, run `cast_detection.py` inside `~/script/helper`
4. Change `episode_number` inside `video_pipeline.py` to reflect episode.
5. Not a necessity, but chose model - can be found in the top of `video_pipeline.py`.

Once these have been addressed, run the following command inside `~/script/pipeline`
```bash
python3 video_pipeline.py
```
The terminal should show a process bar, once that is done results should be available inside `~/results/video`, and be named after the `episode number`.
