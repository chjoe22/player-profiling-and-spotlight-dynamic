## How to run
### First
macOS/Linux
```bash
python3 -m venv .venv`
```
Windows
```bash
python3 -m venv .venv`
```

### Second
Bash
```bash
source .venv/bin/activate
```

### Third
To install required libraries
```bash
pip install -r requirements.txt
```

### Last
To run the project
```bash
python "file-name.py"
```


## Disclaimer
You most likely need to download `ffmpeg` in order to use and run the python scripts.

Futhermore, run \
 `ffmpeg -i episode1.mp3 -acodec pcm_s16le -ac 1 -ar 16000 episode1.wav`

 This changes the format from MP3 to WAV and 16 kHz
