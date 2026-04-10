import os
import subprocess
import re
from tqdm import tqdm

def convert_episodes(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp3")]
    files.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
    
    if not files:
        print(f"No .mp3 files found in {input_dir}")
        return

    print(f"Found {len(files)} files to convert.")

    for filename in tqdm(files, desc="Converting Audio", unit="file"):
        input_path = os.path.join(input_dir, filename)
        
        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            continue

        command = [
            'ffmpeg',
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            output_path,
            '-y',
            '-loglevel', 'error'
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {filename}: {e}")
        except FileNotFoundError:
            print("Error: FFmpeg not found on your system.")
            return

    print(f"\nConversion complete. Files are in: {output_dir}")

if __name__ == "__main__":
    INPUT_FOLDER = "../../audio/"
    OUTPUT_FOLDER = "../../segmented-audio"
    
    convert_episodes(INPUT_FOLDER, OUTPUT_FOLDER)