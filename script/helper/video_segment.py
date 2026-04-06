import subprocess
import os
from pathlib import Path

def split_video(input_video_path, output_base_folder, segment_time=600):
    """
    Splits a single video file into segments.
    """
    episode_name = input_video_path.stem

    episode_output_dir = output_base_folder / episode_name
    episode_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing: {episode_name} ---")

    output_pattern = str(episode_output_dir / f"{episode_name}_%03d.mp4")

    command = [
        "ffmpeg",
        "-i", str(input_video_path),
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(segment_time),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern
    ]

    subprocess.run(command)

def process_all_episodes(input_dir, output_dir):
    video_path = Path(input_dir)
    output_path = Path(output_dir)

    episodes = list(video_path.glob("episode*.mp4"))

    if not episodes:
        print(f"No episodes found in {video_path.absolute()}")
        return

    for ep_file in episodes:
        split_video(ep_file, output_path)

if __name__ == "__main__":
    INPUT_FOLDER = "../../video/"
    OUTPUT_FOLDER = "../../segmented-video/"

    process_all_episodes(INPUT_FOLDER, OUTPUT_FOLDER)