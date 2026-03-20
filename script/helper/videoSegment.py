import sys
from pathlib import Path
# To find the path in the root folder - too many .parent but they are necessary for it to work
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import subprocess
import os

# Change to reflect the episode number
episode_number = "episode100"

def split_video(input_video, output_folder, segment_time=600):
    """
    Split a video into smaller clips.

    input_video: path to video file
    output_folder: folder where clips will be saved
    segment_time: length of each clip in seconds
    """

    os.makedirs(output_folder, exist_ok=True)

    output_pattern = os.path.join(output_folder, episode_number+"_%03d.mp4")

    command = [
        "ffmpeg",
        "-i", input_video,
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(segment_time),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern
    ]

    subprocess.run(command)


if __name__ == "__main__":
    split_video(
        input_video="../../video/episode100.mp4",
        output_folder="../../segmented-video/",
        segment_time=600  # 600 seconds = 10 minutes
    )