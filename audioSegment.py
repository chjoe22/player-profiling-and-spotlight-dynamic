from pydub import AudioSegment
from datetime import datetime

def cut_audio(input_file, output_file, start_time, end_time):
    """
    Cut parts of an audio file and save it to a new file.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the output audio file.
        start_time (int): Start time in milliseconds.
        end_time (int): End time in milliseconds.
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Cut the audio
    cut_audio = audio[start_time:end_time]

    # Save the cut audio to a new file
    cut_audio.export(output_file, format="wav")

def hhmmss_to_ms(timestamp: str) -> int:
    t = datetime.strptime(timestamp, "%H:%M:%S")
    return (t.hour * 3600 + t.minute * 60 + t.second) * 1000

# Examples
input_file = "audio/episode1.wav"
output_file = "segmented-audio/output.wav"
start_time = hhmmss_to_ms("00:31:59")  # Start at 0 seconds
end_time = hhmmss_to_ms("00:32:00")  # End at 24 seconds

cut_audio(input_file, output_file, start_time, end_time)
