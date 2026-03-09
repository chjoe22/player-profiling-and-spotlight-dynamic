from pydub import AudioSegment

def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds using PyDub.

    Args:
        file_path (str): Path to the audio file

    Returns:
        float or None: Duration in seconds, or None if an error occurs
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)

        # Get duration in milliseconds and convert to seconds
        duration_seconds = len(audio) / 1000.0
        return duration_seconds
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

input_file = "audio/episode1.wav"

duration = get_audio_duration(input_file)
if duration is not None:
    print(f"Duration: {duration} seconds")
else:
    print('Failed to retrieve duration.')