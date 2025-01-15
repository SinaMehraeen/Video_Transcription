import whisper
from moviepy.editor import VideoFileClip

def transcribe_video(
    video_path: str, 
    output_audio_path: str = "temp_audio.wav"
) -> str:
    """
    Extracts audio from a given video file and then transcribes the audio using 
    OpenAI Whisper.

    Parameters:
    -----------
    video_path : str
        Path to the video file to be transcribed.
    output_audio_path : str, optional
        Path (including file name) for the temporary audio file to be created. 
        Defaults to 'temp_audio.wav'.

    Returns:
    --------
    str
        The transcribed text from the video’s audio.
    """
    # 1. Load the video file
    #    MoviePy’s VideoFileClip allows us to read the video file and handle its audio track.
    clip = VideoFileClip(video_path)

    # 2. Extract the audio track and save it to a temporary WAV file
    #    The write_audiofile method will handle converting the audio to the desired format.
    clip.audio.write_audiofile(output_audio_path)

    # 3. Load the Whisper model
    #    There are several model sizes available such as 'tiny', 'base', 'small',
    #    'medium', or 'large'. Larger models generally provide better accuracy 
    #    but require more computational resources.
    model = whisper.load_model("base")

    # 4. Transcribe the audio file using the model
    #    The transcribe method returns a dictionary containing various information,
    #    including 'text', 'segments', etc. We only need the 'text' field here.
    transcription_result = model.transcribe(output_audio_path)

    # 5. Return the transcribed text portion from the dictionary
    return transcription_result["text"]


if __name__ == "__main__":
    # Replace with your actual video file path
    input_video_file = "example_video.mp4"

    # Call the transcription function
    transcribed_text = transcribe_video(input_video_file)

    # Print the transcribed text on the console
    print("Transcribed Text:\n", transcribed_text)
