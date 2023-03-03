import os
import time
import wave
import torch
import base64
import whisper
import requests
import datetime
import contextlib
import numpy as np
import pandas as pd
from io import BytesIO
from pytube import YouTube
# from pydub import AudioSegment
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global model_name
    global embedding_model
    #medium, large-v1, large-v2
    model_name = "medium"
    model = whisper.load_model(model_name)
    embedding_model = PretrainedSpeakerEmbedding( 
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))


def get_wav(audio_url):
    print("-----Downloading wav-----")
    audio_file = "local.wav"
    response = requests.get(audio_url)
    with open(audio_file, "wb") as f:
        f.write(response.content)
    print("-----Success downloaded wav-----")
    return audio_file


def get_mp3(audio_url):
    print("-----Downloading mp3:-----")
    audio_file = "local.mp3"
    response = requests.get(audio_url)
    with open(audio_file, "wb") as f:
        f.write(response.content)
    os.system(f'ffmpeg -i local.mp3 local.wav')
    print("-----Success downloaded audio-----")
    # os.system(f'ffmpeg -i "{audio_file}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')
    return audio_file


def get_youtube(video_url):
    yt = YouTube(video_url)
    abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    print("-----Success downloaded video-----")
    print(abs_video_path)
    return abs_video_path


def extract_audio_from_youtube(video_file_path):
    if(video_file_path == None):
        raise ValueError("Error no video input")
    print(video_file_path)
    try:
        # Read and convert youtube video
        _,file_ending = os.path.splitext(f'{video_file_path}')
        print(f'file enging is {file_ending}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        print("-----starting conversion to wav-----")
        os.system(f'ffmpeg -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le local.wav')
    except Exception as e:
        raise RuntimeError("Error converting video to audio")
    return audio_file


def speech_to_text(selected_source_lang, whisper_model, num_speakers):
    audio_file="local.wav"
    model = whisper.load_model(whisper_model)
    time_start = time.time()
    # Get duration of audio file
    with contextlib.closing(wave.open(audio_file,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    print(f"conversion to wav ready, duration of audio file: {duration}")
    # Transcribe audio
    options = dict(language=selected_source_lang, beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)
    result = model.transcribe(audio_file, **transcribe_options)
    segments = result["segments"]
    print("starting whisper done with whisper")
    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')
        # Assign speaker label
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
        # Make output
        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)
        
        time_end = time.time()
        time_diff = time_end - time_start
     
        system_info = f"""-----Processing time: {time_diff:.5} seconds-----"""
        print(system_info)
        return pd.DataFrame(objects)

    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global model_name
    global embedding_model

    # Clean old files
    os.system('rm -rf *.wav')
    os.system('rm -rf *.mp3')

    # Parse out your arguments
    youtube_url = model_inputs.get('youtube_url', "")
    wav_url = model_inputs.get('wav_url', "")
    mp3_url = model_inputs.get('mp3_url', "")
    selected_source_lang = model_inputs.get('language', "en")
    number_speakers = model_inputs.get('num_speakers', 2)

    # Check that at least one url is passed
    if youtube_url == "" and wav_url == "" and mp3_url == "":
        return {'message': "No input provided"}
    
    audio_in = ''
    # Run the model if youtube
    if wav_url == "" and mp3_url == "":
        video_in = get_youtube(youtube_url)
        audio_in = extract_audio_from_youtube(video_in)

    # Run the model if wav url
    if youtube_url == "" and mp3_url == "":
        audio_in = get_wav(wav_url)

    # Run the model if mp3 url
    if youtube_url == "" and wav_url == "":
        audio_in = get_mp3(mp3_url)

    transcription_df = speech_to_text(selected_source_lang, model_name, number_speakers)
    df = transcription_df.copy()
    df['content'] = df['Text'].str.strip()
    df['timestamp'] = df[['Start', 'End']].apply(lambda x: {'start': x['Start'], 'end': x['End']}, axis=1)
    df = df[['Speaker', 'content', 'timestamp']]
    df['speaker'] = df['Speaker']
    df = df[['speaker', 'content', 'timestamp']]
    formatted_data = df.to_json(orient='records', indent=2)
    return formatted_data