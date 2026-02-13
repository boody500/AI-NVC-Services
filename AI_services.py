# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import json
import numpy as np
import torch
import os
import yt_dlp
import webvtt
import re
import glob
import subprocess
import requests

# whisper
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5Model
from faster_whisper import WhisperModel
from openAI import adjust_transcript
from langdetect import detect

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

tokenizer = None
model = None
whisper = None
ara_tokenizer = None
ara_model = None
from gradio_client import Client, file

import datetime

#tts_client = Client("bestoai/text-to-video")
#tts_client = Client("MohamedRashad/Multilingual-TTS")
wav2lip_client = Client("pragnakalp/Wav2lip-ZeroGPU")


def load_models():
    global tokenizer, model, whisper, ara_tokenizer, ara_model
    if tokenizer is None or model is None or whisper is None or ara_tokenizer is None or ara_model is None:
        print("Loading T5 model...")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        ara_tokenizer = T5Tokenizer.from_pretrained("UBC-NLP/AraT5-base")
        model = T5Model.from_pretrained("t5-base")
        ara_model = T5Model.from_pretrained("UBC-NLP/AraT5-base")
        print(f"Loading Whisper on {DEVICE}...")
        whisper = WhisperModel("base", device=DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        print("Models loaded successfully!")


def ensure_models_loaded():
    global tokenizer, model, whisper

    if tokenizer is None or model is None or whisper is None:
        load_models()


# ------------------- EMBEDDING -------------------

def get_t5_embedding(text):
    """Get T5 embeddings for a given text."""
    global tokenizer, model, ara_tokenizer, ara_model
    if tokenizer is None or model is None or ara_tokenizer is None or ara_model is None:
        load_models()

    lang = detect(text)

    tokens = ""
    input_text = f"sentence similarity: {text}"

    if lang == "ar":
        tokens = ara_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = ara_model.encoder(**tokens)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    else:
        tokens = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model.encoder(**tokens)
        return outputs.last_hidden_state.mean(dim=1).numpy()


def find_best_segment_sequence(similarities, starts, ends, threshold):
    print("begin of find_best_seq function ", datetime.datetime.now())
    """Find the best segment sequence based on similarity scores."""
    n = len(similarities)
    if n == 0:
        return None

    above_threshold_indices = [i for i in range(n) if similarities[i] > threshold]
    if not above_threshold_indices:
        best_start_idx = np.argmax(similarities)
    else:
        best_start_idx = max(above_threshold_indices, key=lambda i: similarities[i])

    segment_indices = [best_start_idx]
    segment_start = starts[best_start_idx]
    segment_end = ends[best_start_idx]
    """
    while segment_end - segment_start < min_duration:
        remaining_indices = [i for i in range(n) if i not in segment_indices]
        if not remaining_indices:
            break

        next_candidates = [i for i in remaining_indices if starts[i] >= segment_end]
        if not next_candidates:
            break

        next_idx = min(next_candidates, key=lambda i: starts[i])
        segment_indices.append(next_idx)
        segment_end = max(segment_end, ends[next_idx])
        segment_start = min(segment_start, starts[next_idx])

    segment_duration = segment_end - segment_start
    if segment_duration <= min_duration:
        return None
    """
    segment_indices = sorted(segment_indices, key=lambda i: starts[i])
    print("end of find_best_seq function ", datetime.datetime.now())
    return segment_indices, segment_start, segment_end


# --------------------Captions--------------------------
def download_mp3(video_id: str, access_token):
    try:

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        save_dir = "videos"
        os.makedirs(save_dir, exist_ok=True)
        ydl_opts = {
            'cookies': '/root/Projects/cookies.txt',
            'verbose': 'True',
            'format': 'bestaudio/best',
            # 'proxy': 'http://aslwsiow-3:1xjoexwufte0@p.webshare.io:80',
            # FIX 1: js_runtimes should be a list, not a dict (if node is in your PATH, you can remove this entirely)
            'js_runtimes': {
                'node': {}
            },

            # FIX 2: extractor_args must be a nested dictionary with lists
            'extractor_args': {
                'youtube': {
                    'pot_provider': ['bgutil_http'],
                    'pot_base_url': ['http://pot-provider:4416'],
                    'player_client': ['web'],
                }
            },

            'outtmpl': f'{save_dir}/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],

        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            audio_url = info['url']
            return audio_url

        for file in os.listdir(save_dir):
            if file.endswith(".mp3"):
                return os.path.join(save_dir, file)

        return False


    except Exception as e:
        print("Error in yt-dlp:", str(e))
        return False


def download_captions(video_id: str, access_token: str):
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        save_dir = "captions"
        os.makedirs(save_dir, exist_ok=True)
        langs = ["en.*", "ar"]  # pick a default language

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsubtitles": True,  # <-- fixed key
            "subtitleslangs": langs,  # allow multiple languages
            "subtitlesformat": "srt",
            "outtmpl": f"{save_dir}/%(id)s.%(ext)s",
            "cookies": "/root/Projects/cookies.txt"
            # "http_headers": {"Authorization": f"Bearer {access_token}"},

            # saves subs
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            info = ydl.extract_info(video_url, download=True)
            # print(info)
            # print(info)

        # Match all .srt files for the video
        pattern = f"{save_dir}/{video_id}.en*.srt"
        matches = glob.glob(pattern)
        for subtitle_file in matches:
            print("Found:", subtitle_file)
            # subtitle_file = f"{save_dir}/{video_id}.{"en" || "en-US" || "en-GB"}.srt"
            if os.path.exists(subtitle_file):
                with open(subtitle_file, "r", encoding="utf-8") as f:
                    srt_data = f.read()
                parsed_transcript = parse_srt(srt_data)
                os.remove(subtitle_file)
                print(f"{subtitle_file} removed")
                return parsed_transcript
        return None

    except Exception as e:
        print("Error in yt-dlp:", str(e))
        return None


def parse_srt(srt_text):
    pattern = re.compile(
        r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.+?)(?=\n\d+\n|\Z)",
        re.DOTALL,
    )

    def time_to_seconds(t):
        hms, ms = t.split(",")
        h, m, s = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    transcript = []
    for match in pattern.finditer(srt_text):
        idx, start, end, text = match.groups()
        transcript.append({
            "text": text.replace("\n", " ").strip(),
            "start": time_to_seconds(start),
            "end": time_to_seconds(end)
        })

    return transcript


def run_ytws_command(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        # We construct the command as a list of strings (More secure than a single string)
        command = [
            "ytws",
            "dt",
            "-u", video_url,
            "-s",
            "--cpu",
            "-m", "base",

        ]

        # check=True raises an error if the command fails (returns non-zero exit code)
        x = subprocess.run(command, check=True)
        print(x)
        print("Transcription finished successfully.")

        # Match all .srt files for the video
        pattern = f"*{video_id}*"
        matches = glob.glob(pattern)
        for subtitle_file in matches:
            print("Found:", subtitle_file)
            # subtitle_file = f"{save_dir}/{video_id}.{"en" || "en-US" || "en-GB"}.srt"
            if os.path.exists(subtitle_file):
                with open(subtitle_file, "r", encoding="utf-8") as f:
                    srt_data = f.read()
                parsed_transcript = parse_srt(srt_data)
                os.remove(subtitle_file)
                print(f"{subtitle_file} removed")
                return parsed_transcript
        return None
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")
    except FileNotFoundError:
        print("Error: 'ytws' command not found. Make sure it is installed via pip.")


# ------------------- TRANSCRIPT FETCH -------------------
def fetch_transcript(video_id, max_retries=3, initial_delay=1):
    api_key = "6941ab603495c95c02ede62a"
    url = "https://api.scrapingdog.com/youtube/transcripts/"

    params = {
        "api_key": api_key,
        "v": video_id
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print(data)
        print("transcript fetched successfully!")
        return [{"text": s['text'], "start": s['start'], "end": s['start'] + s['duration']} for s in data['transcripts']]
    else:
        print(f"Request failed with status code: {response.status_code}")
        return False


def fetch_transcript2(video_id, access_token):
    file = download_mp3(video_id, access_token)
    if not file:
        return False
    try:
        segments, info = whisper.transcribe(
            file, beam_size=5, vad_filter=True, word_timestamps=True
        )

        return [{"text": s.text, "start": float(s.start), "end": float(s.end)} for s in segments]
    except Exception as e:
        print(e)
        return False


# --------------------------AI----------------------------

def best_video_clip(video_id, prompt, headings, access_token, merge):
    print("begin of best_video_clip function ", datetime.datetime.now())
    """Find the best video clip segment for a given prompt."""
    transcript = fetch_transcript(video_id)
    if not transcript:
        print("try to run whisper")
        transcript = download_captions(video_id, access_token)
        if not transcript:
            return {"error": "Transcript not available","best_start":0, "best_end":0}

    if merge:
        adjusted_raw = adjust_transcript(transcript)
        try:
            # First, ensure it's valid JSON
            data = json.loads(adjusted_raw)

            # Second, check if 'data' is actually a dictionary and has the key
            if isinstance(data, dict) and 'transcript' in data:
                transcript = data['transcript']
                print(transcript)
            else:
                print(f"Unexpected format from adjust_transcript: {data}")
                return {"error": "Invalid transcript format after merge","best_start":0, "best_end":0}

        except json.JSONDecodeError:
            print(f"adjust_transcript did not return valid JSON. It returned: {adjusted_raw}")
            return {"error": "Failed to parse merged transcript","best_start":0, "best_end":0}

    documents = [caption["text"] for caption in transcript]
    starts = np.array([caption["start"] for caption in transcript])
    ends = np.array([caption["end"] for caption in transcript])

    similarity_threshold = 0.70

    prompt_emb = get_t5_embedding(prompt)
    caption_embs = np.array([get_t5_embedding(doc) for doc in documents])
    similarities = [cosine_similarity(prompt_emb, emb)[0][0] for emb in caption_embs]

    result = find_best_segment_sequence(similarities, starts, ends, threshold=similarity_threshold)

    if result is None:
        return {"error": "No suitable segment found","best_start":0, "best_end":0}

    segment_indices, start_time, end_time = result

    sorted_output = sorted(
        [{"text": documents[i], "start": float(starts[i]), "end": float(ends[i]), "similarity": float(similarities[i])}
         for i in segment_indices],
        key=lambda x: x["start"]
    )
    print("end of best_video_clip function ", datetime.datetime.now())
    return {
        "best_start": float(start_time),
        "best_end": float(end_time),
        "captions": sorted_output,
    }


def compute_video_avg_embeddings(prompt, videos):
    """Compute average T5 embedding for each valid video."""
    results = [i for i in range(len(videos))]
    prompt_emb = get_t5_embedding(prompt)

    for i in range(len(videos)):
        video_id = videos[i]["video_id"]
        result = videos[i]["match_result"]

        if "error" in result:
            print(f"Skipping video {video_id}: {result['error']}")
            continue

        captions = result.get("captions", [])
        embeddings = []

        for caption in captions:
            text = caption["text"]
            emb = get_t5_embedding(text)
            embeddings.append(emb)

        if embeddings:
            video_embedding = np.mean(np.vstack(embeddings), axis=0)
            results[i] = {
                "video_id": video_id,
                "embedding_avg": video_embedding.tolist()  # convert to list for JSON compatibility
            }

    if results != []:
        similarity_scores = []
        for i in range(len(results)):
            if type(results[i]) is not int:
                similarity_scores.append(
                    cosine_similarity(prompt_emb, np.array(results[i]["embedding_avg"]).reshape(1, -1))[0][0])
            else:
                similarity_scores.append(0)

        print(f"Similarity scores: {similarity_scores}")
        best_video_indices = np.argsort(similarity_scores)[::-1].tolist()
        print(f"Best video indices: {best_video_indices}")
        videos = np.array(videos)
        return videos[best_video_indices].tolist()

    return videos

"""
def tts(voice_name, summary):
    result = tts_client.predict(
        text=summary,
        language_code="English",
        speaker="Ryan",
        api_name="/text_to_speech_edge"
    )
    print("TTS result:", result)
    return result
"""


def tts(voice_name, summary):
    try:
        command = [
            'edge-tts',
            '--voice',voice_name,
            '--text',summary,
            '--write-media','file.mp3',
            '--write-subtitles','subtitle.srt'
        ]
        subprocess.run(command,check=True)

        return os.path.abspath("file.mp3")

    except:
        print("TTS command failed")