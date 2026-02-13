from firebase_admin import auth, initialize_app, credentials
import firebase_admin
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_cors import cross_origin
import os
import warnings
import requests
import json
from AI_services import ensure_models_loaded, fetch_transcript, download_captions

from openAI import generate_story_AI, highlight_pdf
from tasks import run_video_prediction
from clients import firebase_creds
from celery.result import AsyncResult
from tasks import celery  # wherever you defined Celery

# from cookie_agent import login_and_get_cookies
# Check if running on Azure
ON_AZURE = os.environ.get('WEBSITE_SITE_NAME') is not None
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
# Allow all origins during development
CORS(app, resources={r"/*": {"origins": "*"}})

#  Initialize Firebase Admin
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)


# ------------------- GPU/CPU FALLBACK -------------------


# def load_models():
#   global tokenizer, model, whisper
#  if tokenizer is None or model is None or whisper is None:
#     print("Loading T5 model...")
#    tokenizer = T5Tokenizer.from_pretrained('t5-base')
#   model = T5Model.from_pretrained('t5-base')
#  print(f"Loading Whisper on {DEVICE}...")
# whisper = WhisperModel("base", device=DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
# print("Models loaded successfully!")

# ------------------- COOKIE FUNCTION -------------------


# ------------------- ROUTES -------------------
@app.route("/")
def home():
    return jsonify({"message": "YouTube Transcript Analysis API is running!", "version": "1.0.0"})


# Add health check endpoint for containers
"""
@app.route("/cookies", methods=["GET"])
def container_health_check():
    response = login_and_get_cookies()
    return response
"""


@app.route("/GetTranscript", methods=["Post"])
def get_transcript():
    """Get transcript for a specific video ID."""
    ensure_models_loaded()  # <--- Ensure models are ready

    data = request.get_json()

    if not data:
        return jsonify({"description": "Missing JSON body"}), 400

    video_id = data.get("video_id")
    start_time = float(data.get("start_time"))
    end_time = float(data.get("end_time"))
    access_token = data.get("access_token")
    merge = data.get("merge")

    print(video_id)
    print(start_time)
    print(end_time)
    print(merge)
    if not video_id:
        return jsonify({"description": "Missing parameter"}), 400

    transcript = fetch_transcript(video_id)
    # transcript = run_ytws_command(video_id)
    # transcript = download_captions(video_id,access_token)
    # print(transcript)

    if not transcript:
        print("try to run whisper")
        transcript = download_captions(video_id, access_token)
        if not transcript:
            return jsonify({"transcript": "Cannot fetch the transcript"}), 404
        else:
            print("wisper runned successfully!")

    # else:
    # for function 1 as it doesn't return end time
    # video_id = transcript.video_id
    # transcript = transcript.snippets
    # transcript = [{"text": caption.text, "start": caption.start, "end": caption.duration + caption.start} for caption in transcript]

    if end_time != 0:
        suitable_start_index = []
        if start_time != 0:
            suitable_start_index = [i for i in range(len(transcript)) if transcript[i]["start"] <= start_time]
        else:
            suitable_start_index.append(0)

        suitable_end_index = [i for i in range(len(transcript)) if transcript[i]["end"] > end_time]
        print(suitable_start_index)
        print(suitable_end_index)
        transcript = transcript[suitable_start_index[len(suitable_start_index) - 1]: suitable_end_index[0] + 1]
        print(transcript)

    return jsonify({"transcript": {"captions": transcript, "video_id": video_id}}), 200


"""
@app.route("/predict", methods=["POST"])
def predict():
    //Main prediction endpoint for finding best video clips.
    ensure_models_loaded()  # <--- Ensure models are ready

    data = request.get_json()
    if not data:
        return jsonify({"description": "Missing JSON body"}), 400

    video_ids = data.get("video_ids")
    prompt = data.get("prompt")
    headings = data.get("headings")
    access_token = data.get("access_token")
    if not video_ids or not prompt:
        return jsonify({"description": "Missing required parameters"}), 400

    if not isinstance(video_ids, list):
        return jsonify({"description": "video_ids must be a list"}), 400

    if headings is not None:
        for heading in headings:
            prompt += " - " + heading
        print(f"Using headings in prompt: {headings}")
        print(f"Updated prompt: {prompt}")


    try:
        results = []
        for video_id in video_ids:
            print(f"Processing video: {video_id}")

            result = best_video_clip( video_id, prompt, headings, access_token)
            results.append({"video_id": video_id, "match_result": result})

        results = compute_video_avg_embeddings(prompt, results)
        return jsonify({"videos": results}),200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

"""


# app.py


@app.route("/predict", methods=["POST"])
def predict():
    ensure_models_loaded()

    data = request.get_json()
    if not data:
        return jsonify({"description": "Missing JSON body"}), 400

    video_ids = data.get("video_ids")
    prompt = data.get("prompt")
    headings = data.get("headings")
    access_token = data.get("access_token")
    merge = data.get("merge")

    if not video_ids or not prompt:
        return jsonify({"description": "Missing required parameters"}), 400

    if not isinstance(video_ids, list):
        return jsonify({"description": "video_ids must be a list"}), 400

    # Submit job to Celery
    task = run_video_prediction.apply_async(args=[video_ids, prompt, headings, access_token, merge])
    return jsonify({"task_id": task.id}), 202


@app.route("/status", methods=["GET"])
def get_status():
    task_id = request.args.get("id")
    if not task_id:
        return jsonify({"error": "Missing id"}), 400

    task = AsyncResult(task_id, app=celery)

    if task.state == "PENDING":
        return jsonify({"state": "PENDING"}), 200
    elif task.state == "PROGRESS":
        return jsonify({"state": task.state, "meta": task.info}), 200
    elif task.state == "SUCCESS":
        return jsonify({"state": "SUCCESS", "result": task.result}), 200
    elif task.state == "FAILURE":
        return jsonify({"state": "FAILURE", "error": str(task.info)}), 500
    else:
        return jsonify({"state": task.state}), 200


"""
@app.route("/avatar", methods=["POST"])
def avatar():

    data = request.get_json()
    pic = data.get('pic')
    summary = data.get('summary')
    voice_name = data.get('voice_name')

    if not pic or not voice_name:
        return jsonify({"error": "pic and voice_name are required"}), 404

    if summary == "":
        return jsonify({"error": "cannot generate avatar"}), 404

    task = lip_sync.apply_async(args=[pic,summary,voice_name])


    return jsonify({"task_id": task.id}), 202
"""


@app.route("/generate-story", methods=["POST"])
def generate_story():
    data = request.get_json()
    result = generate_story_AI(data)
    return jsonify(json.loads(result)), 200


@app.route("/highlight-pdf", methods=["POST"])
def handle_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    content = highlight_pdf(file)
    print(json.loads(content))
    return jsonify(json.loads(content)), 200


"""
@app.route("/AI-video-generation",methods=["POST"])
def AI_generate_video():
    data = request.get_json()
    prompt = data.get("prompt")

    response = AI_video_generation(prompt)
    if(type(response) == str):
        return jsonify({"task_id":response}),200


    else:
        return jsonify({"task_id":"error"}),200

@app.route("/AI-video-generation-status",methods=["GET"])
def AI_generate_video_status():
    task_id = request.args.get("task_id")

    response = AI_video_generation_status(task_id)
    status = response['status']

    if(status == "succeeded"):
        video_id = response['video_id']
        result = AI_video_generation_response(video_id)
        return jsonify({"status":status,"video_data":result}),200

    elif(status == "running"):
        return jsonify({"status":status,"video_data":"running video"}),201

    elif(status == "processing"):
        return jsonify({"status":status,"video_data":"processing video"}),201

    else:
        return jsonify({"status":status,"video_data":"failed to generate video"}),500
"""
if __name__ == "__main__":
    # ensure_models_loaded()   # ❌ slow, removed
    # port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=8000)




