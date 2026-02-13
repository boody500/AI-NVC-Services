# tasks.py
import os
import ssl
import signal
from celery import Celery

from AI_services import ensure_models_loaded, best_video_clip, compute_video_avg_embeddings, wav2lip_client, file,tts
from clients import cloudinary
import time
from timeout_decorator import timeout, TimeoutError

# redis_url = f"rediss://:{redis_password}@{redis_host}:{redis_port}/0"
redis_url = "rediss://default:AdQjAQIncDFiNjM4MGUwNjgzYzc0YjllODBhY2FiNTBmMmE1NDM0NHAxNTQzMDc@touched-titmouse-54307.upstash.io:6379"

celery = Celery("tasks", broker=redis_url, backend=redis_url)
# Azure Redis requires ssl_cert_reqs=ssl.CERT_NONE unless you upload a CA cert
celery.conf.update(
    broker_use_ssl={"ssl_cert_reqs": "required"},
    redis_backend_use_ssl={"ssl_cert_reqs": "required"},
)

import threading
import functools

"""
class TimeoutError(Exception):
    pass


def timeout(seconds, use_signals=False):
    
    #A Windows-safe replacement for timeout_decorator.
    #It uses threads instead of multiprocessing to avoid WinError 6.

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = []
            exception = []

            def worker():
                try:
                    res = func(*args, **kwargs)
                    result.append(res)
                except Exception as e:
                    exception.append(e)

            # Create a thread to run the function
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()

            # Wait for the thread to finish or time out
            t.join(seconds)

            if t.is_alive():
                # If still running after 'seconds', raise TimeoutError
                raise TimeoutError(f"Task timed out after {seconds} seconds")

            if exception:
                raise exception[0]

            if not result:
                # Should not happen if function returns None, handled by list check
                return None

            return result[0]

        return wrapper

    return decorator


"""

# ===== Timeout Wrapper =====
@timeout(1200, use_signals=False)  # 300 seconds per video (thread-based timeout)
def safe_best_video_clip(video_id, prompt, headings, access_token, merge):
    """Run best_video_clip safely with a per-video timeout."""
    return best_video_clip(video_id, prompt, headings, access_token, merge)


@celery.task(bind=True)
def run_video_prediction(self, video_ids, prompt, headings, access_token, merge):
    """Background job for heavy video prediction"""
    ensure_models_loaded()

    if headings:
        for heading in headings:
            prompt += " - " + heading

    results = []

    for idx, video_id in enumerate(video_ids):
        try:
            print(f"Processing video: {video_id}")
            self.update_state(state="PROGRESS", meta={"step": idx + 1, "total": len(video_ids)})
            result = safe_best_video_clip(video_id, prompt, headings, access_token, merge)
            results.append({"video_id": video_id, "match_result": result})

        except TimeoutError:
            print("Task exceeded soft time limit — cleaning up...")
            results.append({"video_id": video_id, "match_result": {"error": "No Suitable Segment Found"}})

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            results.append({
                "video_id": video_id, "match_result": {"error": str(e)}})

    results = compute_video_avg_embeddings(prompt, results)
    return {"videos": results}



@celery.task(bind=True)
def lip_sync(self,pic, summary, voice_name,audio_path):

    if voice_name != "":
        print("preset audio",voice_name)
        tts_result = tts(voice_name, summary)
        audio_path = tts_result

    self.update_state(state="PROGRESS")

    result = wav2lip_client.predict(
        input_image=file(pic),
        input_audio=file(audio_path),
        api_name="/run_infrence"
    )

    print("Video output URL:", result)
    video_path = result["video"]
    response = cloudinary.uploader.upload(video_path, resource_type="video")

    try:
        os.remove(video_path)
        print(f"{video_path} removed")
        os.remove(audio_path)
        print(f"{audio_path} removed")
    except:
        pass
    image = response['secure_url'].replace("mp4","jpg")

    return {"url":response["secure_url"],"id":response['asset_id'],"thumbnail":image,"duration":response['duration']}

