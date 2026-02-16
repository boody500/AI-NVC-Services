import os
import json
from PyPDF2 import PdfReader
import requests
import time
from clients import openAI_client, cloudinary


def generate_story_AI(data):
    user_content = data.get("user_content")
    summary = data.get("summary")
    system_text = ""
    if summary:
        system_text = "i will gave you some highlighted points extracted from pdf(topics and subtopics) in json format, " \
                      "i want you to act as a creator and take this pdf content and to split it into scenes and to give each scene a title that will give a good results if i entered it on youtube search engine. " \
                      "i want you for each scene to generate a content that describe it as it will be givin for AI video generation tool like sora,google veo." \
                      "i want you to make scenes contant related to each other as it will be narated using avatar. i want you to return the scenes content and titles in json format like this({\"scenes\":[{\"content\" : \"scene content\",\"title\" : \"scene title\",\"AI_video_describtion\":\"AI video describtion\"}]}) and with the same language as the highlighted points is. max 5 scenes"





    else:
        system_text = (
            "You will receive full content extracted from a PDF or a plain text, provided in JSON format. "
            "The content discusses one or more related topics. "

            "Your task is to transform this content into a short, coherent video script divided into scenes.\n\n"

            "Requirements:\n"
            "- Split the content into a maximum of 5 scenes.\n"
            "- Each scene must:\n"
            "  - Represent a clear idea or topic from the content.\n"
            "  - Have a title optimized for YouTube search.\n"
            "  - Contain narration text suitable for an AI avatar.\n"
            "  - Include a visual description suitable for AI video generation tools (e.g., Sora, Google Veo).\n"
            "- Scenes must be logically connected and flow naturally as a narrated story.\n"
            "- Use the same language as the input content.\n\n"

            "Return the output strictly in the following JSON format:\n"
            "{"
            "\"scenes\": ["
            "{"
            "\"title\": \"YouTube-optimized scene title\", "
            "\"content\": \"Narration content for the avatar\", "
            "\"AI_video_describtion\": \"Visual description for AI video generation\""
            "}"
            "]"
            "}"
        )

    completion = openAI_client.chat.completions.create(
        model="OpenAI GPT-oss-120b",
        messages=[
            {
                "role": "system",
                "content": system_text

            },
            {
                "role": "user",
                "content": user_content,
            }
        ],
        max_tokens=20000
    )
    return completion.choices[0].message.content


def highlight_pdf(pdf_file):
    pdf = PdfReader(pdf_file)

    # Extract text from all pages
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    completion = openAI_client.chat.completions.create(
        model="OpenAI GPT-oss-120b",
        response_format={"type": "json_object"},  # ✅ force JSON

        messages=[
            {
                "role": "system",
                "content": "You will receive full content extracted from a PDF, written in a specific language. Your task is NOT to summarize, shorten, or rewrite the ideas.\n\nRewrite the content only to improve readability and structure while preserving all original information, explanations, and level of detail exactly as provided.\n\nRequirements:\n- Do NOT summarize, compress, or remove any information.\n- Do NOT add new information or interpretations.\n- Keep all topic explanations exactly as they appear, if available.\n- Improve readability by fixing grammar, spacing, punctuation, and flow.\n- Preserve the original language of the input content.\n- The final output should read like a clean, human-readable document.\n\nReturn the output strictly in the following JSON format:\n{\"content\": \"document_content\"}\n\nDo not include any text outside the JSON response."

            },
            {
                "role": "user",
                "content": text,
            }
        ],
        max_tokens=20000
    )
    return completion.choices[0].message.content


def adjust_transcript(transcript):
    completion = openAI_client.chat.completions.create(

        model="OpenAI GPT-oss-120b",
        response_format={"type": "json_object"},  # ✅ force JSON

        messages=

        [

            {

                "role": "system",

                "content": "You will receive a list representing a video transcript. Each item contains caption text with start and end timestamps.\n\nYour task is to merge related and consecutive captions into coherent chunks while preserving the original video sequence.\n\nRequirements:\n- Merge only captions that are contextually related and consecutive in time.\n- Each chunk must contain a maximum of 100 tokens.\n- Preserve chronological order exactly as in the original transcript.\n- Each chunk must have:\n  - \"text\": the merged caption text\n  - \"start\": the start time of the first caption in the chunk\n  - \"end\": the end time of the last caption in the chunk\n- Always try to end each chunk at a natural or meaningful stopping point in the captions (e.g., sentence end, idea completion).\n- Do NOT skip, reorder, summarize, or rewrite caption content.\n- Do NOT add any new text or interpretation.\n\nReturn the output strictly in the following JSON format:\n{\"transcript\":[{\"text\":\"merged caption\",\"start\":\"start time\",\"end\":\"end time\"}]}\n\nDo not include any explanations, comments, or text outside the JSON response."

            },

            {

                "role": "user",

                "content": f"{transcript}",

            }

        ],

        max_tokens=20000

    )

    return completion.choices[0].message.content


"""
def AI_video_generation(prompt):

    headers = {
        "Content-Type": "application/json",
        "Api-key": sora_key
    }
    # Request body
    data = {
        "model": "sora",
        "prompt": prompt,
        "height": "1080",
        "width": "1080",
        "n_seconds": "5",
        "n_variants": "1"
    }

    task_id = ""
    try:
        response = requests.post(url=sora_endpoint+"/jobs?api-version=preview",data=json.dumps(data),headers=headers)
        response_json = response.json()
        task_id = response_json['id']
        return task_id

    except Exception:
        print("response",response)
        return({"error":response.text})

def AI_video_generation_status(task_id):  
    headers = {
        "Content-Type": "application/json",
        "Api-key": sora_key
    }  
    print("task_id: ",task_id)
    status = ""
    response2 = ""
    json2 = ""


    try:
        time.sleep(5)
        response2 = requests.get(url=sora_endpoint+f"/jobs/{task_id}?api-version=preview",headers=headers)
        json2 = response2.json()
        status = json2['status']
        #return status
    except Exception:
        print("response2",response2)
        return(response2.text)

    data = json2['generations']

    if status == "succeeded":
        video_id = data[0]['id']
        return {"status":status,"video_id": video_id}
    else:
        return{"status":status}


def AI_video_generation_response(video_id):
    try:
        print("video_id: ",video_id)
        response3 = requests.get(url=sora_endpoint+f"/{video_id}/content/video?api-version=preview",headers={"Api-key":sora_key})
        file_name="output.mp4"

        if response3.status_code == 200:
            with open(file_name, "wb") as f:
                for chunk in response3.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Video saved at",file_name)
            cloud_response = cloudinary.uploader.upload(file_name,resource_type="video")

            os.remove(file_name)
            print("video deleted from storage")

            image = cloud_response['secure_url'].replace("mp4","jpg")
            return {"url":cloud_response["secure_url"],"id":cloud_response['asset_id'],"thumbnail":image,"start_time":0,"end_time":5}
        else:
            print("❌ Failed to save video:", response3.status_code, response3.text)

    except Exception:
        print("response3",response3)
        return(response3.text)
"""