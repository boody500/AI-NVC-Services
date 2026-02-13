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
        system_text = "i will give you a full content extracted from a pdf or a simple text, this content should be talking about some topics. " \
                      "i want you to act as a creator and take this pdf content or this text and to split it into scenes and to give each scene a title that will give a good results if i entered it on youtube search engine. " \
                      "i want you to make scenes contant related to each other as it will be narated using avatar. " \
                      "i want you for each scene to generate a content that describe it as it will be givin for AI video generation tool like sora,google veo." \
                      "i want you to return the scenes content and titles in json format like this({\"scenes\":[{\"content\" : \"scene content\",\"title\" : \"scene title\",\"AI_video_describtion\":\"AI video describtion\"}]}) and with the same language as the pdf content is and i will give you the pdf content in json format. max 5 scenes"

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
        max_tokens=10000
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
                "content": "you will be givin a pdf talking about some topic, i want you without summrizing the pdf content to just make the content readable as a document with the same language it's given, return json as this{\"content\":\"document_content\"}. again and again don't summarize the pdf and keep the explinations of the topics as it is if available.make the document content human readable"

            },
            {
                "role": "user",
                "content": text,
            }
        ],
        max_tokens=10000
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

                "content": "you will be givin a list contains video transcript includes captions with time stamps i want you to merge the related captions with each other to form a chunk with start and end time max 100 token caption per chunk and to keep the video sequance in consideration and try to always end the chunk with the most appropirate ending in the captions. return only a json with the new chunks like this {\"transcript\":[{\"text\":\"merged caption\",\"start\":\"start time\",\"end\":\"end time\"}]} don't add any extra discriptions just return the json format. "

            },

            {

                "role": "user",

                "content": f"{transcript}",

            }

        ],

        max_tokens=10000

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