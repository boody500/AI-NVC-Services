from openai import OpenAI
import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests


openAI_client = OpenAI(
    base_url="https://jcdty44x3zy34s7k5g4pwjma.agents.do-ai.run/api/v1",
    api_key="oxvQnSCb9TGKKq-y1QlqwNGQ1ykZ0g_e",
    #api_version="2025-01-01-preview",
)


#sora_endpoint = os.getenv("sora_endpoint")
#sora_key = os.getenv("sora_key")

cloudinary.config(
  cloud_name = "dm5msh04j",
  api_key = "261323371413573",
  api_secret = "6MXlYddwaHfsBQJ-2Zyd4pJMsik"
)

#  Fetch Firebase credentials JSON from Cloudinary
CLOUDINARY_URL = "https://res.cloudinary.com/dm5msh04j/raw/upload/v1756647828/ai-nvc-1873f-firebase-adminsdk-fbsvc-6baaef323b_sotsb7.json"
res = requests.get(CLOUDINARY_URL)
res.raise_for_status()  # make sure it downloads fine
firebase_creds = res.json()