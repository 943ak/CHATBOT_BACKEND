from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Adjust this to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

def google_search(query: str):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "answer_box" in data:
        return data["answer_box"].get("answer") or data["answer_box"].get("snippet")
    elif "organic_results" in data and len(data["organic_results"]) > 0:
        return data["organic_results"][0].get("snippet", "No snippet found.")
    return "No results found."

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    messages = [message.dict() for message in chat_request.messages]
    user_input = messages[-1]["content"]

    if "search" in user_input.lower() or "google" in user_input.lower():
        query = user_input.replace("search", "").replace("google", "").strip()
        result = google_search(query)
        return {"reply": result}

    try:
        completion = client.chat_completion(model="meta-llama/Llama-3.1-8B-Instruct", messages=messages)
        bot_reply = completion.choices[0].message.content
        return {"reply": bot_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
