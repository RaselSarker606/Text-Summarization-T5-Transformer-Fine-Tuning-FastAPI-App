# pip install fastapi uvicorn transformers sentencepiece
# pip install jinja2
# uvicorn app:app --reload
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import re

# Initialize FastAPI app
app = FastAPI(title="Text Summarization System", description="Summarize dialogues with T5!", version="1.0")

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Ensure the model is on the correct device
device = "cuda" if model.device.type == "cuda" else "cpu"
model = model.to(device)

# Mount templates
templates = Jinja2Templates(directory="templates")


# Input schema for requests
class DialogueInput(BaseModel):
    dialogue: str


# Clean text function
def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text


# Summarization function
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_text(dialogue)
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate summary
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# API endpoint for text summarization
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


# HTML UI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})