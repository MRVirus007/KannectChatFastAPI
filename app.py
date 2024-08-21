from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util
import json

app = FastAPI()

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

# Load the QA data from the JSON file
with open('qa_data.json') as f:
    qa_data = json.load(f)

# Function to find the most relevant answer for the given question
def get_relevant_answer(question, threshold=0.5):
    questions = [item["question"] for item in qa_data]
    answers = [item["answer"] for item in qa_data]

    # Encode the input question and the questions in the QA data
    question_embedding = model.encode(question, convert_to_tensor=True)
    qa_embeddings = model.encode(questions, convert_to_tensor=True)

    # Compute cosine similarity between the input question and QA questions
    similarities = util.pytorch_cos_sim(question_embedding, qa_embeddings)

    # Find the index of the most similar question
    best_match_idx = similarities.argmax().item()
    best_score = similarities[0][best_match_idx].item()

    # Check if the best score is above the threshold
    if best_score >= threshold:
        return answers[best_match_idx]
    else:
        return "Sorry, I don't have an answer for that question."

# API endpoint to get the most relevant answer
@app.get("/get_answer/")
async def get_answer(question: str):
    answer = get_relevant_answer(question)
    return {"answer": answer}