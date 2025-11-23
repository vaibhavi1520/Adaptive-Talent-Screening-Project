
import os
import time
import whisper
import sounddevice as sd
import soundfile as sf
import spacy
from PyPDF2 import PdfReader
from spacy.lang.en.stop_words import STOP_WORDS
import google.generativeai as genai
import re
from gtts import gTTS
import pygame
from tkinter import Tk, filedialog
import datetime
import numpy as np
import random
import json
import pickle


# --------------------- Text-to-Speech (gTTS + pygame) ---------------------
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = "temp.mp3"
    tts.save(filename)

    
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.music.unload()
    os.remove(filename)

# --------------------- Speech-to-Text (Whisper) ---------------------
model = whisper.load_model("base", device="cpu")

def record_audio(filename="response.wav", duration=20, fs=16000):
    print("Recording answer...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, recording, fs)
    print("Recording complete.")

def transcribe_audio(filename="response.wav"):
    print("Transcribing...")
    result = model.transcribe(filename, fp16=False)
    print("Candidate Answer:", result["text"])
    return result["text"]

# --------------------- NLP + PDF Keyword Extraction ---------------------
nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return ' '.join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_keywords(text):
    doc = nlp(text)
    keywords = set()
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in STOP_WORDS and token.is_alpha:
            keywords.add(token.lemma_.lower())
    return keywords

# --------------------- File Paths ---------------------
def pick_file(title):
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title=title, filetypes=[("PDF Files", "*.pdf")])
    root.destroy()
    return file_path


resume_path = pick_file(input("Select your Resume PDF"))
job_desc_path = pick_file(input("Select Job Description PDF"))

if not resume_path or not os.path.exists(resume_path):
    print("Resume file not selected or doesn't exist.")
    exit()

if not job_desc_path or not os.path.exists(job_desc_path):
    print("Job Description file not selected or doesn't exist.")
    exit()

resume_text = extract_text_from_pdf(resume_path)
job_text = extract_text_from_pdf(job_desc_path)

resume_keywords = extract_keywords(resume_text)
job_keywords = extract_keywords(job_text)
matched_keywords = resume_keywords & job_keywords

# --------------------- Gemini API Configuration and Functions ---------------------
genai.configure(api_key="AIzaSyB3Hfn2PHJtrkbjYYVOERXStLtF5vqMUus")

def generate_dynamic_question(job_description, resume, candidate_history, prev_answer, prev_questions, question_type):
    """
    Generate a new must be short length interview question dynamically using Gemini API.
    question_type can be: "library", "tool", "concept".
    
    """
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-001")

    # Build context from history
    history_text = "\n".join([f"Q: {h['question']} | A: {h['answer']} | Score: {h['score']}" for h in candidate_history[-3:]])

    prompt = f"""
    You are an adaptive AI interviewer.
    The candidate has already answered some questions. Here is their recent history:

    {history_text if history_text else "No prior answers yet."}

    Based on their performance, generate a new {question_type} interview question.
    -Keep it short and clear (1-2 lines max).
    -basic easy questions.
    -Don't ask question such as describe your experience with.
    -Rotate between different types of questions:
       - Fundamental concepts (theory, definitions, basics)
       - Tools/technologies (specific libraries, frameworks, methods)
       - Project-based (resume/project experience)
       
    
    -Ensure each new question belongs to a different theme than before
    -The question should relate to this job description and resume:
    

    Job Description:
    {job_description}

    Resume:
    {resume}

    Important: Do NOT repeat or paraphrase previous questions. 
    Make the next question different in focus and scope.

    Candidate's previous answer: "{prev_answer}"
    Questions already asked: {prev_questions}
    """

    response = model.generate_content(prompt)
    return response.text.strip()

    
    

# --------------------- RL-Specific Functions ---------------------
def get_gemini_evaluation(question, answer):
    """
    Evaluates a candidate's answer using the Gemini API and returns a numerical score and text feedback about how well or poor the candidiate answered.
    """
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
    
    prompt = f"""
    You are an expert technical interviewer. Your task is to evaluate a candidate's response to an interview question.

    Question: {question}
    Candidate's Answer: {answer}

    Provide a score from 0.0 to 1.0 (where 1.0 is an excellent answer) and concise,2 line and constructive feedback in the following JSON format:

    {{
      "score": <score_as_a_float>,
      "feedback": "<detailed_feedback_text_here>"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        # Find the JSON object within the response text, in case there's extra text
        json_string = response.text.strip().strip('`') # Remove markdown code block fences
        
        # A more flexible way to find the JSON object
        start_index = json_string.find('{')
        end_index = json_string.rfind('}') + 1
        
        if start_index != -1 and end_index != 0:
            json_string = json_string[start_index:end_index]
        
        evaluation = json.loads(json_string)
        score = evaluation.get("score")
        feedback = evaluation.get("feedback")
        
        if isinstance(score, (int, float)) and isinstance(feedback, str):
            return score, feedback
        else:
            print("Warning: Gemini returned an unexpected JSON format.")
            return 0.1, f"Evaluation failed: Response format unexpected. Raw response: {response.text}"
            
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error getting Gemini evaluation: {e}")
        # Print the raw response text for debugging
        print(f"Raw Gemini response for debugging: {response.text}")
        return 0.1, "Evaluation failed: Could not parse the model's response."

# --------------------- RL Model Parameters and Q-Table Loading ---------------------
alpha = 0.1
gamma = 0.9
epsilon = 0.2
q_table_file = "q_table.pkl"

if os.path.exists(q_table_file):
    with open(q_table_file, 'rb') as f:
        q_table = pickle.load(f)
    
else:
    q_table = {}
    

# --------------------- RL-Driven Interview Loop ---------------------
# RL-Driven Dynamic Interview Loop
asked_questions = []

candidate_history = []
total_score = 0
num_questions_to_ask = 5

question_types = ["library", "tool", "concept", "project"]

for i in range(num_questions_to_ask):
    current_state = tuple([q['type'] for q in candidate_history])  # State = past question types

    # Exploration vs Exploitation
    if random.uniform(0, 1) < epsilon or not q_table:
        action_index = random.choice(range(len(question_types)))
    else:
        q_values = [q_table.get((current_state, a), 0.0) for a in range(len(question_types))]
        action_index = int(np.argmax(q_values))

    selected_type = question_types[action_index]

    # Generate dynamic question based on candidate's history
    selected_question = generate_dynamic_question(job_text, resume_text, candidate_history, asked_questions, candidate_history, question_type=selected_type)
    asked_questions.append(selected_question)
    print(f"\nQuestion {i + 1} : ({selected_type}): {selected_question}")
    

    speak(selected_question)
    time.sleep(1.5)

    # Record and transcribe candidate's response
    record_audio()
    candidate_answer = transcribe_audio()
    time.sleep(2)

    # Evaluate candidate’s response
    score, feedback = get_gemini_evaluation(selected_question, candidate_answer)
    total_score += score
    
    if len(candidate_answer.split()) < 3:  # very short or noise
       print(f"\n--- Evaluation for Question {i+1} ---")
       print("Score: 0.0/1.00")
       print("Feedback: Inaudible or too short. Please provide a more detailed answer.")
       continue

    print(f"\n--- Evaluation for Question {i + 1} ---")
    print(f"Score: {score:.2f}/1.00")
    print(f"Feedback: {feedback}")
    print("---------------------------------------")

    # Save history
    candidate_history.append({
        "type": selected_type,
        "question": selected_question,
        "answer": candidate_answer,
        "score": score,
        "feedback": feedback
    })

    # RL Update
    next_state = tuple([q['type'] for q in candidate_history])
    current_q = q_table.get((current_state, action_index), 0.0)
    next_q_values = [q_table.get((next_state, a), 0.0) for a in range(len(question_types))]
    max_next_q = max(next_q_values) if next_q_values else 0.0

    new_q = current_q + alpha * (score + gamma * max_next_q - current_q)
    q_table[(current_state, action_index)] = new_q


# --------------------- Final Results and Q-Table Saving ---------------------
print("\nInterview completed.")
if all(h['score'] == 0 for h in candidate_history):
    total_score = 0

print(f"Final Cumulative Score: {total_score:.2f} / {num_questions_to_ask:.2f}")

with open(q_table_file, 'wb') as f:
    pickle.dump(q_table, f)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_file = f"q_table_{timestamp}.pkl"
with open(backup_file, 'wb') as f:
    pickle.dump(q_table, f)


