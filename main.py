from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import os
import shutil

app = FastAPI()

# Allow any frontend (React, etc.) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

resumes_texts = {}
job_description = ""

@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text from PDF
    with fitz.open(file_path) as doc:
        text = " ".join([page.get_text() for page in doc])
    resumes_texts[file.filename] = text
    return {"filename": file.filename, "message": "Resume uploaded and parsed successfully."}

@app.post("/upload_jd/")
async def upload_jd(jd: str = Form(...)):
    global job_description
    job_description = jd
    return {"message": "Job description uploaded successfully."}

@app.get("/score_resumes/")
async def score_resumes():
    if not job_description:
        return {"error": "No job description uploaded."}

    corpus = [job_description] + list(resumes_texts.values())
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vec = tfidf_matrix[0:1]
    resume_vecs = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vec, resume_vecs).flatten()

    results = []
    for idx, (filename, _) in enumerate(resumes_texts.items()):
        results.append({"filename": filename, "score": round(similarities[idx] * 100, 2)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"ranked_resumes": results}

if __name__ == "_main_":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)