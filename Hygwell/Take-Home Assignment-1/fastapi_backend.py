#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[2]:


# Importing Libraries
from fastapi import FastAPI, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Dict
from uuid import uuid4
import requests
from bs4 import BeautifulSoup
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nest_asyncio
import uvicorn
import os

# Apply nest_asyncio to allow running FastAPI in Jupyter Notebook
nest_asyncio.apply()


# ### Initialize FastAPI App and Database

# In[4]:


app = FastAPI()

# In-memory database to store processed content
DATABASE: Dict[str, str] = {}

def generate_chat_id():
    return str(uuid4())


# ### Define the Models

# In[6]:


class URLRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    chat_id: str
    question: str


# ### Implement the /process_url Endpoint

# In[8]:


@app.post("/process_url")
def process_url(request: URLRequest):
    url = request.url
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        chat_id = generate_chat_id()
        DATABASE[chat_id] = text
        return {"chat_id": chat_id, "message": "URL content processed and stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing URL: {str(e)}")


# ### Implement the /process_pdf Endpoint

# In[10]:


@app.post("/process_pdf")
def process_pdf(file: UploadFile):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Uploaded file is not a PDF.")
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() or ''
        chat_id = generate_chat_id()
        DATABASE[chat_id] = text.strip()
        return {"chat_id": chat_id, "message": "PDF content processed and stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")


# ### Implement the /chat Endpoint

# In[12]:


@app.post("/chat")
def chat(request: ChatRequest):
    chat_id = request.chat_id
    question = request.question

    if chat_id not in DATABASE:
        raise HTTPException(status_code=404, detail="Chat ID not found.")

    content = DATABASE[chat_id]
    texts = [content, question]

    # Generate TF-IDF embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts)

    # Calculate similarity
    similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])

    if similarity[0][0] > 0.1:  # Threshold for relevance
        response = f"The main idea of the document is: {content[:200]}..."
    else:
        response = "Sorry, I couldn't find a relevant response."

    return {"response": response}


# ### Run the FastAPI Application

# In[14]:


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# In[ ]:




