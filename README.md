# Audio RAG AI Agent

This is a Streamlit-based AI application that allows users to:

Upload an audio file (e.g., `.mp3`, `.wav`, `.m4a`)  
Transcribe the audio using OpenAI Whisper  
Summarize the transcript using ChatGroq (LLaMA 3 model)  
Ask questions about the audio content using RAG (Retrieval-Augmented Generation)  
Download the transcript as a PDF

---

## Tech Stack

- **Whisper** – for audio transcription
- **ChatGroq (LLaMA 3.3 70B)** – for summarization and Q&A
- **LangChain** – for chaining LLM with document retrieval
- **FAISS** – for semantic similarity search (vector DB)
- **Hugging Face Sentence Transformers** – for text embeddings
- **Streamlit** – for the user interface

---
