import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import datetime
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import whisper
from fpdf import FPDF
import base64

# Load environment variables from .env file
load_dotenv()

# Load Whisper model once at startup (silent load)
@st.cache_resource(show_spinner=False)  # Disable spinner for initial load
def load_whisper_model():
    return whisper.load_model("tiny")

whisper_model = load_whisper_model()

# Set page configuration
st.set_page_config(page_title="Audio RAG AI Agent", layout="wide")
st.title("Audio RAG AI Agent")
st.markdown("Upload an audio file to transcribe, summarize, or download as PDF.")

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Function to transcribe audio using pre-loaded Whisper model
def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file)  # No spinner here, let it process silently
    return result["text"]

# Function to summarize text using ChatGroq
def summarize_text(text):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"Summarize the following text in 100 words or less:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

# Function to create and download PDF
def create_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf_output = f"{filename}.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Function to create downloadable link
def get_binary_file_downloader_html(file_path, file_label):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{file_label}</a>'
    return href

# Function to process audio and create vector store
def process_audio_for_rag(audio_file):
    transcript = transcribe_audio(audio_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(transcript.encode('utf-8'))
        tmp_file_path = tmp_file.name
    loader = TextLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return transcript, vector_store

def main():
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_file_path = tmp_file.name
            st.write(f"File Successfully uploaded: {uploaded_file.name}")

        try:
            with st.spinner("Processing audio..."):  # Spinner for the entire processing step
                transcript, vector_store = process_audio_for_rag(audio_file_path)
            st.write("GROQ_API_KEY loaded:", os.getenv("GROQ_API_KEY") is not None)
            st.audio(uploaded_file)
            
            option = st.radio("Select an action:", ["Transcribe", "Summarize", "Download PDF"])
            
            if option == "Transcribe":
                st.header("Transcription")
                st.write(transcript)
                
            elif option == "Summarize":
                st.header("Summary")
                with st.spinner("Generating summary..."):
                    summary = summarize_text(transcript)
                st.write(summary)
                
            elif option == "Download PDF":
                st.header("Download PDF")
                with st.spinner("Creating PDF..."):
                    pdf_file = create_pdf(transcript, f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                st.markdown(get_binary_file_downloader_html(pdf_file, "Download Transcript PDF"), unsafe_allow_html=True)
            
            st.header("Ask Questions About the Audio Content")
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(),
                return_source_documents=True
            )
            query = st.text_input("Enter your question about the audio content:")
            if query:
                with st.spinner("Fetching answer..."):
                    result = chain({"question": query, "chat_history": st.session_state['history']})
                st.session_state['history'].append((query, result["answer"]))
                st.write("**Answer:**", result["answer"])
                st.subheader("Chat History")
                for q, a in st.session_state['history']:
                    st.write(f"**Q:** {q}")
                    st.write(f"**A:** {a}")
                    st.write("---")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()