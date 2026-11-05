# Document Chatbot

An AI chatbot that answers questions from your own PDF documents, built with Python, Gemini and Streamlit.

## How it works
1. ingest.py reads the PDFs in the docs folder, splits them into chunks and stores them in a vector database.
2. app.py takes a question, finds the most relevant chunks and asks Gemini to answer using only those.
3. Every answer shows which PDF it came from.

## Tech
Python, Streamlit, ChromaDB, Google Gemini API

## Run it locally
1. Install the packages: pip install google-genai chromadb pypdf python-dotenv streamlit
2. Create a .env file with GEMINI_API_KEY=your_key
3. Put PDFs in the docs folder and run: python ingest.py
4. Start the app: streamlit run app.py

Built by Tirth.
