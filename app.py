from flask import Flask, request, jsonify, render_template
import os
import pdfplumber
from transformers import pipeline

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Extract text from PDF using pdfplumber
def extract_text_from_pdf(file_path):
    try:
        text = ''
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  
                    text += page_text
        if not text:
            raise ValueError("No text found in PDF.")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

# Initialize the Hugging Face pipeline for Question Answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@app.route('/')
def home():
    return render_template('index.html')

# Upload PDF and extract text
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        try:
            file.save(file_path)
            text = extract_text_from_pdf(file_path)
            if text is None:
                return jsonify({"error": "Failed to extract text. The file may not contain readable text."}), 400
            print("Extracted Text:", text[:200])  
            return jsonify({"message": "File uploaded successfully", "text": text}), 200
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return jsonify({"error": "An error occurred while processing the PDF."}), 500
    else:
        return jsonify({"error": "Invalid file format. Only PDF is allowed."}), 400

# Answer a question based on document text
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    document_text = data.get('document_text')

    if not question or not document_text:
        return jsonify({"error": "Both 'question' and 'document_text' are required."}), 400

    try:
        # Answering question based on document text
        answer = qa_pipeline(question=question, context=document_text)
        print(f"Question: {question}, Answer: {answer['answer']}")  
        return jsonify({"answer": answer['answer']}), 200
    except Exception as e:
        print(f"Error in QA pipeline: {e}")
        return jsonify({"error": "An error occurred while processing the question."}), 500

if __name__ == '__main__':
    app.run(debug=True)
