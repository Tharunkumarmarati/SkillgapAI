from flask import Flask, render_template, request
import os
import pdfplumber
import docx

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""
    word_count = 0
    char_count = 0

    if request.method == "POST":
        file = request.files["document"]
        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            extracted_text = extract_text(path)
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)

    return render_template(
        "index.html",
        text=extracted_text,
        words=word_count,
        chars=char_count
    )

if __name__ == "__main__":
    app.run(debug=True)
