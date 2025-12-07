import PyPDF2
import sys
import os

pdf_path = '9a35a69227a649c6a1124bba1ea8d6fb.marked_uJoACAb.pdf'

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

try:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"--- Page {i+1} ---\n"
            text += page.extract_text() + "\n"
        with open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Text extracted to pdf_content.txt")
except Exception as e:
    print(f"Error: {e}")
