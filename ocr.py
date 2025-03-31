import os
import json
import fitz  # PyMuPDF
from PIL import Image, ImageOps
from paddleocr import PaddleOCR
import pdfplumber
from pathlib import Path

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en", ocr_version="PP-OCRv4")

def pdf_to_images(pdf_path, output_folder="output_images"):
    """Convert PDF to images using PyMuPDF (fitz)."""
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img_path = os.path.join(output_folder, f"{Path(pdf_path).stem}_page_{i+1}.png")
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(img_path, "PNG")
        image_paths.append((img_path, i+1))
    
    return image_paths

def preprocess_image(image_path):
    """Improve OCR accuracy by enhancing the image."""
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = ImageOps.autocontrast(img)  # Increase contrast
    img = img.resize((img.width * 2, img.height * 2))  # Upscale
    img.save(image_path)  # Overwrite with processed image
    return image_path

def extract_text_from_image(image_path):
    """Extract text using PaddleOCR after preprocessing."""
    preprocess_image(image_path)  # Enhance image before OCR
    result = ocr.ocr(image_path)

    if not result:  # If OCR returns None, handle it
        return None

    extracted_text = []
    for res in result:
        if res:  # Check if OCR detected text
            for line in res:
                text, _ = line[1]  # Ignore confidence score
                if text.strip():
                    extracted_text.append(text)

    return extracted_text if extracted_text else None

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [
                (page.extract_text(), page_num + 1)
                for page_num, page in enumerate(pdf.pages)
                if page.extract_text() and page.extract_text().strip()
            ]
        return pages if pages else None
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return None

def process_pdfs(pdf_paths, save_folder="ocr_results"):
    """Process multiple PDFs: convert to images, extract text, and store metadata."""
    os.makedirs(save_folder, exist_ok=True)
    all_docs = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            continue

        print(f"Processing PDF: {pdf_path}")

        # Convert PDF pages to images
        image_paths = pdf_to_images(pdf_path)
        pdf_has_text = False

        for img_path, page_num in image_paths:
            extracted_text = extract_text_from_image(img_path)

            if extracted_text:
                doc_metadata = {"file": Path(pdf_path).name, "page_number": page_num}
                all_docs.append({"metadata": doc_metadata, "text": extracted_text})
                pdf_has_text = True  # At least one page had valid OCR text

        # Fallback to pdfplumber if OCR failed on all pages
        if not pdf_has_text:
            print(f"Using pdfplumber for {pdf_path} (OCR failed)")
            extracted_pages = extract_text_from_pdf(pdf_path)
            if extracted_pages:
                for text, page_num in extracted_pages:
                    doc_metadata = {"file": Path(pdf_path).name, "page_number": page_num}
                    all_docs.append({"metadata": doc_metadata, "text": [text]})

    return all_docs

# Example usage
pdf_files = ["D:/New folder/Alices_Adventures_in_Wonderland.pdf"]  # Add your PDFs here
results = process_pdfs(pdf_files)

# Save results
output_file = "ocr_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"OCR processing complete. Results saved in '{output_file}'.")
