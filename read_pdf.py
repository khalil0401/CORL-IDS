import PyPDF2
import sys

def extract_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for i, page in enumerate(reader.pages):
                text += f"\n--- Page {i+1} ---\n"
                text += page.extract_text()
            
            with open("C:/Users/ATECH STORE/Desktop/projects/CORL-IDS/extracted_paper.txt", "w", encoding="utf-8") as out:
                out.write(text)
            print("Successfully extracted text to extracted_paper.txt")
    except Exception as e:
        print(f"Error: {e}")
        
        # Try fitz if PyPDF2 fails
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for i, page in enumerate(doc):
                text += f"\n--- Page {i+1} ---\n"
                text += page.get_text()
            
            with open("C:/Users/ATECH STORE/Desktop/projects/CORL-IDS/extracted_paper.txt", "w", encoding="utf-8") as out:
                out.write(text)
            print("Successfully extracted text using PyMuPDF to extracted_paper.txt")
        except Exception as e2:
            print(f"Error with fitz: {e2}")

if __name__ == "__main__":
    extract_text("C:/Users/ATECH STORE/Desktop/projects/CORL-IDS/CORL-IDS.pdf")
