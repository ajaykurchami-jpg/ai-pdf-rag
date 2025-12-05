import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file using pdfplumber.
    """
    full_text = ""
    
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Loop through every page
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
                print(f"[INFO] Extracted text from page {i+1}...")
            else:
                print(f"[WARNING] No text found on page {i+1} (might be an image).")
                
    return full_text

if __name__ == "__main__":
    # Test the function
    pdf_file = "sample.pdf"  # Make sure this file exists in your folder!
    
    try:
        print(f"--- Processing {pdf_file} ---")
        extracted_content = extract_text_from_pdf(pdf_file)
        
        print("\n--- Final Output Sample (First 500 chars) ---")
        print(extracted_content[:500]) # Print only the first 500 characters to keep terminal clean
        print("\n--- Extraction Complete ---")
        
    except FileNotFoundError:
        print(f"Error: The file '{pdf_file}' was not found. Please add a sample PDF.")
    except Exception as e:
        print(f"An error occurred: {e}")