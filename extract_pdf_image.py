import fitz  # PyMuPDF
import os

def extract_second_image_from_pdfs(pdf_folder, output_folder):
    # List all PDF files in the provided folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        # Extract the student name from the PDF file name
        student_name = pdf_file.split('-')[1].split('.')[0]  # Adjust according to file naming convention

        # Create a folder for the student
        student_folder = os.path.join(output_folder, student_name)
        if not os.path.exists(student_folder):
            os.makedirs(student_folder)

        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Initialize image count
        image_count = 0

        # Loop through each page in the PDF
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            # Get the images on the page
            image_list = page.get_images(full=True)

            if len(image_list) >= 2:
                # Extract the second image
                xref = image_list[1][0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save the image
                image_filename = os.path.join(student_folder, "image.png")
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
                
                print(f"Saved second image from {pdf_file} to {image_filename}")
                break  # Exit the loop after saving the second image

        pdf_document.close()  # Close the PDF document

    print("Image extraction completed.")

# Usage
pdf_folder = r"C:\Users\sanjay\Downloads\Results-bca-sem-4-se\Results"  # Folder containing the PDFs
output_folder = "extracted_images"  # Folder to save the extracted images

extract_second_image_from_pdfs(pdf_folder, output_folder)
