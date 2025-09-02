from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfReader

def combine_pdfs_vertically(pdf1_path, pdf2_path, output_path):
    # Open the first PDF
    reader1 = PdfReader(pdf1_path)
    page1 = reader1.pages[0]
    
    # Open the second PDF
    reader2 = PdfReader(pdf2_path)
    page2 = reader2.pages[0]
    
    # Create a new PDF with ReportLab
    c = canvas.Canvas(output_path, pagesize=letter)
    
    # Get the dimensions of the first page
    width = page1.mediabox.getWidth()
    height1 = page1.mediabox.getHeight()
    height2 = page2.mediabox.getHeight()
    
    # Draw the first page
    c.drawImage(pdf1_path, 0, height2, width=width, height=height1)
    
    # Draw the second page below the first page
    c.drawImage(pdf2_path, 0, 0, width=width, height=height2)
    
    # Save the new PDF
    c.save()

# Paths to the input PDFs and the output PDF
pdf1_path = 'findings3_1.pdf'
pdf2_path = 'findings3_2.pdf'
output_path = 'combined.pdf'

# Combine the PDFs
combine_pdfs_vertically(pdf1_path, pdf2_path, output_path)