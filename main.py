import os
from tkinter import Tk, filedialog
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Ensure the libraries are installed. Uncomment and run these lines if needed:
# os.system('pip install langchain')
# os.system('pip install openai')
# os.system('pip install PyPDF2')
# os.system('pip install faiss-cpu')
# os.system('pip install tiktoken')

# Set API keys
os.environ["OPENAI_API_KEY"] = "api-key"

# Function to browse and select a PDF file
def browse_pdf():
  root = Tk()
  root.withdraw()  # Hide the root window
  pdf_path = filedialog.askopenfilename(
      title="Select PDF File",
      filetypes=(("PDF Files", "*.pdf"), ("All Files", "*.*"))
  )
  root.destroy()  # Destroy the root window
  return pdf_path

# Read text from PDF
def read_pdf(pdf_path):
  pdfreader = PdfReader(pdf_path)
  raw_text = ''
  for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
      raw_text += content
  return raw_text

# Main function
def main():
  pdf_path = browse_pdf()
  if not pdf_path:
    print("No PDF file selected.")
    return

  raw_text = read_pdf(pdf_path)

  # Split the text using CharacterTextSplitter to ensure token size is managed
  text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=800,
      chunk_overlap=200,
      length_function=len,
  )
  texts = text_splitter.split_text(raw_text)

  # Download embeddings from OpenAI
  embeddings = OpenAIEmbeddings()

  # Create a FAISS vector store from texts and embeddings
  document_search = FAISS.from_texts(texts, embeddings)

  # Load the QA chain
  chain = load_qa_chain(OpenAI(), chain_type="stuff")

  # Get user query
  query = input("Enter your query: ")

  # Perform a similarity search and run the QA chain
  docs = document_search.similarity_search(query)
  answer = chain.run(input_documents=docs, question=query)

  print("Answer:", answer)

if __name__ == "__main__":
  main()
