
# AskPDF (PDF Question Answering System with LangChain and HuggingFace ) 

This project demonstrates how to build a **Question Answering (QA)** system using **LangChain**, **HuggingFace models**, and **FAISS vector database** to retrieve relevant information from PDF documents.

---

## Features  
- Loads and processes multiple PDF files.  
- Splits PDF content into manageable text chunks.  
- Embeds text using **HuggingFace sentence-transformers**.  
- Stores embeddings in a FAISS vector database for efficient retrieval.  
- Utilizes a custom prompt to provide accurate answers.  
- Supports querying via a **HuggingFace LLM (Mistral-7B-Instruct)** for natural language responses.

---

## Requirements  

Ensure you have the following installed:  

- Python 3.8+  
- `torch`  
- `langchain`  
- `transformers`  
- `faiss-cpu`  
- `langchain_huggingface`  
- `langchain_community`  
- `sentence-transformers`  

---

## Setup  

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/your-username/your-repo.git  
   cd your-repo  
   ```  

2. **Install Dependencies**  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Prepare Your Data**  
   Place your PDF files in the `PDF/` directory.

4. **Set Your HuggingFace Token**  
   Replace `HF_TOKEN` in the script with your actual HuggingFace token.

5. **Run the Program**  
   ```bash  
   python main.py  
   ```  

---

## How It Works  

1. **Load PDF Documents:**  
   Reads all PDF files from the `PDF/` directory using `DirectoryLoader`.  

2. **Text Splitting:**  
   Splits the content into smaller chunks for efficient processing.  

3. **Embedding and Vector Store:**  
   Embeds text chunks using a pre-trained **sentence-transformer** and stores them in a FAISS vector database.  

4. **Custom Prompt & Query:**  
   Defines a custom prompt to ensure accurate answers. User queries are processed through the QA system.

5. **Output:**  
   Returns the answer and relevant source documents.  

---

## Example  

```text  
Enter your query: What is the main topic of the document?  
Result: The document discusses the implementation of machine learning models.  
```  

---

## Notes  

- Replace sensitive information like HuggingFace tokens and database paths as needed.  
- Ensure all required Python packages are installed.  
