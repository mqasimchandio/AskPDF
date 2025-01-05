{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install langchain langchain_community langchain_huggingface faiss-cpu python-dotenv\n",
    "# !pip install -U langchain-huggingface\n",
    "# !pip install python-dotenv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import torch\n",
    "from langchain.chains import RetrievalQA\n",
    "from langchain.vectorstores import FAISS\n",
    "from langchain.prompts import PromptTemplate\n",
    "from langchain.llms import HuggingFacePipeline\n",
    "from langchain.embeddings import HuggingFaceEmbeddings\n",
    "from langchain.chains.question_answering import load_qa_chain\n",
    "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline\n",
    "from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint\n",
    "from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [],
   "source": [
    "DATA_PATH = \"PDF/\"\n",
    "def load_pdf_files(data):\n",
    "  loader = DirectoryLoader(data,\n",
    "                           glob='*.pdf',\n",
    "                           loader_cls = PyPDFLoader)\n",
    "  documents = loader.load()\n",
    "  return documents\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [],
   "source": [
    "documents = load_pdf_files(DATA_PATH)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [],
   "source": [
    "# print(\"LENGTH: \", len(documents))\n",
    "# documents[10].page_content"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_chunks(documents):\n",
    "  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,\n",
    "                                                 chunk_overlap=50)\n",
    "  text_chunks=text_splitter.split_documents(documents)\n",
    "  return text_chunks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Length of text_chunks:  376\n"
     ]
    }
   ],
   "source": [
    "text_chunks = create_chunks(documents)\n",
    "print(\"Length of text_chunks: \", len(text_chunks))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_embeddin_model():\n",
    "  embedding_model = HuggingFaceEmbeddings(model_name=\"sentence-transformers/all-MiniLM-L6-v2\")\n",
    "  return embedding_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {},
   "outputs": [],
   "source": [
    "embedding_model = get_embeddin_model()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [],
   "source": [
    "DB_FAISS_PATH = \"vectorstore/db_faiss\"\n",
    "db = FAISS.from_documents(text_chunks, embedding_model)\n",
    "db.save_local(DB_FAISS_PATH)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Defining constants\n",
    "HF_TOKEN = \"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\"\n",
    "HUGGINGFACE_REPO_ID = \"mistralai/Mistral-7B-Instruct-v0.2\"\n",
    "DB_FAISS_PATH = \"vectorstore/db_faiss\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading the LLM\n",
    "def load_llm(repo_id, token):\n",
    "    llm = HuggingFaceEndpoint(\n",
    "        repo_id=repo_id,\n",
    "        temperature=0.8,\n",
    "        model_kwargs={\"token\": token, \"max_length\": 512}\n",
    "    )\n",
    "    return llm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define custom prompt\n",
    "CUSTOM_PROMPT_TEMPLATE = \"\"\"\n",
    "Utilize the information provided in the context to accurately respond to the user's question. \n",
    "If the answer is not present in the context, simply state that you do not know the answer; do not fabricate responses.\n",
    "\n",
    "Context: {context}\n",
    "Question: {question}\n",
    "\"\"\"\n",
    "def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):\n",
    "    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=[\"context\", \"question\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize components\n",
    "llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)\n",
    "embedding_model = HuggingFaceEmbeddings(model_name=\"sentence-transformers/all-MiniLM-L6-v2\")\n",
    "db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create the QA chain\n",
    "qa_chain = RetrievalQA.from_chain_type(\n",
    "    llm=llm,\n",
    "    chain_type=\"stuff\",\n",
    "    retriever=db.as_retriever(search_kwargs={\"k\": 3}),\n",
    "    return_source_documents=True,\n",
    "    chain_type_kwargs={\n",
    "        \"prompt\": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE),\n",
    "        \"document_variable_name\": \"context\",  # This matches the prompt's input variable\n",
    "    }\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Han bol:  what is deep learning?\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Result:  Answer: Deep learning is not related to the context provided. It is a subcategory of machine learning which uses neural networks with three or more layers to model high-level abstractions in data.\n"
     ]
    }
   ],
   "source": [
    "user_query = input(\"Han bol: \")\n",
    "response = qa_chain.invoke({'query': user_query})\n",
    "print(\"Result: \", response[\"result\"])\n",
    "# print(\"SourceDOCS: \", response[\"source_documents\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
