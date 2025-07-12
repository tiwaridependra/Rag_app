# rag_utils.py

import faiss
import numpy as np
from PyPDF2 import PdfReader
from typing import List
import io

# Environment loader
from dotenv import load_dotenv
load_dotenv()

# LangChain - Updated imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_experimental.text_splitter import SemanticChunker


class RAGPipeline:
    def __init__(self):
        self.embedder = OpenAIEmbeddings()
        self.llm = ChatOpenAI( model="gpt-3.5-turbo", temperature=0)
        self.memory = ConversationBufferMemory(return_messages=True)
        self.index = None
        self.text_chunks = []

    def read_pdf(self, pdf_file) -> str:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text  

 
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 80) -> List[str]:
        # splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=overlap,
        #     separators=["\n\n", "\n", ".", " ", ""]
        # )
        splitter = SemanticChunker(
            embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
            breakpoint_threshold_type="percentile",  # or "standard_deviation"
            breakpoint_threshold_amount=95  # try 90–98 to control granularity
        )
        return splitter.split_text(text)

    def build_index(self, chunks: List[str]):
        self.text_chunks = chunks
        embeddings = self.embedder.embed_documents(chunks)
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self.embedder.embed_query(query)
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), top_k)
        return [self.text_chunks[i] for i in I[0]]

    def chat(self, query: str) -> str:
        context_chunks = self.retrieve(query)
        context = "\n\n".join(context_chunks)

        messages = [
            SystemMessage(content="You are a helpful assistant. Use provided context to answer."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]

        # Add memory/history messages if available
        past_messages = self.memory.chat_memory.messages
        full_convo = past_messages + messages

        response = self.llm(full_convo)
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response.content)
        return response.content.strip()