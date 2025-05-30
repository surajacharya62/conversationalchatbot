import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document


class DocumentProcessor:
    def __init__(self, google_api_key: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore = None
    

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from various file types"""
        documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found")
                continue 
                
            file_extension = os.path.splitext(file_path)[1].lower()
            
            try:
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_extension == '.docx':
                    loader = Docx2txtLoader(file_path)
                elif file_extension == '.txt':
                    loader = TextLoader(file_path)
                else:
                    print(f"Unsupported file type: {file_extension}")
                    continue
                
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} documents from {file_path}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                        
        return documents
    

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create vector store from documents"""
        if not documents:
            raise ValueError("No documents provided")
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        print(f"Split documents into {len(texts)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return self.vectorstore
    

    def clear_vectorstore(self):
        """Delete existing vector store"""
           
        # Clear the in-memory reference
        # print("Vectorstore cleared completely")
        self.vectorstore = None
        return self.vectorstore
        
        # Delete the actual database files
        # self.clear_vectorstore()  # This should delete ./chroma_db        
        

    def load_existing_vectorstore(self) -> Chroma:
        """Load existing vector store"""        

        if os.path.exists("./chroma_db"):
            self.vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings
            )
            return self.vectorstore
        else:
            raise FileNotFoundError("No existing vector store found")
    
    # def add_documents(self, file_paths: List[str]):
    #     """Add new documents to existing vector store"""
    #     documents = self.load_documents(file_paths)
    #     texts = self.text_splitter.split_documents(documents)
        
    #     if self.vectorstore is None:
    #         self.create_vectorstore(documents)
    #     else:
    #         self.vectorstore.add_documents(texts)
    
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)