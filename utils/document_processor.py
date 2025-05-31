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
            chunk_size=500,  
            chunk_overlap=50,  
            length_function=len,
        )
        self.vectorstore = None

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from various file types"""
        documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue 
                
            file_extension = os.path.splitext(file_path)[1].lower()
            
            try:
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_extension == '.docx':
                    loader = Docx2txtLoader(file_path)
                elif file_extension == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                else:
                    print(f"Unsupported file type: {file_extension}")
                    continue
                
                docs = loader.load()
                
                # Fix: Filter out empty documents and combine all content
                valid_content = []
                for doc in docs:
                    if doc.page_content and doc.page_content.strip():
                        valid_content.append(doc.page_content.strip())
                
                if valid_content:
                    # Create a single document with all content
                    combined_content = "\n\n".join(valid_content)
                    combined_doc = Document(
                        page_content=combined_content,
                        metadata={"source": file_path}
                    )
                    documents.append(combined_doc)
                    print(f"Loaded document with {len(combined_content)} characters")
                else:
                    print(f"No content found in {file_path}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                        
        if not documents:
            raise ValueError("No documents could be loaded")
            
        return documents

    def create_vectorstore(self, documents: List[Document], force_refresh: bool = False) -> Chroma:
        """Create vector store from documents"""
        if not documents:
            raise ValueError("No documents provided")
        
        if force_refresh:
            self._clear_vectorstore()
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks")        
        
        if not texts:
            raise ValueError("No text chunks created")
       
        # Create vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"Vector store created with {len(texts)} chunks")
        return self.vectorstore

    def _clear_vectorstore(self):
        """Clear existing vectorstore"""
        try:
            if hasattr(self, 'vectorstore') and self.vectorstore:
                del self.vectorstore
            self.vectorstore = None
            
            import shutil
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                print("Cleared old vectorstore")
        except Exception as e:
            print(f"Error clearing vectorstore: {e}")

    def load_existing_vectorstore(self) -> Chroma:
        """Load existing vector store"""        
        if os.path.exists("./chroma_db"):
            try:
                self.vectorstore = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=self.embeddings
                )
                return self.vectorstore
            except Exception as e:
                raise FileNotFoundError(f"Could not load existing vector store: {e}")
        else:
            raise FileNotFoundError("No existing vector store found")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []  

    def get_vectorstore_info(self) -> dict:
        """Get information about the current vectorstore"""
        info = {
            "vectorstore_exists": self.vectorstore is not None,
            "embeddings_model": "models/embedding-001",
            "chunk_size": self.text_splitter._chunk_size,
            "chunk_overlap": self.text_splitter._chunk_overlap
        }
        
        if self.vectorstore:
            try:                
                test_search = self.vectorstore.similarity_search("", k=1)
                info["has_documents"] = len(test_search) > 0
                info["sample_content"] = test_search[0].page_content[:100] if test_search else "No content"
            except Exception as e:
                info["error"] = str(e)
        
        return info