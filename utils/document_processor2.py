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
                print(f"Warning: File {file_path} not found")
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
                documents.extend(docs)
                print(f"Loaded {len(docs)} documents from {file_path}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                        
        return documents

    def create_vectorstore(self, documents: List[Document], force_refresh: bool = False) -> Chroma:
        """Create vector store from documents"""
        if not documents:
            raise ValueError("No documents provided")
        
       
        if force_refresh:
            import shutil
            import time
            import tempfile
            
            # clearing the vectorstore object first
            if hasattr(self, 'vectorstore') and self.vectorstore:
                try:
                    del self.vectorstore
                except:
                    pass
            self.vectorstore = None            
            
            db_paths = ["./chroma_db", "./.chroma", "./chromadb"]
            for db_path in db_paths:
                if os.path.exists(db_path):
                    try:
                        shutil.rmtree(db_path)
                        print(f"🗑️ Removed {db_path}")
                    except Exception as e:
                        print(f"Warning: Could not remove {db_path}: {e}")
            
            # waiting a moment for file system to catch up
            time.sleep(0.1)
        
        # splitting documents into chunks
        texts = self.text_splitter.split_documents(documents)
        print(f"Split documents into {len(texts)} chunks")        
        
        total_content = sum(len(doc.page_content) for doc in texts)
        print(f"📊 Total content length: {total_content} characters")
        
        if total_content == 0:
            raise ValueError("No content found in documents")        
       
        import time
        import tempfile        
       
        temp_dir = tempfile.gettempdir()
        db_path = os.path.join(temp_dir, f"chroma_db_{int(time.time())}")
        
        print(f"🔧 Creating new vector database at: {db_path}")
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=db_path
        )
        
        print(f"✅ Vector store created successfully with {len(texts)} chunks")
        
        
        try:
            test_results = self.vectorstore.similarity_search("test", k=1)
            print(f"🧪 Vectorstore verification: Found {len(test_results)} test results")
        except Exception as e:
            print(f"⚠️ Vectorstore verification failed: {e}")
        
        return self.vectorstore

    # def clear_vectorstore(self):
    #     """Delete existing vector store - FIXED: Proper clearing"""
    #     import shutil
        
    #     try:
    #         # Clear the vectorstore object first
    #         if hasattr(self, 'vectorstore') and self.vectorstore:
    #             try:
    #                 del self.vectorstore
    #             except:
    #                 pass
            
    #         self.vectorstore = None
            
    #         # Remove all possible database directories
    #         db_paths = ["./chroma_db", "./.chroma", "./chromadb"]
    #         for db_path in db_paths:
    #             if os.path.exists(db_path):
    #                 try:
    #                     shutil.rmtree(db_path)
    #                     print(f"🗑️ Removed {db_path}")
    #                 except Exception as e:
    #                     print(f"Warning: Could not remove {db_path}: {e}")
            
    #         # Also try to remove temp directories
    #         import tempfile
    #         import glob
    #         temp_dir = tempfile.gettempdir()
    #         chroma_temp_dirs = glob.glob(os.path.join(temp_dir, "chroma_db_*"))
    #         for temp_db in chroma_temp_dirs:
    #             try:
    #                 shutil.rmtree(temp_db)
    #                 print(f"🗑️ Removed temp DB: {temp_db}")
    #             except:
    #                 pass
            
    #         print("✅ Vectorstore cleared completely")
            
    #     except Exception as e:
    #         print(f"Error clearing vectorstore: {e}")
        
    #     return None

    def load_existing_vectorstore(self) -> Chroma:
        """Load existing vector store"""        
        if os.path.exists("./chroma_db"):
            try:
                self.vectorstore = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=self.embeddings
                )
                print("✅ Loaded existing vectorstore")
                return self.vectorstore
            except Exception as e:
                print(f"Error loading existing vectorstore: {e}")
                raise FileNotFoundError(f"Could not load existing vector store: {e}")
        else:
            raise FileNotFoundError("No existing vector store found")


    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents - FIXED: Better error handling"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            print(f"🔍 Similarity search returned {len(results)} results for query: '{query[:50]}...'")
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []  

    
    def add_documents(self, file_paths: List[str]):
        """Add new documents to existing vector store"""
        try:
            documents = self.load_documents(file_paths)
            if not documents:
                print("No documents to add")
                return
                
            texts = self.text_splitter.split_documents(documents)
            
            if self.vectorstore is None:
                self.create_vectorstore(documents, force_refresh=True)
            else:
                self.vectorstore.add_documents(texts)
                print(f"Added {len(texts)} document chunks to existing vectorstore")
                
        except Exception as e:
            print(f"Error adding documents: {e}")

    
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