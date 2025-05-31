import os
from typing import List, Dict, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import tempfile
import time
import shutil


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
        """Load documents from various file types with enhanced error handling"""
        documents = []
        successful_loads = 0
        
        print(f"🔧 Attempting to load {len(file_paths)} files...")
        
        for i, file_path in enumerate(file_paths):
            print(f"\n📄 Processing file {i+1}/{len(file_paths)}: {file_path}")
            
            # Check if file exists and is accessible
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            
            try:
                file_size = os.path.getsize(file_path)
                print(f"📊 File size: {file_size} bytes")
                
                if file_size == 0:
                    print(f"⚠️ File is empty: {file_path}")
                    continue
                    
            except Exception as e:
                print(f"❌ Cannot access file stats: {e}")
                continue
                
            file_extension = os.path.splitext(file_path)[1].lower()
            print(f"📋 File extension: {file_extension}")
            
            try:
                docs = []
                
                if file_extension == '.pdf':
                    print("🔄 Loading PDF...")
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                elif file_extension == '.docx':
                    print("🔄 Loading DOCX...")
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                    
                elif file_extension == '.txt':
                    print("🔄 Loading TXT...")
                    # Try multiple encodings for text files
                    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
                    loaded = False
                    
                    for encoding in encodings:
                        try:
                            loader = TextLoader(file_path, encoding=encoding)
                            docs = loader.load()
                            print(f"✅ Successfully loaded with {encoding} encoding")
                            loaded = True
                            break
                        except Exception as encoding_error:
                            print(f"⚠️ Failed with {encoding}: {encoding_error}")
                            continue
                    
                    if not loaded:
                        print(f"❌ Could not load text file with any encoding")
                        continue
                        
                else:
                    print(f"❌ Unsupported file type: {file_extension}")
                    continue
                
                if docs:
                    # Validate document content
                    valid_docs = []
                    for doc in docs:
                        if doc.page_content and doc.page_content.strip():
                            valid_docs.append(doc)
                            print(f"✅ Valid document found: {len(doc.page_content)} characters")
                        else:
                            print(f"⚠️ Empty document content found")
                    
                    if valid_docs:
                        documents.extend(valid_docs)
                        successful_loads += 1
                        print(f"✅ Loaded {len(valid_docs)} valid documents from {os.path.basename(file_path)}")
                    else:
                        print(f"❌ No valid content found in {file_path}")
                else:
                    print(f"❌ No documents returned from loader for {file_path}")
                    
            except Exception as e:
                print(f"❌ Error loading {file_path}: {type(e).__name__}: {e}")
                import traceback
                print(f"📋 Traceback: {traceback.format_exc()}")
                continue
        
        print(f"\n📊 Final Results:")
        print(f"   • Files attempted: {len(file_paths)}")
        print(f"   • Files successfully loaded: {successful_loads}")
        print(f"   • Total documents: {len(documents)}")
        
        if documents:
            total_chars = sum(len(doc.page_content) for doc in documents)
            print(f"   • Total content: {total_chars} characters")
            
            # Show sample content
            if documents[0].page_content:
                sample = documents[0].page_content[:200]
                print(f"   • Sample content: {sample}...")
        
        if not documents:
            raise ValueError(f"No documents could be loaded from {len(file_paths)} files. Check file formats and content.")
                        
        return documents

    def create_vectorstore(self, documents: List[Document], force_refresh: bool = False) -> Chroma:
        """Create vector store from documents with enhanced error handling"""
        if not documents:
            raise ValueError("No documents provided to create vectorstore")
        
        print(f"🔧 Creating vectorstore from {len(documents)} documents...")
        
        # Clear existing vectorstore if force refresh
        if force_refresh:
            self._clear_vectorstore()
        
        try:
            # Split documents into chunks
            print("📝 Splitting documents into chunks...")
            texts = self.text_splitter.split_documents(documents)
            print(f"✅ Split documents into {len(texts)} chunks")
            
            if not texts:
                raise ValueError("Document splitting produced no text chunks")
            
            # Validate chunks have content
            valid_chunks = [chunk for chunk in texts if chunk.page_content.strip()]
            if not valid_chunks:
                raise ValueError("No valid content chunks after splitting")
            
            total_content = sum(len(chunk.page_content) for chunk in valid_chunks)
            print(f"📊 Total content length: {total_content} characters")
            print(f"📊 Valid chunks: {len(valid_chunks)}/{len(texts)}")
            
            if total_content < 10:  # Minimum content threshold
                raise ValueError(f"Insufficient content: only {total_content} characters")
            
            # Create vectorstore in a more reliable location
            try:
                # Try current directory first
                db_path = "./chroma_db"
                print(f"🔧 Attempting to create vectorstore at: {db_path}")
                
                self.vectorstore = Chroma.from_documents(
                    documents=valid_chunks,
                    embedding=self.embeddings,
                    persist_directory=db_path
                )
                
            except Exception as e:
                print(f"⚠️ Failed to create in current directory: {e}")
                print("🔧 Trying temporary directory...")
                
                # Fallback to temp directory
                temp_dir = tempfile.gettempdir()
                db_path = os.path.join(temp_dir, f"chroma_db_{int(time.time())}")
                
                self.vectorstore = Chroma.from_documents(
                    documents=valid_chunks,
                    embedding=self.embeddings,
                    persist_directory=db_path
                )
            
            print(f"✅ Vector store created successfully with {len(valid_chunks)} chunks")
            
            # Verify vectorstore works
            try:
                test_results = self.vectorstore.similarity_search("test", k=1)
                if test_results:
                    print(f"🧪 Vectorstore verification: Found {len(test_results)} test results")
                    print(f"🧪 Sample result: {test_results[0].page_content[:100]}...")
                else:
                    print("⚠️ Vectorstore verification: No results found (but this might be normal)")
            except Exception as e:
                print(f"⚠️ Vectorstore verification failed: {e}")
                # Don't raise error here, vectorstore might still work
            
            return self.vectorstore
            
        except Exception as e:
            print(f"❌ Failed to create vectorstore: {type(e).__name__}: {e}")
            import traceback
            print(f"📋 Full traceback: {traceback.format_exc()}")
            raise e

    def _clear_vectorstore(self):
        """Clear existing vectorstore"""
        try:
            # Clear the vectorstore object first
            if hasattr(self, 'vectorstore') and self.vectorstore:
                try:
                    del self.vectorstore
                except:
                    pass
            
            self.vectorstore = None
            
            # Remove database directories
            db_paths = ["./chroma_db", "./.chroma", "./chromadb"]
            for db_path in db_paths:
                if os.path.exists(db_path):
                    try:
                        shutil.rmtree(db_path)
                        print(f"🗑️ Removed {db_path}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {db_path}: {e}")
            
            # Clean up temp directories
            try:
                import glob
                temp_dir = tempfile.gettempdir()
                chroma_temp_dirs = glob.glob(os.path.join(temp_dir, "chroma_db_*"))
                for temp_db in chroma_temp_dirs:
                    try:
                        shutil.rmtree(temp_db)
                        print(f"🗑️ Removed temp DB: {temp_db}")
                    except:
                        pass
            except:
                pass
            
            print("✅ Vectorstore cleared")
            
        except Exception as e:
            print(f"⚠️ Error clearing vectorstore: {e}")

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
                print(f"❌ Error loading existing vectorstore: {e}")
                raise FileNotFoundError(f"Could not load existing vector store: {e}")
        else:
            raise FileNotFoundError("No existing vector store found")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents with better error handling"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized - please upload documents first")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            print(f"🔍 Similarity search returned {len(results)} results for query: '{query[:50]}...'")
            return results
        except Exception as e:
            print(f"❌ Error in similarity search: {type(e).__name__}: {e}")
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