import os
import sys
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("=" * 60)
print("Financial RAG Chatbot - Document Ingestion")
print("=" * 60)

# Check if data directory exists
data_dir = "./data/"
if not os.path.exists(data_dir):
    print(f"❌ Error: Data directory '{data_dir}' not found.")
    print("   Please create a './data/' directory and add PDF files to it.")
    sys.exit(1)

print("\n[STEP 1] Loading PDFs from ./data/...")
try:
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    if not docs:
        print("❌ Error: No PDF files found in ./data/")
        print("   Please add PDF files to the './data/' directory.")
        sys.exit(1)
    
    print(f"✅ Loaded {len(docs)} documents.")
except Exception as e:
    print(f"❌ Error loading PDFs: {e}")
    sys.exit(1)

print("\n[STEP 2] Splitting documents into chunks...")
try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"✅ Created {len(splits)} chunks.")
except Exception as e:
    print(f"❌ Error splitting documents: {e}")
    sys.exit(1)

print("\n[STEP 3] Initializing HuggingFace embeddings...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("✅ Embeddings model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")
    sys.exit(1)

print("\n[STEP 4] Creating ChromaDB vectorstore...")
db_path = "./chroma_db"
try:
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="financial_pdfs"
    )
    print(f"✅ ChromaDB created with persist_directory at {db_path}")
except Exception as e:
    if "persist_directory" in str(e) or "chroma_server_nofile" in str(e):
        print(f"⚠️  persist_directory parameter not fully supported. Creating without it...")
        try:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name="financial_pdfs"
            )
            print(f"✅ ChromaDB created successfully (in-memory).")
        except Exception as e2:
            print(f"❌ Error creating vectorstore: {e2}")
            sys.exit(1)
    else:
        print(f"❌ Error creating vectorstore: {e}")
        sys.exit(1)

print("\n[STEP 5] Persisting vectorstore...")
try:
    if hasattr(vectorstore, 'persist'):
        vectorstore.persist()
        print(f"✅ Vectorstore persisted to {db_path}")
    else:
        print("⚠️  persist() method not available (vectorstore is in-memory only)")
except Exception as e:
    print(f"⚠️  Persistence warning: {e}")

print("\n" + "=" * 60)
print("✅ INGESTION COMPLETE!")
print("=" * 60)
print(f"\nVectorstore ready with {len(splits)} chunks of financial documents.")
print(f"Collection: 'financial_pdfs'")
print(f"Location: {db_path}")
