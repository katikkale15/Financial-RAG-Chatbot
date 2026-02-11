# 💰 Financial RAG Chatbot


**100% Local RAG chatbot** for financial PDFs using **LangChain + Ollama + ChromaDB + Streamlit**. Upload PDFs dynamically, chat instantly—no cloud costs!

## ✨ Features

- 🧠 **Retrieval-Augmented Generation (RAG)**: PDF → Chunks → Embeddings → ChromaDB → Ollama
- 📤 **Dynamic PDF Upload**: Add new documents on-the-fly (appends to existing DB)
- 💬 **Interactive Chat UI**: Streamlit with conversation history
- 🚀 **Fully Offline**: Local LLM (llama3.2) + sentence-transformers embeddings
- 📊 **Financial-Optimized**: Handles balance sheets, P&Ls, 10-Ks, tables
- ⚡ **Incremental Updates**: No full re-indexing when adding new PDFs

## 🎯 Live Demo

```
Ask: "What is Q1 revenue?" → "Revenue $10.2M from Balance Sheet (Page 5)"
Ask: "Define EBITDA?" → "Earnings Before Interest, Taxes... (General knowledge)"
```

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.12 (3.14 doesnt work well with all the langchain dependencies)
- Ollama installed

### 1. Clone & Setup
```bash
git clone https://github.com/[YOUR_USERNAME]/Financial-RAG-Chatbot.git
cd Financial-RAG-Chatbot
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama (New Terminal)
```bash
ollama pull llama3.2
ollama serve
```

### 3. Add Your PDFs
```bash
mkdir data
# Copy financial PDFs to data/ folder
python ingest.py  # Build ChromaDB (1st time only)
```

### 4. Launch Chatbot
```bash
streamlit run app.py
```

**Open**: http://localhost:8501 → Start chatting! 🎉

## 📁 Project Structure

```
Financial-RAG-Chatbot/
├── app.py              # 🗣️ Streamlit chatbot + PDF upload
├── ingest.py           # 📚 PDF → ChromaDB pipeline
├── requirements.txt    # 📦 Dependencies
├── README.md           # 📖 You're reading it!
├── .env.example        # 🔑 Environment template
├── .gitignore          # 🗑️ Git ignores
├── data/               # 📁 Sample PDFs (add yours)
└── chroma_db/          # 🧠 Vector DB (auto-created)
```

## 🏗️ How It Works

```
1. PDFs ──PyPDFLoader──> Raw Docs ──Splitter──> Chunks
   ↓
2. Chunks ──HuggingFace──> Embeddings ──ChromaDB──> Vector Store
   ↓
3. Query ──Retriever──> Top-5 Chunks ──Prompt──> Ollama ──> Answer
```

**RAG Pipeline**: `Load → Split → Embed → Store → Retrieve → Generate`

## 🔧 Customization

### 🔄 Change LLM Model
```python
# app.py, line with OllamaLLM
llm = OllamaLLM(model="llama3.1:8b")  # Larger model
# or
llm = OllamaLLM(model="gemma2:9b")    # Alternative
```

### 📏 Adjust Chunking
```python
# ingest.py
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,     # Bigger chunks
    chunk_overlap=300    # More context overlap
)
```

### 🎯 Better Embeddings
```python
# app.py / ingest.py
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Higher quality
)
```

## 🌐 Deployment Options

### Streamlit Cloud (Free)
1. Fork repo → Connect to Streamlit Cloud
2. Deploy → Done! (Needs Ollama Cloud or local hosting)


## 🧪 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: langchain.chains` | `pip install -r requirements.txt --upgrade` |
| `Ollama not found` | `ollama serve` in separate terminal |
| `Port 11434 bind error` | `taskkill /F /IM ollama.exe` (Windows) |
| Slow responses | Use `llama3.2:1b` (smaller) or GPU |

## 📈 Performance

```
Model: llama3.2 (3B)
Response Time: ~2-5s/query
Context Window: 4096 tokens
Max PDFs: 1000+ (scales linearly)
RAM: 8GB minimum
```

## 🤝 Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request 🎉

## 🙏 Acknowledgments

- **LangChain** - Orchestration framework
- **Ollama** - Local LLM serving
- **ChromaDB** - Vector database
- **Streamlit** - Amazing UI framework
- **HuggingFace** - Sentence transformers
- **Perplexity AI** - Code guidance inspiration

