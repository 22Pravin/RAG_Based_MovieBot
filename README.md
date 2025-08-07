Here is a comprehensive `.README.md` file for your **RAG-based MovieBot** project, based on the analysis of all the uploaded files:

---

```markdown
# 🎬 MovieBot — A RAG-Based Conversational Movie Assistant

MovieBot is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to ask natural language questions about movies and receive insightful, data-driven answers. It uses a combination of vector similarity search (via FAISS), semantic embeddings (via Sentence Transformers), and prompt-engineered LLM completions (via OpenAI) to deliver responses grounded in movie metadata and reviews.

---

## 🚀 Features

- 🔎 **Semantic Search**: Retrieves relevant movie documents using FAISS vector indexing.
- 🤖 **LLM Augmentation**: Leverages OpenAI's GPT models for natural language generation.
- 🗣️ **Chat Interface**: Simple command-line interaction via `app.py`.
- 🧠 **Modular Architecture**: Decoupled components for data loading, embedding, retrieval, and LLM interaction.

---

## 📂 Project Structure

```

.
├── app.py                    # CLI interface to talk to the bot
├── movie\_bot.py              # Orchestrator for embedding + LLM interaction
├── rag\_pipeline.py           # Core RAG logic
├── vector\_store.py           # FAISS index creation and storage
├── data\_preparation.py       # Load and process movie documents
├── prompt\_templates.py       # Prompt formatting for LLM
├── test.py                   # Basic unit tests
├── movie\_documents.json      # Raw movie data
├── processed\_movies.csv      # Preprocessed metadata
├── movies\_faiss\_index        # FAISS index binary
├── index\_to\_doc\_mapping.json# Index → Movie mapping
└── README.md                 # You are here

````

---

## 🧠 How It Works

### 1. **Data Preparation**
- Raw movie metadata is loaded from `movie_documents.json`.
- Processed into cleaned format via `data_preparation.py`.
- Each movie is represented as a document with `content` and `metadata`.

### 2. **Embedding & Indexing**
- `vector_store.py`:
  - Uses `all-MiniLM-L6-v2` from SentenceTransformers to embed movie documents.
  - Normalizes embeddings and indexes them using FAISS (cosine similarity).
  - Saves `movies_faiss_index` and `index_to_doc_mapping.json`.

### 3. **Query Pipeline**
- `rag_pipeline.py`:
  - Encodes user query into embedding.
  - Retrieves top-k most relevant movies from FAISS index.
  - Crafts a system + user prompt with movie snippets and forwards to GPT.

### 4. **Response Generation**
- `movie_bot.py` and `prompt_templates.py`:
  - Construct informative prompts combining retrieved movies.
  - Query OpenAI's GPT for a human-like answer.

### 5. **Interaction**
- `app.py`: Provides a simple CLI loop to interact with the chatbot.
- `test.py`: Unit tests the pipeline components.

---

## 💬 Example Interaction

```bash
> python app.py

Welcome to MovieBot 🎬
Ask me anything about movies!
(Type 'exit' to quit)

User: Recommend me some comedy movies from 1995.
Bot: Sure! Here are some comedy movies from 1995 you might enjoy:
1. Clueless — A fun, quotable classic.
2. Ace Ventura: When Nature Calls — Jim Carrey in his hilarious best.
3. Mighty Aphrodite — A Woody Allen film with romance and wit.
...
````

---

## 📦 Setup & Requirements

### 📋 Dependencies

Install using pip:

```bash
pip install -r requirements.txt
```

Requirements include:

* `openai`
* `faiss-cpu`
* `sentence-transformers`
* `numpy`
* `pandas`

### 🔑 OpenAI Key

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-key-here"  # On Unix
set OPENAI_API_KEY=your-key-here      # On Windows
```

---

## ⚙️ Running the Bot

### 1. **Index Documents**

```bash
python vector_store.py
```

Creates the FAISS index and saves it.

### 2. **Chat with MovieBot**

```bash
python app.py
```

Start a chat session via terminal.

---

## 🧪 Testing

```bash
python test.py
```

Tests vector loading, retrieval, and OpenAI call structure.

---

## 📊 Dataset

* Source: Synthetic movie metadata for \~100+ movies.
* Format: JSON and CSV.
* Fields: Title, Year, Genres, Rating, Tags, MovieID

---

## 🧱 RAG Design Breakdown

| Component            | Purpose                     |
| -------------------- | --------------------------- |
| Embeddings           | Convert text to vector form |
| Vector Store (FAISS) | Fast similarity search      |
| Retriever            | Select top relevant docs    |
| Prompt Template      | Format docs for GPT         |
| LLM (GPT)            | Generate final response     |

---

## 📈 Future Improvements

* Add UI via Streamlit or Flask
* Add support for movie trailers or images
* Implement chat history memory
* Expand dataset to modern films
* Replace OpenAI with open-source LLMs

---

## 🧑‍💻 Author

**Pravin Gohil**

---

## 🪪 License

MIT License — free to use and modify.
