# FirstRagBot

RAG (Retrieval-Augmented Generation) Q&A bot that uses embeddings for document retrieval and a chat model for answers.

## Setup

1. Copy the example config to create your own config:
   ```bash
   cp config.json.example config.json
   ```

2. Edit `config.json` and add your API keys and endpoints:
   - **EMBED_KEY** – API key for the embedding service
   - **CHAT_KEY** – API key for the chat model
   - **EMBED_API** – Embedding API base URL
   - **CHAT_API** – Chat API base URL
   - **EMBED_MODEL** / **CHAT_MODEL** – Model names (examples are in the file)

Do not commit `config.json`; it contains secrets and is listed in `.gitignore`.

## Usage

- **Generate embeddings** from your documents: `python generate_embeddings.py`
- **Run Q&A**: `python rag_qa.py` (then ask questions at the prompt)
