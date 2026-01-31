
import json
import os
import sys
import urllib.request



def get_embedding(text, api_url, model, api_key):
    """Get embedding vector for text"""
    data = json.dumps({
        "model": model,
        "input": text
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{api_url}/v1/embeddings",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read())
        return result['data'][0]['embedding']

def main():
    # Get configuration from environment or config file
    api_url = os.getenv('EMBED_API')
    model = os.getenv('EMBED_MODEL')
    api_key = os.getenv('EMBED_KEY')

    # If not in environment, try loading from config.json
    if not all([api_url, model, api_key]):
        config_file = 'config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    api_url = api_url or config.get('EMBED_API')
                    model = model or config.get('EMBED_MODEL')
                    api_key = api_key or config.get('EMBED_KEY')
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error: Failed to read config.json: {e}")
                sys.exit(1)

    if not all([api_url, model, api_key]):
        print("Error: Missing required configuration")
        print("Set environment variables (EMBED_API, EMBED_MODEL, EMBED_KEY) or create config.json")
        print("\nExample config.json:")
        print('{')
        print('  "EMBED_API": "https://your-api-url.com",')
        print('  "EMBED_MODEL": "your-model-name",')
        print('  "EMBED_KEY": "your-api-key"')
        print('}')
        sys.exit(1)

    # Load documents
    with open('documents.json', 'r') as f:
        documents = json.load(f)

    print(f"Generating embeddings for {len(documents)} documents...")

    # Generate embeddings for each document
    for doc in documents:
        print(f"Processing: {doc['title']}")
        embedding = get_embedding(doc['content'], api_url, model, api_key)
        doc['embedding'] = embedding
        print(f"  ✓ Generated {len(embedding)}-dimensional embedding")

    # Save documents with embeddings
    with open('documents_with_embeddings.json', 'w') as f:
        json.dump(documents, f, indent=2)

    print(f"\n✓ Saved embeddings to documents_with_embeddings.json")

if __name__ == "__main__":
    main()
