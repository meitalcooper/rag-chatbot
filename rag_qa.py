#!/usr/bin/env python3
import json
import os
import sys
import urllib.request
import math

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

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

def find_relevant_docs(question, documents, api_url, model, api_key, top_k=2):
    """Find most relevant documents for a question"""
    # Get question embedding
    question_embedding = get_embedding(question, api_url, model, api_key)

    # Calculate similarity with each document
    similarities = []
    for doc in documents:
        similarity = cosine_similarity(question_embedding, doc['embedding'])
        similarities.append((doc, similarity))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top K documents
    return [(doc, score) for doc, score in similarities[:top_k]]

def contains_documented_error(user_response, documents):
    """Check if user response contains any documented error from the protocols"""
    user_lower = user_response.lower()
    
    # Known errors from the protocols
    documented_errors = [
        "disk does not exist",
        "entering emergency mode",
        "yad",
        "error: 2100",
        "error 2100",
        "2100 detection error"
    ]
    
    # Check if any documented error appears in the user's response
    for error in documented_errors:
        if error in user_lower:
            return True
    
    return False

def is_asking_for_error_message(response_text):
    """Check if the bot's response is asking for an error message"""
    response_lower = response_text.lower()
    error_question_indicators = [
        "what specific error",
        "error message",
        "what error",
        "specific error message",
        "do you see any of the following specific errors"
    ]
    
    for indicator in error_question_indicators:
        if indicator in response_lower:
            return True
    
    return False

def generate_answer(question, context, chat_api, chat_model, api_key, history=None):
    """Generate answer using context"""
    history = history or []
    conversation = "\n".join(
        f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
        for message in history[-12:]
    ) or "(No previous messages)"
    prompt = f"""You are a Level 1 IT Support Agent. Your role is to follow strict troubleshooting protocols and guide users through structured decision trees.

CRITICAL INSTRUCTIONS:
1. **Strict Protocol Adherence**: If the Context contains a step-by-step workflow or decision tree (with IF/THEN logic, conditional questions, or sequential steps), you MUST follow that protocol EXACTLY. Do not skip steps or provide solutions out of order.

2. **Ask Questions First and WAIT**: If the protocol requires user input (e.g., "Ask if the laptop is in warranty", "Ask for the laptop type", or "What specific error message do you see?"), you MUST stop and ask ONLY that question. Do NOT provide any troubleshooting steps or solutions until the user responds. Your response should contain ONLY the question, nothing else. Wait for the user's answer before proceeding.

3. **Follow Decision Trees**: When the protocol contains IF/THEN statements (e.g., "IF LENOVO:", "IF YES (In Warranty):", "IF NO (Out of Warranty):", "IF ERROR is..."), you must follow the appropriate branch based on the user's response. Wait for the user's answer before proceeding to the next step.

4. **Preserve Links Exactly**: When the protocol contains URLs or links, copy them EXACTLY as they appear in the Context. Do not modify, shorten, or paraphrase URLs. Always provide the specific links found in the context.

5. **Handle Unknown Errors**: If the user reports an error that is not explicitly covered in the Context, check the protocol for instructions on handling unknown cases (e.g., "IF ERROR is not in the documents"). Follow those instructions exactly, such as asking the user if they see any of the specific errors listed in the protocol.

6. **Sequential Execution**: Follow the protocol steps in the order they appear. Do not jump ahead or combine steps unless the protocol explicitly allows it.

Context:
{context}

Previous Conversation:
{conversation}

User's Question/Issue: {question}

Provide your response as a Level 1 IT Support Agent following the protocol above:"""

    data = json.dumps({
        "model": chat_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{chat_api}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read())
        return result['choices'][0]['message']['content']

def main():
    # Get configuration from environment or config file
    embed_api = os.getenv('EMBED_API')
    embed_model = os.getenv('EMBED_MODEL')
    embed_key = os.getenv('EMBED_KEY')
    chat_api = os.getenv('CHAT_API')
    chat_model = os.getenv('CHAT_MODEL')
    chat_key = os.getenv('CHAT_KEY')

    # If not in environment, try loading from config.json
    if not all([embed_api, embed_model, embed_key, chat_api, chat_model, chat_key]):
        config_file = 'config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    embed_api = embed_api or config.get('EMBED_API')
                    embed_model = embed_model or config.get('EMBED_MODEL')
                    embed_key = embed_key or config.get('EMBED_KEY')
                    chat_api = chat_api or config.get('CHAT_API')
                    chat_model = chat_model or config.get('CHAT_MODEL')
                    chat_key = chat_key or config.get('CHAT_KEY')
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error: Failed to read config.json: {e}")
                sys.exit(1)

    if not all([embed_api, embed_model, embed_key, chat_api, chat_model, chat_key]):
        print("Error: Missing required configuration")
        print("Set environment variables (EMBED_API, EMBED_MODEL, EMBED_KEY, CHAT_API, CHAT_MODEL, CHAT_KEY) or create config.json")
        sys.exit(1)

    # Load documents with embeddings
    try:
        with open('documents_with_embeddings.json', 'r') as f:
            documents = json.load(f)
    except FileNotFoundError:
        print("Error: Run generate_embeddings.py first")
        sys.exit(1)

    print("=== RAG Document Q&A System ===")
    print(f"Loaded {len(documents)} documents with embeddings\n")

    # Conversation state tracking
    waiting_for_error = False
    conversation_history = []

    # Interactive Q&A loop
    while True:
        question = input("You: ").strip()

        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not question:
            continue

        # If we're waiting for an error message, check if the response contains a documented error
        if waiting_for_error:
            if contains_documented_error(question, documents):
                # User provided a documented error, proceed with troubleshooting
                waiting_for_error = False
                # Continue to process the error with full context
            else:
                # User response doesn't contain a documented error, ask again
                print("AI: I need to identify the specific error to help you. Please tell me: Do you see any of the following specific errors: YAD, Disk does not exist, Entering emergency mode, or Error 2100?\n")
                continue

        # Find relevant documents
        previous_user_turns = [
            message["content"] for message in conversation_history[-12:]
            if message["role"] == "user"
        ]
        retrieval_question = " ".join(previous_user_turns[-3:] + [question])
        relevant_docs = find_relevant_docs(
            retrieval_question, documents, embed_api, embed_model, embed_key
        )

        # Build context from relevant documents
        context_parts = []
        for doc, score in relevant_docs:
            context_parts.append(f"{doc['title']}: {doc['content']}")
            print(f"[Found relevant context: {doc['title']} (similarity: {score:.3f})]")

        context = "\n\n".join(context_parts)

        # Generate answer
        answer = generate_answer(
            question, context, chat_api, chat_model, chat_key,
            history=conversation_history
        )

        print(f"AI: {answer}\n")
        conversation_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        conversation_history = conversation_history[-12:]

        # Check if the bot's response is asking for an error message
        if is_asking_for_error_message(answer):
            # Check if user's question already contains a documented error
            if not contains_documented_error(question, documents):
                # We're waiting for the user to provide an error message
                waiting_for_error = True
            else:
                # User already provided a documented error, we can proceed
                waiting_for_error = False
        else:
            # Not asking for error, reset waiting state
            waiting_for_error = False

if __name__ == "__main__":
    main()
