import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(key))
            if key == "name":
                num_tokens += tokens_per_name
            elif key == "content":
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                elif isinstance(value, list):
                    for content in value:
                        if isinstance(content, dict) and content.get("type") == "image_url":
                            # Add a fixed number of tokens for image URL
                            num_tokens += 100  # Adjust this number as needed
                        elif isinstance(content, str):
                            num_tokens += len(encoding.encode(content))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens