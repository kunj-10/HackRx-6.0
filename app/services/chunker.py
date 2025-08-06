from typing import List
import tiktoken

def find_smart_boundary(chunk_text: str) -> str:
    if '\n\n' in chunk_text:
        last_paragraph = chunk_text.rfind('\n\n')
        if last_paragraph > len(chunk_text) * 0.3:
            return chunk_text[:last_paragraph].strip()
    
    sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    best_sentence_end = -1
    
    for ending in sentence_endings:
        pos = chunk_text.rfind(ending)
        if pos > best_sentence_end and pos > len(chunk_text) * 0.3:
            best_sentence_end = pos + len(ending) - 1
    
    if best_sentence_end > -1:
        return chunk_text[:best_sentence_end + 1].strip()
    
    last_space = chunk_text.rfind(' ')
    if last_space > len(chunk_text) * 0.3:
        return chunk_text[:last_space].strip()
    
    return chunk_text.strip()

def token_chunking(text: str, max_tokens: int = 1500, overlap_tokens: int = 50) -> List[str]:
    if not text.strip():
        return []
    
    encoding = tiktoken.get_encoding("cl100k_base")  # same tokenizer as OpenAI/Gemini-compatible
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text.strip()]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        chunk_tokens = tokens[start_idx:end_idx]
        chunk = encoding.decode(chunk_tokens)
        
        if end_idx < len(tokens):
            chunk = find_smart_boundary(chunk)
        
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        
        if end_idx >= len(tokens):
            break
            
        overlap_start = max(0, end_idx - overlap_tokens)
        start_idx = overlap_start if overlap_start > start_idx else end_idx

    return chunks

def cahrcter_chunking(text: str, chunk_size: int = 1000) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        if '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end)

    return chunks

if __name__ == "__main__":
    text = """The Gemini API offers text embedding models to generate embeddings for words, phrases, sentences, and code. These foundational embeddings power advanced NLP tasks such as semantic search, classification, and clustering, providing more accurate, context-aware results than keyword-based approaches.

Building Retrieval Augmented Generation (RAG) systems is a common use case for embeddings. Embeddings play a key role in significantly enhancing model outputs with improved factual accuracy, coherence, and contextual richness. They efficiently retrieve relevant information from knowledge bases, represented by embeddings, which are then passed as additional context in the input prompt to language models, guiding it to generate more informed and accurate responses.

To learn more about the available embedding model variants, see the Model versions section. For enterprise-grade applications and high-volume workloads, we suggest using embedding models on Vertex AI."""

    chunks = token_chunking(text)
    print(len(chunks))
    for chunk in chunks:
        print(chunk, end="\n -------------- \n")