import math
import os

from langsmith import traceable

# Check if we should use local embeddings
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

if USE_LOCAL_EMBEDDINGS:
    from sentence_transformers import SentenceTransformer
    # Use a good code embedding model
    _local_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Using local embeddings (sentence-transformers)")
else:
    from google import genai
    from codebloatguard.config import EMBED_BATCH, EMBED_MODEL, GEMINI_API_KEY
    _client = genai.Client(api_key=GEMINI_API_KEY)
    print("Using Gemini embeddings")

@traceable(name="embed", run_type="embedding")
def embed(texts: list[str]) -> list[list[float]]:
    if USE_LOCAL_EMBEDDINGS:
        # Local embedding - no API calls
        embeddings = _local_model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
    else:
        # Gemini API embedding
        from codebloatguard.llm import with_backoff
        from codebloatguard.config import EMBED_BATCH
        
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            r = with_backoff(
                lambda: _client.models.embed_content(model=EMBED_MODEL, contents=batch)
            )
            vectors.extend(e.values for e in r.embeddings)
        return vectors

def embed_one(text: str) -> list[float]:
    return embed([text])[0]

def cosine_distance(a: list[float], b: list[float]) -> float:
    """
    Computes the semantic angle between text embeddings
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 1.0 - dot / (na * nb) if na and nb else 1.0
