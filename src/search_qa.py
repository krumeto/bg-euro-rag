import json
import numpy as np
from sklearn.metrics import pairwise_distances
from model2vec import StaticModel

def _perform_search(query: str, embeddings_path: str, texts_path: str, top_k: int) -> str:
    embeddings = np.load(embeddings_path)
    
    with open(texts_path, "r", encoding="utf-8") as f:
        all_texts = json.load(f)

    model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
    query_embedding = model.encode(query)[None, :]
    distances = pairwise_distances(query_embedding, embeddings, metric="cosine")[0]
    
    sorted_indices = np.argsort(distances)
    top_indices = sorted_indices[:top_k]
    top_scores = [1.0 - distances[i] for i in top_indices]

    lines = []
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        block = (
            f"{rank}. (Score: {score:.4f})\n"
            f"{all_texts[idx]}\n"
            f"{'-' * 40}"
        )
        lines.append(block)

    return "\n\n".join(lines)

def search_qa(query: str) -> str:
    """
    Finds the most similar items to `query` and returns a single formatted string.

    :param query: The query string.
    :return: One big string, where each line-block is:
             "{rank}. (Score: 0.XXXX)\n{question-and-answer}\n"
    """
    return _perform_search(
        query=query,
        embeddings_path="src/resources/q_and_a_embeddings.npy",
        texts_path="src/resources/q_and_a_texts.json",
        top_k=20
    )

def search_bnb_law(query: str) -> str:
    """
    Finds the most similar items from BNB law documents to `query` and returns a single formatted string.

    :param query: The query string.
    :return: One big string with the top 10 most relevant articles, where each line-block is:
             "{rank}. (Score: 0.XXXX)\n{article-text}\n"
    """
    return _perform_search(
        query=query,
        embeddings_path="src/resources/bnb_law_embeddings.npy",
        texts_path="src/resources/bnb_law_texts.json",
        top_k=10
    )

if __name__ == "__main__":
    query = "Ще се повишат ли цените?"
    results = search_qa(query)
    print(results)

    bnb_results = search_bnb_law(query)
    print(bnb_results)
