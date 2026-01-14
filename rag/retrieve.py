"""
RÔLE :
Ce fichier gère la récupération intelligente des recettes (RAG).

CE QU'IL FAIT :
- Effectue une recherche sémantique dans ChromaDB (retrieval large)
- Applique un reranking pour améliorer la pertinence des résultats
- Retourne uniquement les recettes les plus pertinentes

POURQUOI C'EST IMPORTANT :
- Le reranking améliore fortement la qualité du contexte fourni au LLM
- Cette étape évite d'envoyer des recettes peu pertinentes à l'agent

QUAND IL EST UTILISÉ :
- À chaque requête utilisateur
- Appelé par l'agent via un tool

SORTIE :
- Une liste restreinte de recettes pertinentes (top-k)

USAGE (test rapide) :
python -m rag.retrieve
"""

"""
RÔLE :
Récupération (retrieval) des recettes depuis ChromaDB pour le RAG.

- Charge la base vectorielle persistée dans chroma_db/
- Utilise les mêmes embeddings HF que l'indexation (multilingual-e5-base)
- Applique le préfixe E5: "query: ..." pour améliorer la recherche
- Retourne les top-k documents (recettes) pertinents

USAGE (test rapide) :
python -m rag.retrieve
"""

from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "recipes"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"


def get_vectordb() -> Chroma:
    """
    Ouvre la base Chroma persistée.
    IMPORTANT: embeddings identiques à ceux utilisés pour indexer.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vectordb


def retrieve(query: str, k: int = 50, rerank_k: int = 3):
    vectordb = get_vectordb()
    e5_query = f"query: {query.strip()}"

    results = vectordb.similarity_search_with_score(e5_query, k=k)

    docs = []
    for doc, score in results:
        docs.append(
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) if score is not None else None,
            }
        )

    # RERANKING désactivé (fonction rerank absente)
    # docs = rerank(query=query, docs=docs, top_k=rerank_k)
    # On retourne les docs tels quels
    docs = docs
    return docs


def format_results(results: List[Dict[str, Any]]) -> str:
    """
    Affichage lisible dans le terminal.
    """
    blocks = []
    for i, item in enumerate(results, start=1):
        meta = item.get("metadata", {}) or {}
        title = meta.get("title") or "Unknown title"
        source = meta.get("source") or "unknown_source"
        score = item.get("score")

        header = f"#{i} — {title} | source={source} | score={score}"
        blocks.append(header + "\n" + item.get("text", "").strip())
    return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(blocks) + "\n"


def main():
    print("Test retrieval (RAG)")
    q = input("Requête: ").strip()
    if not q:
        print("Requête vide. Stop.")
        return

    results = retrieve(q, k=6)
    print(format_results(results))


if __name__ == "__main__":
    main()


