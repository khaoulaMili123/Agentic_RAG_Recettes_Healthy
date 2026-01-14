"""
RÔLE :
Indexer data/recipes_merged.jsonl dans ChromaDB avec des embeddings Hugging Face
(multilingual-e5-base) pour alimenter le RAG.

ENTRÉE :
- data/recipes_merged.jsonl

SORTIE :
- chroma_db/ (index persistant)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------- Paramètres ---------
INPUT_JSONL = Path("data/recipes_merged.jsonl")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "recipes"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

BATCH_SIZE = 128


def recipe_to_text(obj: Dict) -> str:
    title = (obj.get("title") or "").strip()
    tags = obj.get("tags") or []
    ingredients = obj.get("ingredients") or []
    steps = obj.get("steps") or []

    tags_str = ", ".join([str(t).strip() for t in tags if str(t).strip()])
    ing_str = "\n".join([f"- {x}" for x in ingredients if str(x).strip()])
    steps_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps) if str(s).strip()])

    #nutrition si dispo
    nutr = obj.get("nutrition") or {}
    calories = nutr.get("calories")

    return f"""Title: {title}
Tags: {tags_str}
Calories: {calories}

Ingredients:
{ing_str}

Steps:
{steps_str}
""".strip()


def load_documents(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = recipe_to_text(obj)

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "id": obj.get("id"),
                        "title": obj.get("title"),
                        "source": obj.get("source"),
                        "source_url": obj.get("source_url"),
                        "time_minutes": obj.get("time_minutes"),
                    },
                )
            )
    return docs


def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_JSONL}")

    print(f"Chargement JSONL: {INPUT_JSONL}")
    docs = load_documents(INPUT_JSONL)
    print(f"Documents chargés: {len(docs)}")

    print(f"Embeddings HF: {EMBED_MODEL_NAME}")
    # Note: E5 recommande souvent de prefixer les requêtes par "query: " et les docs par "passage: "
    # Ici on indexe des passages; on appliquera le prefix côté retrieve pour maximiser la perf.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Construction Chroma: {CHROMA_DIR} / collection={COLLECTION_NAME}")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    vectordb.persist()

    print("Index terminé.")
    print(f"Chroma DB: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
