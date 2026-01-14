"""
RÔLE :
Fusionner plusieurs fichiers JSONL de recettes en un seul fichier final
destiné à l'indexation RAG.

STRATÉGIE DE DÉDUPLICATION :
1) Si source_url existe → déduplication par source_url
2) Sinon → déduplication par title (normalisé)

ENTRÉES :
- data/recipes.jsonl              (Kaggle transformé)
- data/scraped_recipes.jsonl      (Scraping transformé)

SORTIE :
- data/recipes_merged.jsonl
"""

import json
from pathlib import Path
from typing import Dict, List


# -----------------------------
# Paramètres
# -----------------------------
INPUT_FILES = [
    Path("data/recipes.jsonl"),           # Kaggle
    Path("data/scraped_recipes.jsonl"),   # Scraping
]

OUTPUT_FILE = Path("data/recipes_merged.jsonl")


# -----------------------------
# Helpers
# -----------------------------
def normalize_title(title: str) -> str:
    return title.strip().lower()


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    if not path.exists():
        print(f"Fichier introuvable: {path}")
        return items

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


# -----------------------------
# Fusion
# -----------------------------
def main():
    print("Chargement des fichiers JSONL...")
    all_items: List[Dict] = []
    for p in INPUT_FILES:
        data = load_jsonl(p)
        print(f"  - {p}: {len(data)} recettes")
        all_items.extend(data)

    print(f"Total avant déduplication: {len(all_items)}")

    seen_urls = set()
    seen_titles = set()
    merged: List[Dict] = []

    for item in all_items:
        source_url = item.get("source_url")
        title = item.get("title", "")

        # Règle 1 : dédup par source_url
        if source_url:
            if source_url in seen_urls:
                continue
            seen_urls.add(source_url)

        # Règle 2 : dédup par titre normalisé
        title_key = normalize_title(title)
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        merged.append(item)

    print(f"Après déduplication: {len(merged)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Fichier fusionné créé: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
