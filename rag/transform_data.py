"""
RÔLE :
Ce fichier transforme les données brutes en un format propre et exploitable
pour le système RAG.

CE QU'IL FAIT :
- Nettoie les données (recettes incomplètes, champs vides)
- Convertit les listes stockées en string (ingredients, steps, tags)
- Réduit la taille du dataset (échantillonnage)
- Normalise les champs importants

QUAND IL EST UTILISÉ :
- Après le chargement des données (load_data.py)
- Avant la création de l'index vectoriel

SORTIE :
- Un fichier JSONL propre : data/recipes.jsonl
- Chaque ligne correspond à une recette prête pour le RAG
"""

import json
import re
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from rag.load_data import load_foodcom_recipes


OUTPUT_PATH = Path("data/recipes.jsonl")

SAMPLE_SIZE = 2000
RANDOM_SEED = 42
MIN_INGREDIENTS = 3
MIN_STEPS = 3


def parse_r_c_vector(value: Any) -> List[str]:
    """
    Parse un champ qui ressemble à :
      c("step1", "step2", "step3")
    ou c('a','b')
    Retourne une liste de strings.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    s = str(value).strip()
    if not s:
        return []

    # Cas: c("..", "..")
    if s.startswith("c(") and s.endswith(")"):
        inner = s[2:-1].strip()

        # extrait les valeurs entre guillemets "..." ou '...'
        items = re.findall(r'"([^"]+)"|\'([^\']+)\'', inner)
        out = []
        for a, b in items:
            txt = (a or b).strip()
            if txt:
                out.append(txt)
        return out

    # fallback: essayer de split sur virgule si c'est une simple string
    return [x.strip() for x in s.split(",") if x.strip()]


def clean_list(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def parse_iso8601_minutes(value: Any) -> Optional[int]:
    """
    Parse une durée ISO 8601 du style:
      PT20M, PT1H30M
    Retourne minutes (int) ou None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None

    # Ex: PT1H30M
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?$", s)
    if not m:
        return None
    hours = int(m.group(1)) if m.group(1) else 0
    mins = int(m.group(2)) if m.group(2) else 0
    return hours * 60 + mins


def main():
    print("Chargement Kaggle (Food.com)...")
    df = load_foodcom_recipes()

    # Colonnes confirmées chez toi
    col_id = "RecipeId"
    col_title = "Name"
    col_ing_parts = "RecipeIngredientParts"
    col_instr = "RecipeInstructions"
    col_keywords = "Keywords"
    col_total_time = "TotalTime"

    print("Transformation...")
    df = df.copy()

    df["__title"] = df[col_title].astype(str).str.strip()

    # Ingredients: parts (et optionnellement quantities)
    df["__ingredients"] = df[col_ing_parts].apply(parse_r_c_vector).apply(clean_list)

    # Steps
    df["__steps"] = df[col_instr].apply(parse_r_c_vector).apply(clean_list)

    # Tags
    df["__tags"] = df[col_keywords].apply(parse_r_c_vector).apply(clean_list)

    # Time minutes
    df["__time_minutes"] = df[col_total_time].apply(parse_iso8601_minutes)

    # Filtrage qualité
    df = df[
        (df["__title"].str.len() > 0)
        & (df["__ingredients"].apply(len) >= MIN_INGREDIENTS)
        & (df["__steps"].apply(len) >= MIN_STEPS)
    ]

    # Dédup simple par titre
    df["__title_key"] = df["__title"].str.lower()
    df = df.drop_duplicates(subset="__title_key")

    print(f"Recettes valides après filtrage: {len(df)}")

    # Échantillonnage
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Export JSONL → {OUTPUT_PATH}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            recipe_id = row.get(col_id)

            recipe = {
                "id": f"foodcom_{int(recipe_id)}" if pd.notna(recipe_id) else f"foodcom_{abs(hash(row['__title']))}",
                "title": row["__title"],
                "country": "Unknown",
                "ingredients": row["__ingredients"],
                "steps": row["__steps"],
                "tags": row["__tags"],
                "time_minutes": int(row["__time_minutes"]) if pd.notna(row["__time_minutes"]) else None,

                # Bonus nutrition (déjà dans le dataset)
                "nutrition": {
                    "calories": float(row["Calories"]) if pd.notna(row.get("Calories")) else None,
                    "protein_g": float(row["ProteinContent"]) if pd.notna(row.get("ProteinContent")) else None,
                    "fat_g": float(row["FatContent"]) if pd.notna(row.get("FatContent")) else None,
                    "carbs_g": float(row["CarbohydrateContent"]) if pd.notna(row.get("CarbohydrateContent")) else None,
                    "fiber_g": float(row["FiberContent"]) if pd.notna(row.get("FiberContent")) else None,
                    "sugar_g": float(row["SugarContent"]) if pd.notna(row.get("SugarContent")) else None,
                    "sodium_mg": float(row["SodiumContent"]) if pd.notna(row.get("SodiumContent")) else None,
                },

                "source": "kaggle_foodcom",
                "source_url": None,
            }

            f.write(json.dumps(recipe, ensure_ascii=False) + "\n")

    print(f"Terminé: {len(df)} recettes exportées.")
    print("Vérifie avec: head -n 2 data/recipes.jsonl")


if __name__ == "__main__":
    main()