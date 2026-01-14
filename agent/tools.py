"""
RÔLE :
Ce fichier définit les outils que l'agent peut appeler dynamiquement.

OUTILS CONTENUS :
- retrieve_recipes : appelle le système RAG (retrieve.py)
- translate_recipe_to_french : traduction EN -> FR
- allergen_check : détecte les allergènes courants
- estimate_macros : estime les valeurs nutritionnelles
- shopping_list : génère une liste de courses

POURQUOI C'EST IMPORTANT :
- Les outils rendent l'agent "agentic" (raisonnement + actions)
- L'agent choisit quels outils utiliser selon la demande utilisateur

UTILISATION :
- Importé par agent.py
"""

from __future__ import annotations
from langchain.tools import tool
import re
from typing import List, Dict, Any, Optional
from langchain_core.tools import Tool
from rag.retrieve import retrieve as rag_retrieve

def _parse_recipe_text(text: str) -> Dict[str, Any]:
    """
    Convertit le page_content indexé en structure exploitable:
    - title, tags, calories, ingredients(list), steps(list)

    Format attendu (depuis index_chroma.py):
      Title: ...
      Tags: ...
      Calories: ...
      Ingredients:
      - ...
      Steps:
      1. ...
    """
    title = ""
    tags: List[str] = []
    calories: Optional[float] = None
    ingredients: List[str] = []
    steps: List[str] = []

    lines = [ln.strip() for ln in (text or "").splitlines()]

    mode = None  # None | "ingredients" | "steps"
    for ln in lines:
        if not ln:
            continue

        if ln.lower().startswith("title:"):
            title = ln.split(":", 1)[1].strip()
            mode = None
            continue

        if ln.lower().startswith("tags:"):
            raw = ln.split(":", 1)[1].strip()
            if raw:
                tags = [t.strip() for t in raw.split(",") if t.strip()]
            mode = None
            continue

        if ln.lower().startswith("calories:"):
            raw = ln.split(":", 1)[1].strip()
            try:
                calories = float(raw) if raw and raw.lower() != "none" else None
            except Exception:
                calories = None
            mode = None
            continue

        if ln.lower() == "ingredients:":
            mode = "ingredients"
            continue

        if ln.lower() == "steps:":
            mode = "steps"
            continue

        if mode == "ingredients":
            if ln.startswith("- "):
                ingredients.append(ln[2:].strip())
            else:
                # fallback
                ingredients.append(ln)
            continue

        if mode == "steps":
            # "1. bla" -> extract
            m = re.match(r"^\d+\.\s*(.+)$", ln)
            steps.append(m.group(1).strip() if m else ln)
            continue

    # Nettoyage léger
    def _dedup_keep_order(items: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in items:
            x = re.sub(r"\s+", " ", str(x)).strip()
            k = x.lower()
            if not x or k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    return {
        "title": title.strip() if title else "Unknown title",
        "tags": _dedup_keep_order(tags),
        "ingredients": _dedup_keep_order(ingredients),
        "steps": _dedup_keep_order(steps),
        "nutrition": {"calories": calories} if calories is not None else {},
    }

_TRANSLATOR = None


def _get_translator():
    global _TRANSLATOR
    if _TRANSLATOR is None:
        from transformers import pipeline

        _TRANSLATOR = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
    return _TRANSLATOR


def _looks_french(text: str) -> bool:
    if not text:
        return True
    markers = [" le ", " la ", " les ", " des ", " une ", " et ", " du ", " au ", " aux "]
    t = " " + text.lower() + " "
    return any(m in t for m in markers)


def _translate_text(text: str) -> str:
    if not text or _looks_french(text):
        return text
    tr = _get_translator()
    out = tr(text, max_length=512)
    return out[0]["translation_text"]

@tool
def retrieve_recipes(query: str, k: int = 50) -> List[Dict[str, Any]]:
    """
    Recherche des recettes dans la base RAG (Chroma) et retourne des recettes STRUCTURÉES.

    Args:
      query: requête utilisateur (ex: "healthy riche en protéines sans lactose")
      k: nombre de résultats

    Returns (liste) :
      [
        {
          "recipe": { "title", "ingredients", "steps", "tags", "nutrition" },
          "metadata": { "id", "title", "source", "source_url", "time_minutes", ... },
          "score": float
        },
        ...
      ]
    """
    raw = rag_retrieve(query=query, k=k)
    out: List[Dict[str, Any]] = []
    for item in raw:
        text = item.get("text", "")
        recipe = _parse_recipe_text(text)
        out.append(
            {
                "recipe": recipe,
                "metadata": item.get("metadata", {}) or {},
                "score": item.get("score"),
            }
        )
    return out


@tool
def translate_recipe_to_french(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traduit une recette (title/ingredients/steps) en français.
    Entrée: dict contenant au moins title, ingredients(list), steps(list)
    Sortie: même dict + translated=True
    """
    translated = dict(recipe)

    translated["title"] = _translate_text(str(recipe.get("title", "")).strip())
    translated["ingredients"] = [_translate_text(str(x)) for x in (recipe.get("ingredients") or [])]
    translated["steps"] = [_translate_text(str(x)) for x in (recipe.get("steps") or [])]

    translated["translated"] = True
    translated["translation_model"] = "Helsinki-NLP/opus-mt-en-fr"
    return translated


@tool
def allergen_check(ingredients: List[str], allergies: List[str]) -> Dict[str, Any]:
    """
    Détecte des allergènes dans une liste d'ingrédients.
    allergies exemples: ["lactose","gluten","arachides","oeuf","soja","fruits de mer","poisson","noix"]
    """
    allergen_keywords = {
        "lactose": ["milk", "cheese", "butter", "cream", "yogurt", "lait", "fromage", "beurre", "crème", "yaourt"],
        "gluten": ["wheat", "flour", "bread", "pasta", "blé", "farine", "pain", "pâtes"],
        "arachides": ["peanut", "peanuts", "arachide", "cacahuète", "cacahuetes"],
        "oeuf": ["egg", "eggs", "oeuf", "oeufs"],
        "soja": ["soy", "soya", "tofu", "soja", "tofu"],
        "fruits de mer": ["shrimp", "prawn", "crab", "lobster", "shellfish", "crevette", "crabe", "homard", "coquillage"],
        "poisson": ["fish", "salmon", "tuna", "cod", "poisson", "saumon", "thon", "cabillaud"],
        "noix": ["nuts", "almond", "walnut", "cashew", "noix", "amande", "noisette", "cajou"],
    }

    ing_lower = [str(x).lower() for x in (ingredients or [])]
    allergies_lower = [str(a).lower() for a in (allergies or [])]

    found = []
    matched_ingredients = []

    for allergy in allergies_lower:
        keys = allergen_keywords.get(allergy, [allergy])
        for kw in keys:
            for ing in ing_lower:
                if kw in ing:
                    found.append(allergy)
                    matched_ingredients.append(ing)

    found = sorted(set(found))
    matched_ingredients = sorted(set(matched_ingredients))
    return {
        "contains_allergens": len(found) > 0,
        "found": found,
        "matched_ingredients": matched_ingredients,
    }


@tool
def estimate_macros(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renvoie nutrition si dispo, sinon estimation simple.
    - Si recipe["nutrition"] contient des champs, on les renvoie.
    - Sinon heuristique légère.
    """
    nutr = recipe.get("nutrition")
    if isinstance(nutr, dict) and any(v is not None for v in nutr.values()):
        return {"method": "dataset_or_doc", "nutrition": nutr}

    ingredients_text = " ".join(recipe.get("ingredients") or []).lower()
    protein_hint = None
    if any(
        x in ingredients_text
        for x in [
            "chicken", "turkey", "tofu", "tuna", "salmon", "lentil", "beans",
            "poulet", "dinde", "tofu", "thon", "saumon", "lentilles", "haricots",
        ]
    ):
        protein_hint = 25

    return {
        "method": "heuristic",
        "nutrition": {"calories": None, "protein_g": protein_hint, "fat_g": None, "carbs_g": None},
        "note": "Estimation simple (à améliorer).",
    }


@tool
def shopping_list(ingredients: List[str]) -> Dict[str, Any]:
    """
    Génère une liste de courses simple à partir des ingrédients (dédup).
    """
    items = []
    seen = set()
    for x in ingredients or []:
        s = re.sub(r"\s+", " ", str(x)).strip()
        key = s.lower()
        if not s or key in seen:
            continue
        seen.add(key)
        items.append(s)
    return {"items": items, "count": len(items)}


# Substitutions healthy pour transformer les recettes
HEALTHY_SUBSTITUTIONS = {
    # Graisses
    "butter": {"substitute": "huile d'olive ou purée d'avocat", "reason": "Moins de graisses saturées"},
    "beurre": {"substitute": "huile d'olive ou purée d'avocat", "reason": "Moins de graisses saturées"},
    "cream": {"substitute": "yaourt grec 0% ou lait de coco allégé", "reason": "Moins de calories et de graisses"},
    "crème": {"substitute": "yaourt grec 0% ou crème de soja", "reason": "Moins de calories et de graisses"},
    "crème fraîche": {"substitute": "yaourt grec 0%", "reason": "Moins de matières grasses"},
    "heavy cream": {"substitute": "lait d'amande épaissi ou yaourt grec", "reason": "Réduction des graisses saturées"},
    "oil": {"substitute": "huile d'olive vierge extra (en quantité réduite)", "reason": "Graisses mono-insaturées bénéfiques"},
    
    # Sucres
    "sugar": {"substitute": "miel, sirop d'érable ou stévia", "reason": "Index glycémique plus bas"},
    "sucre": {"substitute": "miel, sirop d'érable ou stévia", "reason": "Index glycémique plus bas"},
    "brown sugar": {"substitute": "sucre de coco ou purée de dattes", "reason": "Plus de nutriments, IG plus bas"},
    "white sugar": {"substitute": "miel ou sirop d'agave", "reason": "Alternatives naturelles"},
    
    # Farines
    "flour": {"substitute": "farine complète ou farine d'amande", "reason": "Plus de fibres et protéines"},
    "farine": {"substitute": "farine complète ou farine d'avoine", "reason": "Plus de fibres"},
    "all-purpose flour": {"substitute": "farine de blé complet ou d'épeautre", "reason": "Céréales complètes"},
    "white flour": {"substitute": "farine d'amande ou de coco", "reason": "Low-carb et sans gluten"},
    
    # Produits laitiers
    "cheese": {"substitute": "fromage allégé ou levure nutritionnelle", "reason": "Moins de graisses"},
    "fromage": {"substitute": "fromage frais 0% ou ricotta allégée", "reason": "Moins de calories"},
    "milk": {"substitute": "lait d'amande, d'avoine ou écrémé", "reason": "Moins de graisses saturées"},
    "lait": {"substitute": "lait d'amande ou lait écrémé", "reason": "Moins de calories"},
    "sour cream": {"substitute": "yaourt grec nature", "reason": "Plus de protéines, moins de graisses"},
    
    # Viandes
    "bacon": {"substitute": "bacon de dinde ou tempeh fumé", "reason": "Moins de graisses saturées"},
    "ground beef": {"substitute": "dinde hachée ou protéines végétales", "reason": "Viande plus maigre"},
    "sausage": {"substitute": "saucisse de poulet ou tofu épicé", "reason": "Moins de graisses"},
    
    # Pâtes et riz
    "pasta": {"substitute": "pâtes complètes ou de légumineuses", "reason": "Plus de fibres et protéines"},
    "pâtes": {"substitute": "pâtes de lentilles ou konjac", "reason": "Low-carb et plus de fibres"},
    "white rice": {"substitute": "riz complet ou quinoa", "reason": "Plus de nutriments"},
    "riz": {"substitute": "quinoa ou riz brun", "reason": "Céréales complètes"},
    
    # Sauces
    "mayonnaise": {"substitute": "avocat écrasé ou yaourt grec", "reason": "Graisses plus saines"},
    "ketchup": {"substitute": "sauce tomate maison sans sucre", "reason": "Sans sucres ajoutés"},
    
    # Autres
    "bread": {"substitute": "pain complet ou pain aux graines", "reason": "Plus de fibres"},
    "pain": {"substitute": "pain complet ou galettes de riz", "reason": "Céréales complètes"},
    "salt": {"substitute": "herbes fraîches, citron ou épices", "reason": "Réduction du sodium"},
    "sel": {"substitute": "herbes, ail, citron", "reason": "Moins de sodium"},
}


@tool
def make_recipe_healthier(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyse une recette et propose des substitutions pour la rendre plus healthy.
    
    Args:
        recipe: dict avec title, ingredients (list), steps (list), nutrition (dict optionnel)
    
    Returns:
        {
            "original_title": str,
            "healthy_title": str,
            "substitutions": [{"original": str, "substitute": str, "reason": str}],
            "health_tips": [str],
            "estimated_improvements": {"calories": str, "fat": str, "fiber": str},
            "healthy_ingredients": [str],
            "healthy_steps": [str]
        }
    """
    ingredients = recipe.get("ingredients", []) or []
    steps = recipe.get("steps", []) or []
    title = recipe.get("title", "Recette")
    
    substitutions = []
    healthy_ingredients = []
    
    for ing in ingredients:
        ing_lower = str(ing).lower()
        found_sub = None
        
        for key, sub_info in HEALTHY_SUBSTITUTIONS.items():
            if key in ing_lower:
                found_sub = {
                    "original": ing,
                    "substitute": sub_info["substitute"],
                    "reason": sub_info["reason"]
                }
                # Créer l'ingrédient modifié
                new_ing = ing_lower.replace(key, sub_info["substitute"])
                healthy_ingredients.append(new_ing.capitalize())
                break
        
        if found_sub:
            substitutions.append(found_sub)
        else:
            healthy_ingredients.append(ing)
    
    # Tips santé généraux
    health_tips = [
        "Réduire les quantités de sel et sucre de 25-50%",
        "Privilégier la cuisson vapeur, au four ou grillée plutôt que la friture",
        "Ajouter des légumes verts pour plus de fibres",
        "Utiliser des herbes fraîches pour plus de saveur sans calories",
    ]
    
    # Adapter les étapes si nécessaire
    healthy_steps = []
    for step in steps:
        step_lower = str(step).lower()
        new_step = step
        if "fry" in step_lower or "frire" in step_lower:
            new_step = step + " (Conseil: préférer la cuisson au four ou à la poêle avec peu de matière grasse)"
        elif "butter" in step_lower or "beurre" in step_lower:
            new_step = step + " (Conseil: utiliser de l'huile d'olive en quantité réduite)"
        healthy_steps.append(new_step)
    
    # Estimation des améliorations
    improvements = {
        "calories": f"-15 à -30% estimé" if substitutions else "Pas de changement majeur",
        "fat": f"-20 à -40% de graisses saturées" if any("butter" in s["original"].lower() or "cream" in s["original"].lower() for s in substitutions) else "Stable",
        "fiber": "+50 à +100% de fibres" if any("flour" in s["original"].lower() or "pasta" in s["original"].lower() for s in substitutions) else "Stable",
    }
    
    return {
        "original_title": title,
        "healthy_title": f"{title} - Version Healthy",
        "substitutions": substitutions,
        "substitutions_count": len(substitutions),
        "health_tips": health_tips,
        "estimated_improvements": improvements,
        "healthy_ingredients": healthy_ingredients,
        "healthy_steps": healthy_steps,
    }


