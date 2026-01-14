"""
RÔLE :
Scraper 50 recettes depuis Fourchette & Bikini avec Playwright (pages parfois dynamiques).

STRATÉGIE :
1) Collecter des URLs de recettes depuis une (ou plusieurs) pages listes.
2) Pour chaque URL recette :
   - tenter extraction via schema.org/Recipe (JSON-LD) (robuste)
   - sinon fallback HTML (ingrédients + étapes)
3) Export JSONL compatible RAG.

BONNES PRATIQUES :
- Limiter le nombre de pages/recettes
- Ajouter un petit délai entre les requêtes
- Garder source_url pour la traçabilité

USAGE :
python rag/scrape_playwright.py
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

START_PAGES = [
    # Page liste principale des recettes
    "https://www.fourchette-et-bikini.fr/recettes/index.html",
    # Optionnel : pages catégories (tu peux en ajouter)
    "https://www.fourchette-et-bikini.fr/recettes/plats/index.html",
    "https://www.fourchette-et-bikini.fr/recettes/desserts/index.html",
    "https://www.fourchette-et-bikini.fr/recettes/entrees/index.html",
]

MAX_RECIPES = 50
OUTPUT_PATH = Path("data/scraped_recipes.jsonl")

DELAY_SECONDS = 2  # Délai entre les requêtes pour éviter de surcharger le serveur

DOMAIN = "www.fourchette-et-bikini.fr"
RECIPE_URL_RE = re.compile(r"^https://www\.fourchette-et-bikini\.fr/recettes/.+\.html$", re.IGNORECASE)

# On exclut ces pages (listes, diaporamas, etc.)
EXCLUDE_SUBSTRINGS = [
    "/recettes/index.html",
    "/recettes/plats/index.html",
    "/recettes/entrees/index.html",
    "/recettes/desserts/index.html",
    "/recettes/aperitifs/index.html",
    "/interne/",  # diaporamas / pages internes
]

def _safe_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    return [s] if s else []


def _extract_steps_from_recipe_obj(recipe_obj: Dict[str, Any]) -> List[str]:
    instr = recipe_obj.get("recipeInstructions")
    if instr is None:
        return []

    if isinstance(instr, str):
        parts = [p.strip() for p in re.split(r"\n+|\.\s+", instr) if p.strip()]
        return parts

    if isinstance(instr, list):
        steps: List[str] = []
        for item in instr:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    steps.append(s)
            elif isinstance(item, dict):
                txt = str(item.get("text", "")).strip()
                if txt:
                    steps.append(txt)
        return steps

    return []


def _find_recipe_jsonld(html: str) -> Optional[Dict[str, Any]]:
    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    for raw in scripts:
        raw = raw.strip()
        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        candidates: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            candidates = [data]
        elif isinstance(data, list):
            candidates = [d for d in data if isinstance(d, dict)]

        expanded: List[Dict[str, Any]] = []
        for c in candidates:
            if "@graph" in c and isinstance(c["@graph"], list):
                expanded.extend([g for g in c["@graph"] if isinstance(g, dict)])
            else:
                expanded.append(c)

        for obj in expanded:
            t = obj.get("@type")
            types: List[str] = []
            if isinstance(t, list):
                types = [str(x).lower() for x in t]
            elif t:
                types = [str(t).lower()]

            if any(x == "recipe" or x.endswith(":recipe") for x in types):
                return obj

    return None

def _fallback_extract_from_html(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    # Titre
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # Ingrédients : on cherche une section contenant "Ingrédients"
    ingredients: List[str] = []
    ingr_header = soup.find(string=re.compile(r"Ingr[ée]dients", re.IGNORECASE))
    if ingr_header:
        # remonte un peu et prend les <li> proches
        container = ingr_header.find_parent()
        if container:
            lis = container.find_all("li")
            ingredients = [li.get_text(" ", strip=True) for li in lis if li.get_text(strip=True)]

    # Étapes : section contenant "Préparation"
    steps: List[str] = []
    prep_header = soup.find(string=re.compile(r"Pr[ée]paration", re.IGNORECASE))
    if prep_header:
        container = prep_header.find_parent()
        if container:
            # souvent des <li> ou des <p>
            lis = container.find_all("li")
            if lis:
                steps = [li.get_text(" ", strip=True) for li in lis if li.get_text(strip=True)]
            else:
                ps = container.find_all("p")
                steps = [p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True)]

    return {"title": title, "ingredients": ingredients, "steps": steps}

def _is_valid_recipe_url(url: str) -> bool:
    if not RECIPE_URL_RE.match(url):
        return False
    for bad in EXCLUDE_SUBSTRINGS:
        if bad in url:
            return False
    return True


def collect_recipe_links(page_html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(page_html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)

        # Normalise
        abs_url = abs_url.split("#")[0]

        # Domaine
        if urlparse(abs_url).netloc != DOMAIN:
            continue

        if _is_valid_recipe_url(abs_url):
            links.append(abs_url)

    # dédup en gardant l'ordre
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# -----------------------------
# Scrape recette
# -----------------------------
def scrape_recipe(browser, url: str) -> Optional[Dict[str, Any]]:
    page = browser.new_page()
    page.set_default_timeout(60_000)

    page.goto(url, wait_until="networkidle")
    html = page.content()
    page.close()

    # 1) JSON-LD
    recipe_obj = _find_recipe_jsonld(html)
    if recipe_obj:
        title = str(recipe_obj.get("name", "")).strip() or "Unknown title"
        ingredients = _safe_list(recipe_obj.get("recipeIngredient"))
        steps = _extract_steps_from_recipe_obj(recipe_obj)

        # keywords/cuisine/category
        tags = []
        kw = recipe_obj.get("keywords")
        if isinstance(kw, str):
            tags.extend([t.strip() for t in kw.split(",") if t.strip()])
        else:
            tags.extend(_safe_list(kw))
        tags.extend(_safe_list(recipe_obj.get("recipeCuisine")))
        tags.extend(_safe_list(recipe_obj.get("recipeCategory")))

        return {
            "id": f"scraped_{abs(hash(url))}",
            "title": title,
            "country": "France",
            "ingredients": ingredients,
            "steps": steps,
            "tags": list(dict.fromkeys([t for t in tags if t])),
            "time_minutes": None,
            "source_url": url,
        }

    # 2) Fallback HTML
    fb = _fallback_extract_from_html(html)
    title = fb.get("title") or "Unknown title"
    ingredients = fb.get("ingredients", [])
    steps = fb.get("steps", [])

    if not ingredients or not steps:
        return None

    return {
        "id": f"scraped_{abs(hash(url))}",
        "title": title,
        "country": "France",
        "ingredients": ingredients,
        "steps": steps,
        "tags": [],
        "time_minutes": None,
        "source_url": url,
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Collecte des liens de recettes…")
    recipe_urls: List[str] = []
    seen: Set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(60_000)

        # 1) collect links from start pages
        for start in START_PAGES:
            if len(seen) >= MAX_RECIPES:
                break
            print(f"Page liste: {start}")
            page.goto(start, wait_until="networkidle")
            html = page.content()

            found = collect_recipe_links(html, start)
            for u in found:
                if u not in seen:
                    seen.add(u)
                    recipe_urls.append(u)
                if len(seen) >= MAX_RECIPES:
                    break

            time.sleep(DELAY_SECONDS)

        page.close()

        # Si on n'a pas assez, on essaye de récupérer plus en revisitant les pages listes
        # (souvent les pages listes en ont déjà largement assez)
        recipe_urls = recipe_urls[:MAX_RECIPES]

        print(f"Liens recettes collectés: {len(recipe_urls)}")

        print("Scraping des recettes…")
        ok = 0
        fail = 0

        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            for i, url in enumerate(recipe_urls, start=1):
                print(f"  [{i}/{len(recipe_urls)}] {url}")
                try:
                    item = scrape_recipe(browser, url)
                    if not item:
                        print("extraction impossible (pas d'ingrédients/étapes)")
                        fail += 1
                    else:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        ok += 1
                except Exception as e:
                    print(f"erreur: {e}")
                    fail += 1

                time.sleep(DELAY_SECONDS)

        browser.close()

    print(f"\nTerminé. OK={ok} | FAIL={fail}")
    print(f"JSONL: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()