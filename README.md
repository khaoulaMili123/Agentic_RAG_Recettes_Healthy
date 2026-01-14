# 🥗 Recettes Saines et Gourmandes — Agentic RAG

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1.14-green?logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?logo=databricks&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)

**Un assistant IA intelligent pour découvrir, transformer et personnaliser des recettes healthy**

[ Démarrage Rapide](#-démarrage-rapide) • [ Documentation](#-architecture) • [ Outils](#-outils-disponibles) • [ Tests](#-tests)

</div>

---

## Table des matières

- [ Fonctionnalités](#-fonctionnalités)
- [ Architecture](#-architecture)
- [ Démarrage Rapide](#-démarrage-rapide)
- [ Structure du Projet](#-structure-du-projet)
- [ Outils Disponibles](#-outils-disponibles)
- [ Pipeline RAG](#-pipeline-rag)
- [ Interface Utilisateur](#-interface-utilisateur)
- [ Configuration](#-configuration)
- [ Tests](#-tests)
- [ Données](#-données)
- [ Contribution](#-contribution)

---

##  Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
|  **Recherche RAG** | Recherche sémantique parmi 2048+ recettes indexées avec ChromaDB |
|  **Transformation Healthy** | Substitutions intelligentes (beurre → huile d'olive, sucre → miel, etc.) |
|  **Détection Allergènes** | Identification automatique de 8 catégories d'allergènes |
|  **Estimation Nutritionnelle** | Calcul approximatif des calories et macronutriments |
|  **Traduction FR/EN** | Traduction automatique des recettes anglaises avec Helsinki-NLP |
|  **Liste de Courses** | Génération automatique d'une liste d'achats dédupliquée |
|  **Agent Conversationnel** | Interface naturelle powered by GPT-4o-mini |

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERFACE UTILISATEUR                     │
│                      (Streamlit - app.py)                        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT LANGCHAIN                          │
│                       (agent/agent.py)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  GPT-4o-mini + OpenAI Functions + Prompts System        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
┌──────────────────────┐ ┌──────────────┐ ┌──────────────────────┐
│   TOOLS (agent/)     │ │   RAG (rag/) │ │   DATA (data/)       │
│ ──────────────────── │ │ ──────────── │ │ ────────────────     │
│ • retrieve_recipes   │ │ • ChromaDB   │ │ • recipes.jsonl      │
│ • make_recipe_healthier│ • E5 Embeds  │ │ • scraped_recipes    │
│ • allergen_check     │ │ • retrieve   │ │ • recipes_merged     │
│ • estimate_macros    │ │              │ │                      │
│ • shopping_list      │ │              │ │                      │
│ • translate_to_fr    │ │              │ │                      │
└──────────────────────┘ └──────────────┘ └──────────────────────┘
```

### Flux de données

```
Requête Utilisateur
        │
        ▼
   ┌─────────┐
   │  Agent  │ ─── Décide quel(s) outil(s) appeler
   └────┬────┘
        │
        ├──► retrieve_recipes() ──► ChromaDB ──► Recettes pertinentes
        │
        ├──► make_recipe_healthier() ──► Substitutions santé
        │
        ├──► allergen_check() ──► Vérification allergènes
        │
        ├──► estimate_macros() ──► Infos nutritionnelles
        │
        ├──► translate_recipe_to_french() ──► Traduction
        │
        └──► shopping_list() ──► Liste de courses
                │
                ▼
        Réponse Structurée à l'Utilisateur
```

---

##  Démarrage Rapide

### Prérequis

- **Python 3.12+**
- **pip** ou **conda**
- **Clé API OpenAI** (GPT-4o-mini)
- *(Optionnel)* Clé API LangSmith pour le tracing

### Installation

```bash
# 1. Cloner le projet
git clone <url-du-repo>
cd agent

# 2. Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API
```

### Configuration `.env`

```env
# OpenAI (obligatoire)
OPENAI_API_KEY=sk-...

# LangSmith (optionnel - pour le tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=healthy-recipes-agent
```

### Indexation des données

```bash
# Indexer les recettes dans ChromaDB (à faire une fois)
python -m rag.index_chroma
```

### Lancement

```bash
# Démarrer l'interface Streamlit
streamlit run app.py
```

L'application sera accessible sur **http://localhost:8501**

---

##  Structure du Projet

```
agent/
├──  app.py                    # Interface Streamlit (point d'entrée UI)
├──  requirements.txt          # Dépendances Python
├──  test_tools.py             # Tests des outils
├──  .env                      # Variables d'environnement (non versionné)
│
├──  agent/                    # Module Agent
│   ├── __init__.py
│   ├── agent.py                 # Orchestration LangChain + OpenAI Functions
│   ├── prompts.py               # System prompts et format de sortie
│   └── tools.py                 # Définition des 6 outils
│
├──  rag/                      # Module RAG (Retrieval-Augmented Generation)
│   ├── __init__.py
│   ├── index_chroma.py          # Indexation des recettes dans ChromaDB
│   ├── retrieve.py              # Recherche sémantique
│   ├── scrape.py                # Scraping Fourchette & Bikini (Playwright)
│   ├── load_data.py             # Chargement datasets Hugging Face
│   ├── merge_jsonl.py           # Fusion des sources de données
│   └── transform_data.py        # Transformation et nettoyage
│
├──  data/                     # Données brutes et traitées
│   ├── recipes.jsonl            # Recettes Hugging Face
│   ├── scraped_recipes.jsonl    # Recettes scrapées
│   └── recipes_merged.jsonl     # Dataset fusionné (2048 recettes)
│
└──  chroma_db/                # Base vectorielle ChromaDB (générée)
    └── ...
```

---

##  Outils Disponibles

### 1.  `retrieve_recipes`

Recherche sémantique dans la base de 2048 recettes.

```python
retrieve_recipes(query="poulet protéiné sans gluten", k=5)
```

**Retourne:**
- Recettes structurées (titre, ingrédients, étapes, tags)
- Métadonnées (source, URL, temps de préparation)
- Score de similarité

---

### 2.  `make_recipe_healthier`

Transforme une recette en version plus saine avec des substitutions intelligentes.

```python
make_recipe_healthier(recipe={
    "title": "Pasta Carbonara",
    "ingredients": ["pasta", "bacon", "butter", "cream", "cheese"],
    "steps": [...]
})
```

**Substitutions disponibles:**

| Original | Substitution | Bénéfice |
|----------|--------------|----------|
| Beurre | Huile d'olive, avocat | -Graisses saturées |
| Crème | Yaourt grec 0% | -Calories |
| Sucre | Miel, stévia | -Index glycémique |
| Farine | Farine complète, amande | +Fibres |
| Pâtes | Pâtes légumineuses | +Protéines |
| Bacon | Bacon de dinde | -Graisses |
| Fromage | Levure nutritionnelle | -Graisses |
| Sel | Herbes, citron | -Sodium |

**Retourne:**
- Titre version healthy
- Liste des substitutions avec raisons
- Améliorations estimées (calories, graisses, fibres)
- Conseils santé

---

### 3.  `allergen_check`

Détecte les allergènes dans une liste d'ingrédients.

```python
allergen_check(
    ingredients=["milk", "flour", "eggs", "peanut butter"],
    allergies=["lactose", "gluten", "arachides"]
)
```

**Allergènes détectables:**
-  Lactose (lait, fromage, beurre, crème)
-  Gluten (blé, farine, pâtes, pain)
-  Arachides
-  Œufs
-  Soja
-  Fruits de mer
-  Poisson
-  Noix

---

### 4.  `estimate_macros`

Estime les valeurs nutritionnelles d'une recette.

```python
estimate_macros(recipe={
    "title": "Grilled Chicken Salad",
    "ingredients": ["chicken breast", "lettuce", "olive oil"],
    "nutrition": {"calories": 350}  # Si disponible
})
```

**Retourne:**
- Calories (si disponible dans le dataset)
- Estimation heuristique des protéines
- Indication de la méthode utilisée

---

### 5.  `translate_recipe_to_french`

Traduit automatiquement les recettes anglaises en français.

```python
translate_recipe_to_french(recipe={
    "title": "Grilled Chicken Salad",
    "ingredients": ["chicken breast", "lettuce"],
    "steps": ["Grill the chicken"]
})
```

**Modèle utilisé:** `Helsinki-NLP/opus-mt-en-fr`

**Retourne:**
- Titre traduit
- Ingrédients traduits
- Étapes traduites
- Flag `translated: true`

---

### 6.  `shopping_list`

Génère une liste de courses dédupliquée.

```python
shopping_list(ingredients=[
    "chicken breast 500g",
    "olive oil",
    "garlic",
    "olive oil",  
    "salt"
])
```

**Retourne:**
- Liste d'items uniques
- Compteur d'articles

---

##  Pipeline RAG

### Embeddings

- **Modèle:** `intfloat/multilingual-e5-base`
- **Dimension:** 768
- **Multilingue:** FR/EN supportés nativement

### Vectorstore

- **Base:** ChromaDB (persistant sur disque)
- **Collection:** `recipes`
- **Documents:** 2048 recettes

### Processus d'indexation

```bash
python -m rag.index_chroma
```

1. Charge `data/recipes_merged.jsonl`
2. Convertit chaque recette en document textuel
3. Génère les embeddings avec E5
4. Stocke dans `chroma_db/`

### Recherche

```python
from rag.retrieve import retrieve

results = retrieve(query="recette healthy protéinée", k=5)
```

---

##  Interface Utilisateur

### Fonctionnalités UI

- **Header** avec gradient vert moderne
- **Sidebar** avec exemples de requêtes cliquables
- **Cards** de fonctionnalités animées
- **Options avancées** (allergènes, nutrition, liste courses, version healthy)
- **Affichage structuré** des résultats
- **Conseils santé** dynamiques


##  Configuration

### Variables d'environnement

| Variable | Obligatoire | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | ✅ | Clé API OpenAI |
| `LANGCHAIN_TRACING_V2` | ❌ | Activer LangSmith (`true/false`) |
| `LANGCHAIN_API_KEY` | ❌ | Clé API LangSmith |
| `LANGCHAIN_PROJECT` | ❌ | Nom du projet LangSmith |

### Paramètres Agent

Dans `agent/agent.py`:

```python
build_agent(
    model="gpt-4o-mini",      # Modèle OpenAI
    temperature=0.2,          # Créativité (0-1)
    verbose=False             # Logs détaillés
)
```

### Paramètres RAG

Dans `rag/index_chroma.py`:

```python
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 128
COLLECTION_NAME = "recipes"
```

---

##  Tests

### Test des outils

```bash
python test_tools.py
```

**Sortie attendue:**

```
============================================================
 TEST: retrieve_recipes
============================================================
 Récupéré 3 recette(s)
   - Titre: Bacon Ranch Chicken Bake
   - Score: 0.28

 TEST: allergen_check
 Allergènes détectés: True
   - Trouvés: ['gluten', 'lactose', 'oeuf']

 TEST: make_recipe_healthier
 Transformation healthy effectuée
   - 6 substitutions proposées
   - Améliorations: -15 à -30% calories

 Tous les outils sont fonctionnels !
```

### Test de l'agent complet

```python
from agent.agent import run_agent

response = run_agent("Je veux une recette de pâtes healthy sans gluten")
print(response)
```

---

##  Données

### Sources

1. **Hugging Face Datasets** (`recipes.jsonl`)
   - Dataset de recettes internationales
   - ~2000 recettes

2. **Scraping Fourchette & Bikini** (`scraped_recipes.jsonl`)
   - Recettes françaises healthy
   - ~50 recettes avec Playwright

### Format JSONL

```json
{
  "id": "recipe_001",
  "title": "Poulet grillé aux herbes",
  "tags": ["healthy", "high-protein", "quick"],
  "ingredients": ["400g poulet", "huile d'olive", "herbes"],
  "steps": ["Préchauffer le four", "Assaisonner le poulet", "..."],
  "nutrition": {"calories": 320, "protein_g": 45},
  "time_minutes": 30,
  "source": "fourchette-et-bikini",
  "source_url": "https://..."
}
```

### Régénérer les données

```bash
# 1. Scraper de nouvelles recettes
python -m rag.scrape

# 2. Fusionner les sources
python -m rag.merge_jsonl

# 3. Réindexer dans ChromaDB
python -m rag.index_chroma
```

---

##  Dépendances Principales

| Package | Version | Usage |
|---------|---------|-------|
| `langchain` | 0.1.14 | Framework agent |
| `openai` | ≥0.27.14 | API GPT-4o-mini |
| `chromadb` | ≥0.5.5 | Vector store |
| `sentence-transformers` | ≥3.0.1 | Embeddings E5 |
| `transformers` | ≥4.38.0 | Traduction Helsinki-NLP |
| `streamlit` | ≥1.30.0 | Interface web |
| `playwright` | ≥1.41.0 | Scraping dynamique |
| `beautifulsoup4` | ≥4.12.3 | Parsing HTML |

---

##  Contribution

### Guidelines

1. **Fork** le repository
2. Créer une **branche** (`git checkout -b feature/ma-feature`)
3. **Commit** les changements (`git commit -m 'Add: ma feature'`)
4. **Push** sur la branche (`git push origin feature/ma-feature`)
5. Ouvrir une **Pull Request**

### Idées d'amélioration

- [ ] Ajouter plus de sources de recettes
- [ ] Améliorer l'estimation nutritionnelle avec une API (USDA, OpenFoodFacts)
- [ ] Ajouter un système de favoris utilisateur
- [ ] Intégrer la génération d'images de recettes (DALL-E)
- [ ] Ajouter le support vocal (Whisper)
- [ ] Déployer sur Streamlit Cloud / Hugging Face Spaces

---

##  Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

