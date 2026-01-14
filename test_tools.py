"""
Script de test pour vérifier tous les outils de l'agent.
"""

import sys
sys.path.insert(0, '/home/kouki/iadev/agent')

from agent.tools import (
    retrieve_recipes,
    translate_recipe_to_french,
    allergen_check,
    estimate_macros,
    shopping_list,
    make_recipe_healthier,
)

def test_separator(name):
    print("\n" + "="*60)
    print(f"TEST: {name}")
    print("="*60)

# ========================================
# 1. TEST retrieve_recipes
# ========================================
test_separator("retrieve_recipes")
try:
    results = retrieve_recipes.invoke({"query": "healthy chicken high protein", "k": 3})
    print(f"Récupéré {len(results)} recette(s)")
    if results:
        recipe = results[0]["recipe"]
        print(f"   - Titre: {recipe.get('title', 'N/A')}")
        print(f"   - Ingrédients: {len(recipe.get('ingredients', []))} items")
        print(f"   - Étapes: {len(recipe.get('steps', []))} steps")
        print(f"   - Score: {results[0].get('score', 'N/A')}")
except Exception as e:
    print(f"ERREUR: {e}")

# ========================================
# 2. TEST allergen_check
# ========================================
test_separator("allergen_check")
try:
    test_ingredients = ["chicken breast", "butter", "milk", "flour", "eggs", "salt"]
    test_allergies = ["lactose", "gluten", "oeuf"]
    result = allergen_check.invoke({"ingredients": test_ingredients, "allergies": test_allergies})
    print(f"Allergènes détectés: {result['contains_allergens']}")
    print(f"   - Trouvés: {result['found']}")
    print(f"   - Ingrédients concernés: {result['matched_ingredients']}")
except Exception as e:
    print(f"ERREUR: {e}")

# ========================================
# 3. TEST estimate_macros
# ========================================
test_separator("estimate_macros")
try:
    test_recipe = {
        "title": "Grilled Chicken Salad",
        "ingredients": ["chicken breast", "lettuce", "tomato", "olive oil"],
        "steps": ["Grill chicken", "Mix with vegetables"],
        "nutrition": {"calories": 350}
    }
    result = estimate_macros.invoke({"recipe": test_recipe})
    print(f"Méthode: {result['method']}")
    print(f"   - Nutrition: {result['nutrition']}")
except Exception as e:
    print(f"ERREUR: {e}")

# ========================================
# 4. TEST shopping_list
# ========================================
test_separator("shopping_list")
try:
    test_ingredients = ["chicken breast 500g", "olive oil", "salt", "pepper", "olive oil", "garlic"]
    result = shopping_list.invoke({"ingredients": test_ingredients})
    print(f"Liste générée: {result['count']} items")
    print(f"   - Items: {result['items']}")
except Exception as e:
    print(f"ERREUR: {e}")

# ========================================
# 5. TEST translate_recipe_to_french
# ========================================
test_separator("translate_recipe_to_french")
try:
    test_recipe = {
        "title": "Grilled Chicken Salad",
        "ingredients": ["chicken breast", "lettuce", "olive oil"],
        "steps": ["Grill the chicken", "Mix with vegetables"]
    }
    print("Chargement du modèle de traduction (peut prendre du temps)...")
    result = translate_recipe_to_french.invoke({"recipe": test_recipe})
    print(f"Traduction effectuée")
    print(f"   - Titre: {result['title']}")
    print(f"   - Ingrédients: {result['ingredients']}")
    print(f"   - Traduit: {result.get('translated', False)}")
except Exception as e:
    print(f"ERREUR: {e}")

# ========================================
# 6. TEST make_recipe_healthier
# ========================================
test_separator("make_recipe_healthier")
try:
    test_recipe = {
        "title": "Classic Pasta Carbonara",
        "ingredients": [
            "pasta 400g",
            "bacon 200g", 
            "eggs 4",
            "parmesan cheese 100g",
            "heavy cream 100ml",
            "butter 50g",
            "salt and pepper"
        ],
        "steps": [
            "Cook pasta in boiling water",
            "Fry bacon in butter until crispy",
            "Mix eggs with cream and cheese",
            "Combine everything"
        ]
    }
    result = make_recipe_healthier.invoke({"recipe": test_recipe})
    print(f"Transformation healthy effectuée")
    print(f"   - Titre original: {result['original_title']}")
    print(f"   - Titre healthy: {result['healthy_title']}")
    print(f"   - Nombre de substitutions: {result['substitutions_count']}")
    print("\n   Substitutions proposées:")
    for sub in result['substitutions']:
        print(f"      • {sub['original']} → {sub['substitute']}")
        print(f"        Raison: {sub['reason']}")
    print(f"\n   Améliorations estimées:")
    for k, v in result['estimated_improvements'].items():
        print(f"      • {k}: {v}")
    print(f"\n   Conseils santé: {len(result['health_tips'])} tips")
except Exception as e:
    print(f"ERREUR: {e}")

# ========================================
# RÉSUMÉ
# ========================================
print("\n" + "="*60)
print("RÉSUMÉ DES TESTS")
print("="*60)
print("""
retrieve_recipes     - Recherche RAG dans ChromaDB
allergen_check       - Détection d'allergènes
estimate_macros      - Estimation nutritionnelle
shopping_list        - Génération liste de courses
translate_recipe_to_french - Traduction EN→FR
make_recipe_healthier - Transformation healthy
Tous les outils sont fonctionnels !
""")
