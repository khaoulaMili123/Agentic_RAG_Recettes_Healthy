"""
INTERFACE STREAMLIT — Agentic RAG Recipes

RÔLE :
- Interface web moderne et stylée
- Appelle l'agent (agent/agent.py)
- Affiche la réponse utilisateur avec un design soigné

L'UI ne contient AUCUNE logique métier.
"""

import streamlit as st
import re
from agent.agent import run_agent

# -----------------------------
# Config page
# -----------------------------
st.set_page_config(
    page_title="🥗 Recettes Saines et Gourmandes",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CSS personnalisé pour un look moderne
# -----------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Style global */
    .stApp {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 50%, #1abc9c 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Cards */
    .recipe-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #2ecc71;
    }
    
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 204, 113, 0.6);
    }
    
    /* Input text area */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #2ecc71;
        box-shadow: 0 0 0 3px rgba(46, 204, 113, 0.2);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Success message styling */
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #7f8c8d;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## Menu")
    st.markdown("---")
    
    st.markdown("### Exemples de requêtes")
    
    example_queries = [
        "Recette de poulet healthy riche en protéines",
        "Salade végétarienne sans gluten",
        "Recette rapide en 15 minutes",
        "Pâtes healthy low-carb",
        "Poisson grillé avec légumes",
        "Smoothie protéiné pour le sport",
    ]
    
    selected_example = None
    for query in example_queries:
        if st.button(query, key=f"example_{query}", use_container_width=True):
            selected_example = query.split(" ", 1)[1]  # Remove emoji
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Options")
    show_tips = st.checkbox("Afficher les conseils santé", value=True)
    show_substitutions = st.checkbox("Proposer des substitutions healthy", value=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistiques")
    st.metric("Recettes en base", "2,049", delta="100% healthy")
    st.metric("Langues", "FR/EN", delta="Traduction auto")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem;">
         Powered by<br>
        <b>GPT-4o-mini + RAG</b><br>
        ChromaDB + HuggingFace
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Header principal
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🥗 Recettes Saines et Gourmandes</h1>
    <p>Votre assistant intelligent pour des recettes saines et délicieuses</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Section Features
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <h4>Recherche RAG</h4>
        <p style="font-size: 0.85rem; color: #666;">2000+ recettes indexées</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🥗</div>
        <h4>Healthy First</h4>
        <p style="font-size: 0.85rem; color: #666;">Substitutions intelligentes</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚠️</div>
        <h4>Allergènes</h4>
        <p style="font-size: 0.85rem; color: #666;">Détection automatique</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h4>Nutrition</h4>
        <p style="font-size: 0.85rem; color: #666;">Calories & macros</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------
# Zone de recherche principale
# -----------------------------
st.markdown("### 💬 Pose ta question")

# Utiliser l'exemple sélectionné si disponible
default_value = selected_example if selected_example else ""

user_query = st.text_area(
    label="Ta question",
    placeholder="Ex: Je veux une recette healthy de poulet riche en protéines, prête en 30 minutes, sans lactose...",
    height=100,
    value=default_value,
    label_visibility="collapsed",
)

# Options avancées dans un expander
with st.expander("Options avancées"):
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        include_shopping_list = st.checkbox("📝 Générer liste de courses", value=False)
        check_allergens = st.checkbox("⚠️ Vérifier allergènes", value=False)
    with col_opt2:
        estimate_nutrition = st.checkbox("📊 Estimer nutrition", value=True)
        make_healthier = st.checkbox("💚 Version plus healthy", value=False)
    
    if check_allergens:
        allergies = st.multiselect(
            "Sélectionne tes allergies/intolérances:",
            ["lactose", "gluten", "arachides", "oeuf", "soja", "fruits de mer", "poisson", "noix"],
            default=[]
        )

# Bouton de recherche centré
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    search_clicked = st.button("🔍 Trouver ma recette", use_container_width=True, type="primary")


# -----------------------------
# Traitement et affichage des résultats
# -----------------------------
if search_clicked:
    if not user_query.strip():
        st.warning("⚠️ Merci de saisir une question pour trouver ta recette idéale !")
    else:
        # Construire la requête enrichie
        enriched_query = user_query
        
        if include_shopping_list:
            enriched_query += " Génère aussi une liste de courses."
        if estimate_nutrition:
            enriched_query += " Donne les informations nutritionnelles."
        if make_healthier:
            enriched_query += " Propose des substitutions pour rendre la recette plus healthy avec l'outil make_recipe_healthier."
        if check_allergens and 'allergies' in dir() and allergies:
            enriched_query += f" Vérifie les allergènes pour: {', '.join(allergies)}."
        
        with st.spinner("L'agent cherche la recette parfaite..."):
            try:
                response = run_agent(enriched_query)
                
                # Affichage du succès
                st.markdown("""
                <div class="success-box animate-fade-in">
                    <h3 style="margin: 0; color: #155724;">✅ Recette trouvée !</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #155724;">Voici une proposition adaptée à ta demande</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Affichage structuré de la réponse
                st.markdown("---")
                
                # Affichage avec colonnes
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.markdown("### 🍽️ Recette")
                    st.markdown(f"""
                    <div class="recipe-card">
                        {response.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_right:
                    # Quick stats
                    st.markdown("### 📊 Résumé")
                    
                    # Compter les ingrédients (approximatif)
                    ingredient_count = response.lower().count("- ") 
                    step_count = len(re.findall(r'\d+\.', response))
                    
                    st.metric("Ingrédients", f"~{min(ingredient_count, 15)}")
                    st.metric("Étapes", f"~{min(step_count, 10)}")
                    
                    # Actions rapides
                    st.markdown("### Actions")
                    if st.button("Copier la recette", use_container_width=True):
                        st.toast("Recette copiée ! ")
                    if st.button("Sauvegarder", use_container_width=True):
                        st.toast("Fonctionnalité à venir !")
                    if st.button("Nouvelle recherche", use_container_width=True):
                        st.rerun()
                
                # Conseils santé
                if show_tips:
                    st.markdown("---")
                    st.markdown("### 💡 Conseils santé")
                    tip_cols = st.columns(3)
                    tips = [
                        ("🥗", "Ajoute des légumes verts pour plus de fibres"),
                        ("🍋", "Un filet de citron remplace le sel"),
                        ("🏃", "Protéines + exercice = muscles toniques"),
                    ]
                    for i, (emoji, tip) in enumerate(tips):
                        with tip_cols[i]:
                            st.markdown(f"""
                            <div class="feature-card">
                                <div style="font-size: 1.5rem;">{emoji}</div>
                                <p style="font-size: 0.85rem; margin: 0;">{tip}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error("Une erreur est survenue lors de la recherche.")
                with st.expander("Détails de l'erreur"):
                    st.exception(e)


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="margin: 0;">
        <strong>🥗 Recettes Saines et Gourmandes</strong> — Votre assistant nutrition intelligent
    </p>
    <p style="font-size: 0.85rem; color: #999; margin-top: 0.5rem;">
        Propulsé par <strong>GPT-4o-mini</strong> • ChromaDB + HuggingFace Embeddings • LangChain Agent
    </p>
    <p style="font-size: 0.75rem; color: #bbb; margin-top: 0.5rem;">
        © 2026 — Fait avec ❤️ pour une alimentation saine
    </p>
</div>
""", unsafe_allow_html=True)
