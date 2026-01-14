
"""
RÔLE :
Ce fichier implémente l'agent Agentic RAG.

CE QU'IL FAIT :
- Initialise le LLM via OpenAI (tool-calling stable)
- Enregistre les outils disponibles
- Permet au LLM de décider quand appeler un outil
- Orchestre retrieval, traduction, vérif allergènes, macros, liste de courses

UTILISATION :
- Appelé par app.py / interface.py pour répondre aux requêtes utilisateur
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langsmith import Client

from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from agent.prompts import SYSTEM_PROMPT, OUTPUT_FORMAT
from agent.tools import (
    retrieve_recipes,
    translate_recipe_to_french,
    allergen_check,
    estimate_macros,
    shopping_list,
    make_recipe_healthier,
)


def build_agent(model: str = "gpt-4o-mini", temperature: float = 0.2, verbose: bool = False) -> AgentExecutor:
    """
    Construit l'agent tool-calling.
    """
    load_dotenv()
    # Initialisation LangSmith (tracing)
    if os.getenv("LANGCHAIN_TRACING_V2") == "true" and os.getenv("LANGCHAIN_API_KEY"):
        client = Client()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY manquant. Mets-le dans .env ou dans tes variables d'environnement.")

    llm = ChatOpenAI(model=model, temperature=temperature)

    tools = [
        retrieve_recipes,
        translate_recipe_to_french,
        allergen_check,
        estimate_macros,
        shopping_list,
        make_recipe_healthier,
    ]

    full_system = SYSTEM_PROMPT + "\n\n" + OUTPUT_FORMAT

    from langchain.prompts import MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", full_system),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose)


def run_agent(user_input: str, model: str = "gpt-4o-mini") -> str:
    """
    Fonction simple pour l'UI/CLI.
    """
    executor = build_agent(model=model, verbose=False)
    result = executor.invoke({"input": user_input})
    return result["output"]
