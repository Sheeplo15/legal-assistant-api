# ==============================================================================
# ASSISTANT JURIDIQUE IA - VERSION FINALE "AGENT STRATÈGE"
# ==============================================================================

import os
import requests
import json
import io
import time
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# --- CONFIGURATION & GLOBALS ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

GOVERNMENT_SITES_DATABASE = {
    "France": [".gouv.fr"], "Gabon": [".gouv.ga"], "USA": [".gov"],
    "UK": [".gov.uk"], "Canada": [".gc.ca"], "Cameroun": [".cm", "impots.cm"],
}
api_cache = {}

# --- MODÈLES DE DONNÉES Pydantic ---
class QueryRequest(BaseModel): question: str; country: str
class AnswerResponse(BaseModel): answer: str; source_url: str | None
class SearchPlan(BaseModel):
    requires_search: bool
    reasoning: str
    search_queries: list[str]
    target_domains: list[str]

# --- INITIALISATION DE L'APPLICATION FastAPI ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- FONCTIONS LOGIQUES DE L'AGENT ---

def create_search_plan(question: str, country: str) -> SearchPlan:
    """
    Étape 0: La couche de réflexion. L'IA utilise sa connaissance générale pour créer un plan d'action.
    """
    print(f"🤔 Étape 0 : Réflexion stratégique pour '{question}' au '{country}'...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        # CORRECTION ICI : Les accolades de l'exemple JSON sont doublées pour fonctionner dans une f-string.
        prompt = f"""
        Tu es un assistant de recherche juridique expert. Ta première tâche est de créer un plan de recherche structuré pour répondre à la question suivante : "{question}" pour le pays : "{country}".
        1.  Analyse la question. Est-ce une question simple, factuelle que tu connais déjà (ex: capitale, monnaie) ou une question complexe nécessitant une recherche web ?
        2.  Si la recherche n'est pas nécessaire, mets "requires_search" à false et explique pourquoi dans "reasoning".
        3.  Si la recherche est nécessaire, crée 2 à 3 requêtes de recherche Google optimisées. Pense aux termes juridiques et aux organismes officiels locaux (ex: ANPI pour l'investissement, DGI pour les impôts).
        4.  Liste les domaines web cibles les plus probables pour ces recherches (ex: "anpi-gabon.com", "sante.gouv.ga").
        
        Réponds UNIQUEMENT avec un objet JSON au format suivant :
        {{
          "requires_search": boolean,
          "reasoning": "string",
          "search_queries": ["string"],
          "target_domains": ["string"]
        }}
        """
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        plan_data = json.loads(json_text)
        plan = SearchPlan(**plan_data)
        print(f"   -> ✅ Plan de recherche créé. Requête(s) : {plan.search_queries}")
        return plan
        
    except Exception as e:
        print(f"   -> ❌ Erreur durant la réflexion stratégique : {e}. Utilisation d'un plan par défaut.")
        return SearchPlan(
            requires_search=True,
            reasoning="Le plan stratégique a échoué, utilisation d'une recherche par défaut.",
            search_queries=[question],
            target_domains=[]
        )

def find_multiple_official_sites(queries: list[str], country: str, target_domains: list[str]) -> list[str]:
    """
    Exécute les requêtes du plan et retourne une LISTE de sources fiables potentielles, triées par pertinence.
    """
    print(f"🔎 Étape 1 : Exécution du plan de recherche multi-sources...")
    potential_sources = []
    unwanted_keywords = ['facebook.com', 'youtube.com', 'wikipedia.org', 'scribd.com', 'researchgate.net', 'twitter.com', 'linkedin.com']
    
    for query in queries:
        print(f"   -> Tentative avec la requête : '{query}'")
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 3}) # On prend 3 résultats par requête
        headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json().get('organic', [])
            
            for result in results:
                link = result['link']
                if any(unwanted in link for unwanted in unwanted_keywords):
                    continue
                if link not in potential_sources: # Éviter les doublons
                    potential_sources.append(link)
        except Exception as e:
            print(f"   -> ❌ Erreur lors de la tentative avec la requête '{query}': {e}")
            continue
    
    # On trie les sources pour donner la priorité aux domaines cibles et officiels
    official_keywords = GOVERNMENT_SITES_DATABASE.get(country, []) + target_domains
    potential_sources.sort(key=lambda link: any(domain in link for domain in official_keywords), reverse=True)
    
    print(f"   -> ✅ Recherche terminée. {len(potential_sources)} sources potentielles trouvées et triées.")
    return potential_sources[:3] # On retourne les 3 meilleures sources

def scrape_content(source_url: str) -> str | None:
    print(f"      -> 📄 Extraction du contenu de : {source_url}...")
    if not source_url: return None
    try:
        if source_url.lower().endswith('.pdf'):
            response = requests.get(source_url, timeout=30)
            response.raise_for_status()
            reader = PdfReader(io.BytesIO(response.content))
            return "".join(page.extract_text() for page in reader.pages if page.extract_text())
        else:
            return FirecrawlApp(api_key=FIRECRAWL_API_KEY).scrape_url(source_url).markdown
    except Exception as e:
        print(f"      -> ❌ Erreur lors du scraping : {e}")
        return None

def validate_content_relevance(text_content: str, question: str) -> bool:
    """
    Vérifie rapidement si le contenu d'un site est pertinent pour la question.
    """
    if not text_content: return False
    print(f"      -> 🔬 Validation rapide de la pertinence du contenu...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Analyse cet extrait. Est-il pertinent pour répondre à la question "{question}"? Réponds SEULEMENT par 'OUI' ou 'NON'. --- EXTRAIT --- {text_content[:4000]}"""
        response = model.generate_content(prompt)
        return "OUI" in response.text.strip().upper()
    except:
        return False

def create_and_search_vector_store(text_content: str, question: str) -> str | None:
    print(f"🎯 Étape 3 : Recherche sémantique dans le document validé...")
    if not text_content: return None
    try:
        chunks = [chunk for chunk in text_content.split('\n\n') if len(chunk.strip()) > 100]
        if not chunks: return None

        embedding_model = 'models/text-embedding-004'
        chunk_embeddings = []
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i+100]
            response = genai.embed_content(model=embedding_model, content=batch, task_type="RETRIEVAL_DOCUMENT")
            chunk_embeddings.extend(response['embedding'])
            if len(chunks) > 100: time.sleep(1.1)

        question_embedding = genai.embed_content(model=embedding_model, content=question, task_type="RETRIEVAL_QUERY")['embedding']
        dot_products = np.dot(np.array(chunk_embeddings), question_embedding)
        top_k_indices = np.argsort(dot_products)[-4:][::-1]
        
        relevant_context = "\n---\n".join([chunks[i] for i in top_k_indices])
        return relevant_context
    except Exception as e:
        print(f"   -> ❌ Erreur lors de la recherche vectorielle : {e}")
        return None

def generate_answer_with_gemini(context: str, question: str, country: str) -> str:
    print(f"✍️  Étape 4 : Génération de la réponse finale...")
    if not context: return "La source officielle a été analysée, mais aucun passage pertinent n'a pu être identifié pour répondre à cette question."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un assistant juridique IA. Réponds précisément à la question en te basant EXCLUSIVEMENT sur le contexte fourni. Contexte: --- {context} ---. Question: "{question}". Pays: {country}. Formule une réponse claire et structurée."""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"   -> ❌ Erreur lors de la génération finale : {e}")
        return "Une erreur est survenue lors de la génération de la réponse finale."

@app.post("/process_query", response_model=AnswerResponse)
async def get_legal_answer_endpoint(request: QueryRequest):
    user_question = request.question
    user_country = request.country
    cache_key = f"{user_country.lower()}:{user_question.lower()}"
    if cache_key in api_cache:
        print("✅ Réponse trouvée dans le cache ! Renvoi instantané.")
        return api_cache[cache_key]

    print("-" * 50); print(f"Requête reçue | Pays : {user_country} | Question : {user_question}"); print("-" * 50)
    
    # Étape 0 : Réflexion
    plan = create_search_plan(user_question, user_country)
    if not plan.requires_search:
        return AnswerResponse(answer=plan.reasoning, source_url=None)

    # Étape 1 : Recherche multi-sources
    source_urls = find_multiple_official_sites(plan.search_queries, user_country, plan.target_domains)
    if not source_urls:
        return AnswerResponse(answer="Impossible de trouver une source officielle fiable après une recherche approfondie.", source_url=None)

    # Étape 2 : Boucle de validation et scraping
    golden_source_url = None
    scraped_content = None
    
    print(f"🔗 Étape 2 : Validation des {len(source_urls)} sources trouvées...")
    for i, url in enumerate(source_urls):
        print(f"   -> Tentative avec la source {i+1}/{len(source_urls)} : {url}")
        content = scrape_content(url)
        if content and validate_content_relevance(content, user_question):
            print(f"      -> ✅ Source validée comme pertinente.")
            golden_source_url = url
            scraped_content = content
            break # On a trouvé notre source, on arrête de chercher
        else:
            print(f"      -> ❌ Source jugée non pertinente ou impossible à lire.")
    
    if not golden_source_url:
        return AnswerResponse(answer="Plusieurs sources officielles ont été trouvées, mais aucune ne semble contenir la réponse à votre question spécifique.", source_url=str(source_urls))
    
    # Étape 3 & 4 : RAG sur la source validée et génération de la réponse
    refined_context = create_and_search_vector_store(scraped_content, user_question)
    final_answer = generate_answer_with_gemini(refined_context, user_question, user_country)
    
    response_to_send = AnswerResponse(answer=final_answer, source_url=golden_source_url)
    api_cache[cache_key] = response_to_send
    print("💾 Sauvegarde de la réponse dans le cache.")
    
    return response_to_send