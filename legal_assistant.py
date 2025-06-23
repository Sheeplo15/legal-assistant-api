# ==============================================================================
# ASSISTANT JURIDIQUE IA - VERSION FINALE "AGENT STRATÈGE" (CORRIGÉE)
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
    print(f"🤔 Étape 0 : Réflexion stratégique pour '{question}' au '{country}'...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Tu es un assistant de recherche juridique expert. Ta première tâche est de créer un plan de recherche structuré pour répondre à la question suivante : "{question}" pour le pays : "{country}".
        1.  Analyse la question. Est-ce une question simple, factuelle que tu connais déjà (ex: capitale, monnaie) ou une question complexe nécessitant une recherche web ?
        2.  Si la recherche n'est pas nécessaire, mets "requires_search" à false.
        3.  Si la recherche est nécessaire, crée 2 à 3 requêtes de recherche Google optimisées. Pense aux termes juridiques et aux organismes officiels locaux (ex: ANPI pour l'investissement, DGI pour les impôts).
        4.  Liste les domaines web cibles les plus probables pour ces recherches (ex: "anpi-gabon.com", "sante.gouv.ga").
        
        Réponds UNIQUEMENT avec un objet JSON au format suivant :
        {{
          "requires_search": true,
          "reasoning": "La question est complexe et spécifique à la législation du pays, une recherche web ciblée est nécessaire pour garantir une réponse précise.",
          "search_queries": ["statut juridique artisan boulanger {country}", "créer une entreprise de boulangerie ANPI {country}"],
          "target_domains": ["anpi-gabon.com", "sante.gouv.ga", "dgi.ga"]
        }}
        """
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        plan_data = json.loads(json_text)
        plan = SearchPlan(**plan_data)
        print(f"   -> ✅ Plan de recherche créé. Requête : '{plan.search_queries[0]}'")
        return plan
        
    except Exception as e:
        print(f"   -> ❌ Erreur durant la réflexion stratégique : {e}. Utilisation d'un plan par défaut.")
        return SearchPlan(
            requires_search=True,
            reasoning="Le plan stratégique a échoué, utilisation d'une recherche par défaut.",
            search_queries=[question],
            target_domains=[]
        )

def search_for_official_sites(queries: list[str], country: str, target_domains: list[str]) -> str | None:
    print(f"🔎 Étape 1 : Exécution du plan de recherche...")
    for query in queries:
        print(f"   -> Tentative avec la requête : '{query}'")
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 5})
        headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json().get('organic', [])
            if not results: continue

            official_keywords = GOVERNMENT_SITES_DATABASE.get(country, []) + target_domains
            unwanted_keywords = ['facebook.com', 'youtube.com', 'wikipedia.org']
            
            # Priorité 1 : Un lien qui correspond aux domaines cibles
            for result in results:
                if any(domain in result['link'] for domain in official_keywords):
                    print(f"   -> ✅ Source prioritaire trouvée : {result['link']}")
                    return result['link']

            # Priorité 2 : Le premier résultat non-indésirable
            for result in results:
                if not any(unwanted in result['link'] for unwanted in unwanted_keywords):
                    print(f"   -> ⚠️ Source de repli trouvée : {result['link']}")
                    return result['link']
        except Exception as e:
            print(f"   -> ❌ Erreur lors de la tentative avec la requête '{query}': {e}")
            continue
    
    print("   -> ❌ Aucune source fiable trouvée après toutes les tentatives.")
    return None

def scrape_content(source_url: str) -> str | None:
    print(f"📄 Étape 2 : Extraction du contenu de la source...")
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
        print(f"   -> ❌ Erreur lors du scraping : {e}")
        return None

def create_and_search_vector_store(text_content: str, question: str) -> str | None:
    print(f"🎯 Étape 3 : Recherche sémantique dans le document...")
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
    
    plan = create_search_plan(user_question, user_country)

    if not plan.requires_search:
        return AnswerResponse(answer=plan.reasoning, source_url=None)

    source_url = search_for_official_sites(plan.search_queries, user_country, plan.target_domains)
    
    if not source_url:
        return AnswerResponse(answer="Impossible de trouver une source officielle fiable pour cette question.", source_url=None)

    scraped_content = scrape_content(source_url)
    if not scraped_content:
        return AnswerResponse(answer="Impossible de récupérer le contenu de la source officielle trouvée.", source_url=source_url)
    
    refined_context = create_and_search_vector_store(scraped_content, user_question)
    final_answer = generate_answer_with_gemini(refined_context, user_question, user_country)
    
    response_to_send = AnswerResponse(answer=final_answer, source_url=source_url)
    api_cache[cache_key] = response_to_send
    print("💾 Sauvegarde de la réponse dans le cache.")
    
    return response_to_send