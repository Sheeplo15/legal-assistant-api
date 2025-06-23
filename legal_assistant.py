# ==============================================================================
# ASSISTANT JURIDIQUE IA - SCRIPT FINAL
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
    "France": [".gouv.fr", "service-public.fr", "legifrance.gouv.fr"],
    "Gabon": [".gouv.ga", "dgi.ga", "pme.gouv.ga", "anpigabon.com"],
    "USA": [".gov"],
    "UK": [".gov.uk"],
    "Canada": [".gc.ca", ".ca/en/government"],
    "Cameroun": [".cm", "impots.cm"],
}

# Cache en mémoire simple pour éviter les requêtes répétées et coûteuses
api_cache = {}

# --- MODÈLES DE DONNÉES Pydantic (contrat de l'API) ---
class QueryRequest(BaseModel):
    question: str
    country: str

class AnswerResponse(BaseModel):
    answer: str
    source_url: str | None

# --- INITIALISATION DE L'APPLICATION FastAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- FONCTIONS LOGIQUES DE L'ASSISTANT ---

def get_contextual_query(question: str, country: str) -> str:
    print(f"🌍 Étape 1/5 : Adaptation de la requête pour le pays : {country}...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un expert en droit comparé. Traduis le concept de la question "{question}" dans le jargon juridique et fiscal le plus probable pour le pays '{country}' afin d'optimiser une recherche web. Ne retourne QUE la requête de recherche, sans aucune autre explication."""
        response = model.generate_content(prompt)
        optimized_query = response.text.strip().replace('"', '')
        print(f"   -> Requête optimisée : {optimized_query}")
        return optimized_query
    except Exception as e:
        print(f"   -> Erreur lors de l'optimisation : {e}")
        return question

def search_for_official_sites(question: str, country: str) -> str | None:
    print(f"🔎 Étape 2/5 : Recherche d'un site officiel pour : '{question}'...")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": question, "num": 5})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        results = response.json().get('organic', [])
        if not results:
            print("   -> ❌ Aucun résultat organique renvoyé par Serper.")
            return None
        
        official_keywords = GOVERNMENT_SITES_DATABASE.get(country, ['.gov', '.gouv', 'go.'])
        unwanted_keywords = ['facebook.com', 'youtube.com', 'twitter.com', 'linkedin.com', 'wikipedia.org']
        
        print("   -> 🔬 Analyse des résultats pour trouver une source fiable...")
        for result in results:
            link, title = result['link'], result['title'].lower()
            if any(unwanted in link for unwanted in unwanted_keywords):
                continue
            if any(official in link for official in official_keywords) or any(official in title for official in official_keywords):
                print(f"   -> ✅ Source officielle identifiée : {link}")
                return link
        
        print("   -> ⚠️ Aucun site clairement officiel trouvé. Prise du premier résultat non-indésirable.")
        for result in results:
            if not any(unwanted in result['link'] for unwanted in unwanted_keywords):
                print(f"   -> ✅ Pris par défaut (meilleur effort) : {result['link']}")
                return result['link']
        
        return None
    except requests.exceptions.RequestException as e:
        print(f"   -> Erreur lors de la recherche : {e}")
        return None

def scrape_content(source_url: str) -> str | None:
    print(f"📄 Étape 3/5 : Extraction du contenu de la source...")
    if not source_url: return None
    try:
        if source_url.lower().endswith('.pdf'):
            print(f"   -> Détection d'un PDF. Utilisation du scraper de PDF...")
            response = requests.get(source_url, timeout=30)
            response.raise_for_status()
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            content = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        else:
            print(f"   -> Détection d'une page web. Utilisation de Firecrawl...")
            app_firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
            scraped_data = app_firecrawl.scrape_url(source_url)
            content = scraped_data.markdown

        if content:
            print("   -> ✅ Contenu extrait avec succès.")
            return content
        else:
            print("   -> ❌ Le scraping n'a retourné aucun contenu.")
            return None
    except Exception as e:
        print(f"   -> Erreur lors du scraping : {e}")
        return None

def create_and_search_vector_store(text_content: str, question: str) -> str | None:
    """
    La fonction "Super-Chercheur" (RAG). Elle transforme le document en une base de données
    vectorielle en mémoire, puis y recherche les passages les plus pertinents.
    """
    print("🧠 Étape 4/5 : Création de la carte sémantique (Embeddings)...")
    if not text_content: return None
    try:
        # 1. Chunking : Découper le texte en morceaux
        chunks = [chunk for chunk in text_content.split('\n\n') if len(chunk.strip()) > 100]
        if not chunks:
            print("   -> ❌ Le document n'a pas pu être découpé en paragraphes significatifs.")
            return None
        print(f"   -> Document découpé en {len(chunks)} morceaux.")

        # 2. Embedding : Transformer chaque morceau en vecteur
        embedding_model = 'models/text-embedding-004'
        chunk_embeddings = []
        for i in range(0, len(chunks), 100): # Traitement par lots de 100
            batch = chunks[i:i+100]
            response = genai.embed_content(model=embedding_model, content=batch, task_type="RETRIEVAL_DOCUMENT", title="Texte de loi et obligations fiscales")
            chunk_embeddings.extend(response['embedding'])
            print(f"   -> Lot d'embeddings {i//100 + 1} créé.")
            if len(chunks) > 100: time.sleep(1) # Pause pour respecter les limites de l'API

        print("   -> ✅ Carte sémantique créée.")

        # 3. Retrieval : Chercher dans la carte
        print("   -> 🎯 Recherche des passages les plus pertinents...")
        question_embedding = genai.embed_content(model=embedding_model, content=question, task_type="RETRIEVAL_QUERY")['embedding']
        
        dot_products = np.dot(np.array(chunk_embeddings), question_embedding)
        top_k_indices = np.argsort(dot_products)[-4:][::-1] # Indices des 4 meilleurs passages
        
        relevant_context = "\n---\n".join([chunks[i] for i in top_k_indices])
        print("   -> ✅ Contexte pertinent assemblé.")
        
        return relevant_context

    except Exception as e:
        print(f"   -> ❌ Erreur lors de la recherche vectorielle : {e}")
        return None

def generate_answer_with_gemini(context: str, question: str, country: str) -> str:
    print(f"✍️  Étape 5/5 : Génération de la réponse finale...")
    if not context: return "La source officielle a été analysée, mais aucun passage pertinent n'a pu être identifié pour répondre à cette question. Le document ne traite peut-être pas de ce sujet spécifique."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un assistant juridique IA. Ta mission est de répondre précisément à la question de l'utilisateur en te basant EXCLUSIVEMENT sur le contexte fourni ci-dessous. Contexte : --- {context} ---. Question : "{question}". Pays concerné : {country}. Formule une réponse claire, structurée et professionnelle. Si le contexte ne permet pas de répondre, indique-le clairement."""
        response = model.generate_content(prompt)
        print("   -> ✅ Réponse finale générée.")
        return response.text
    except Exception as e:
        print(f"   -> Erreur lors de la génération de la réponse finale : {e}")
        return "Une erreur est survenue lors de la génération de la réponse finale."


# --- POINT D'ENTRÉE PRINCIPAL DE L'API ---
@app.post("/process_query", response_model=AnswerResponse)
async def get_legal_answer_endpoint(request: QueryRequest):
    user_question = request.question
    user_country = request.country
    cache_key = f"{user_country.lower()}:{user_question.lower()}"
    
    if cache_key in api_cache:
        print("✅ Réponse trouvée dans le cache ! Renvoi instantané.")
        return api_cache[cache_key]

    print("-" * 50)
    print(f"Requête reçue | Pays : {user_country} | Question : {user_question}")
    print("-" * 50)
    
    # Étape 1 & 2 : Trouver la source
    search_query = get_contextual_query(user_question, user_country)
    source_url = search_for_official_sites(search_query, user_country)

    # Étape 3 : Scraper le contenu
    scraped_content = scrape_content(source_url)
    if not scraped_content:
        return AnswerResponse(answer="Impossible de récupérer le contenu de la source officielle.", source_url=source_url)

    # Étape 4 (RAG) : Créer la carte sémantique et trouver le contexte pertinent
    refined_context = create_and_search_vector_store(scraped_content, user_question)
    
    # Étape 5 : Générer la réponse finale à partir de ce contexte de haute qualité
    final_answer = generate_answer_with_gemini(refined_context, user_question, user_country)
    
    response_to_send = AnswerResponse(answer=final_answer, source_url=source_url)
    
    print("💾 Sauvegarde de la réponse dans le cache.")
    api_cache[cache_key] = response_to_send
    
    return response_to_send