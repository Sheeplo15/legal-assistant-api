# ==============================================================================
# ASSISTANT JURIDIQUE IA - SCRIPT FINAL
# ==============================================================================

import os
import requests
import json
import io
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
    print(f"🌍 Étape 1/6 : Adaptation de la requête pour le pays : {country}...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un expert en droit comparé. Traduis le concept de la question "{question}" dans le jargon juridique et fiscal le plus probable pour le pays '{country}' afin d'optimiser une recherche web. Ne retourne QUE la requête de recherche, sans aucune autre explication."""
        response = model.generate_content(prompt)
        optimized_query = response.text.strip().replace('"', '')
        print(f"   -> Requête optimisée : {optimized_query}")
        return optimized_query
    except Exception as e:
        print(f"   -> Erreur lors de l'optimisation, utilisation de la requête originale. Erreur: {e}")
        return question

def search_for_official_sites(question: str, country: str) -> str | None:
    print(f"🔎 Étape 2/6 : Recherche d'un site officiel pour : '{question}'...")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": question, "num": 5})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        results = response.json().get('organic', [])
        if not results:
            print("❌ Aucun résultat organique renvoyé par Serper.")
            return None
        
        official_keywords = GOVERNMENT_SITES_DATABASE.get(country, ['.gov', '.gouv', 'go.'])
        unwanted_keywords = ['facebook.com', 'youtube.com', 'twitter.com', 'linkedin.com', 'wikipedia.org']
        
        print("   -> Analyse des résultats pour trouver une source fiable...")
        for result in results:
            link, title = result['link'], result['title'].lower()
            if any(unwanted in link for unwanted in unwanted_keywords): continue
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
        print(f"   -> Erreur lors de la recherche avec Serper : {e}")
        return None

def scrape_content(source_url: str) -> str | None:
    print(f"📄 Étape 3/6 : Extraction du contenu de la source...")
    if not source_url: return None
    try:
        if source_url.lower().endswith('.pdf'):
            print(f"   -> Détection d'un PDF. Utilisation du scraper de PDF...")
            response = requests.get(source_url, timeout=30)
            response.raise_for_status()
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PdfReader(pdf_file)
            content = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
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

def find_local_terminology(text_content: str, base_question: str) -> str | None:
    print(f"🕵️  Étape 4/6 : Étape Détective - Recherche de la terminologie locale...")
    if not text_content: return None
    try:
        clean_text = text_content.encode('utf-8', 'replace').decode('utf-8')
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Analyse cet extrait de texte juridique. Ma question de base est : "{base_question}". Trouve le nom officiel spécifique utilisé dans CE texte pour désigner un régime fiscal simplifié pour petites entreprises ou entrepreneurs individuels. Exemples possibles : "régime de l'entreprenant", "impôt libératoire". Réponds SEULEMENT avec le nom officiel trouvé, ou "Non trouvé" si aucun n'est présent. --- EXTRAIT --- {clean_text[:40000]}"""
        response = model.generate_content(prompt)
        terminology = response.text.strip()
        if "non trouvé" in terminology.lower() or len(terminology) > 100:
            print("   -> Aucune terminologie spécifique trouvée.")
            return None
        else:
            print(f"   -> ✅ Terminologie locale identifiée : '{terminology}'")
            return terminology
    except Exception as e:
        print(f"   -> Erreur lors de la recherche de terminologie : {e}")
        return None

def find_relevant_context_in_text(full_text: str, question: str) -> str | None:
    print(f"🎯 Étape 5/6 : Recherche des passages pertinents avec la question ciblée...")
    if not full_text: return None
    try:
        clean_text = full_text.encode('utf-8', 'replace').decode('utf-8')
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un assistant de recherche. À partir du texte complet fourni, extrais et retourne UNIQUEMENT les quelques paragraphes ou sections les plus pertinents pour répondre à la question : "{question}". N'invente rien. Ne résume pas. Extrais seulement le texte brut. --- DÉBUT DU TEXTE --- {clean_text[:40000]} --- FIN DU TEXTE ---"""
        response = model.generate_content(prompt)
        context = response.text.strip()
        if context:
            print("   -> ✅ Passages pertinents extraits.")
            return context
        else:
            print("   -> ❌ Aucun passage pertinent trouvé.")
            return None
    except Exception as e:
        print(f"   -> Erreur lors de l'extraction de contexte : {e}")
        return None

def generate_answer_with_gemini(context: str, question: str, country: str) -> str:
    print(f"✍️  Étape 6/6 : Génération de la réponse finale...")
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
    
    # Étape 1 : Optimiser la question de recherche web
    search_query = get_contextual_query(user_question, user_country)
    
    # Étape 2 : Trouver la meilleure source
    source_url = search_for_official_sites(search_query, user_country)
    
    # Étape 3 : Extraire le contenu brut
    scraped_content = scrape_content(source_url)
    if not scraped_content:
        return AnswerResponse(answer="Impossible de récupérer le contenu de la source officielle.", source_url=source_url)

    # Étape 4 (Détective) : Découvrir la terminologie locale
    local_term = find_local_terminology(scraped_content, user_question)
    
    context_search_question = f"Quelles sont les obligations du '{local_term}' ?" if local_term else user_question

    # Étape 5 : Extraire les passages pertinents
    refined_context = find_relevant_context_in_text(scraped_content, context_search_question)
    
    # Étape 6 : Générer la réponse finale
    final_answer = generate_answer_with_gemini(refined_context, user_question, user_country)
    
    response_to_send = AnswerResponse(answer=final_answer, source_url=source_url)
    
    print("💾 Sauvegarde de la réponse dans le cache pour la prochaine fois.")
    api_cache[cache_key] = response_to_send
    
    return response_to_send