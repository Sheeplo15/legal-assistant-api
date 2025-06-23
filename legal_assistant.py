import os
import io
import json
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from firecrawl import FirecrawlApp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CHARGEMENT DES VARIABLES D'ENVIRONNEMENT ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- CONFIGURATION DES SITES GOUVERNEMENTAUX ---
GOVERNMENT_SITES_DATABASE = {
    "Cameroun": [".gouv.cm", "minfi.cm", "minefop.gov.cm", "primeminister.cm"],
    "Côte d'Ivoire": [".gouv.ci", "service-public.gouv.ci", "finances.gouv.ci", "dgi.gouv.ci"],
    "Gabon": [".gouv.ga", "dgi.ga", "pme.gouv.ga", "anpigabon.com"],
    "Sénégal": [".gouv.sn", "service-public.gouv.sn", "dgi.gouv.sn", "impotsetdomaines.gouv.sn"]
}


# --- DÉFINITION DE L’APPLICATION ---
app = FastAPI()

# --- CONFIGURATION CORS ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHÉMAS DE DONNÉES ---
class QueryRequest(BaseModel):
    question: str
    country: str

class AnswerResponse(BaseModel):
    answer: str
    source_url: str | None

# --- FONCTIONS ---

def get_contextual_query(question: str, country: str) -> str:
    print(f"🌍 Adaptation de la requête pour le pays : {country}")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un expert en droit comparé. Ton rôle est de traduire un concept juridique commun dans le jargon spécifique d'un pays donné pour optimiser une recherche web. Concept de base : "{question}". Pays cible : {country}. Reformule la question en une requête de recherche Google optimale. Ne retourne QUE la requête de recherche, sans aucune autre explication."""
        response = model.generate_content(prompt)
        optimized_query = response.text.strip().replace('"', '')
        print(f"   -> Requête optimisée : {optimized_query}")
        return optimized_query
    except Exception as e:
        print(f"   -> Erreur Gemini (fallback sur question originale) : {e}")
        return question

def search_for_official_sites(question: str, country: str) -> str | None:
    print(f"🔎 Recherche d’un site officiel pour : '{question}'...")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": question, "num": 5})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()

        if 'organic' not in results or not results['organic']:
            print("❌ Aucun résultat organique.")
            return None

        official_keywords = GOVERNMENT_SITES_DATABASE.get(country, ['.gov', '.gouv', 'go.'])
        unwanted = ['facebook.com', 'youtube.com', 'twitter.com', 'linkedin.com', 'wikipedia.org']

        for r in results['organic']:
            link = r['link']
            title = r['title'].lower()
            if any(u in link for u in unwanted):
                print(f"  - Ignoré (non pertinent) : {link}")
                continue
            if any(k in link for k in official_keywords) or any(k in title for k in official_keywords):
                print(f"✅ Source officielle trouvée : {link}")
                return link

        # Si aucun lien officiel, retour par défaut
        for r in results['organic']:
            link = r['link']
            if not any(u in link for u in unwanted):
                print(f"⚠️ Source par défaut : {link}")
                return link

        print("❌ Aucun lien pertinent trouvé.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Erreur avec Serper : {e}")
        return None

def scrape_pdf_content(url: str) -> str | None:
    if not url: return None
    print(f"📄 Téléchargement du PDF : {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
        if full_text:
            print("✅ Texte extrait du PDF.")
            return full_text
        else:
            print("❌ PDF vide ou illisible.")
            return None
    except Exception as e:
        print(f"Erreur lors du scraping PDF : {e}")
        return None

def scrape_website_content(url: str) -> str | None:
    if not url: return None
    print(f"🔥 Scraping du site web : {url}")
    try:
        app_firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        scraped_data = app_firecrawl.scrape_url(url)
        content = scraped_data.markdown
        if content:
            print("✅ Contenu extrait du site.")
            return content
        else:
            print("❌ Aucun contenu trouvé.")
            return None
    except Exception as e:
        print(f"Erreur Firecrawl : {e}")
        return None

def generate_answer_with_gemini(context: str, question: str, country: str) -> str:
    if not context:
        return "Je n'ai pas pu extraire d'informations de la source pour répondre à votre question."
    print("🧠 Génération de réponse avec Gemini...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Tu es un assistant juridique IA. Ta mission est de répondre de manière précise et factuelle à la question de l'utilisateur en te basant EXCLUSIVEMENT sur le texte source fourni.
        Pays concerné : {country}
        Question : "{question}"
        Texte source : --- {context} ---
        Instructions : Lis le texte. Formule une réponse claire et structurée. Si l'information n'est pas présente, indique clairement : "Je n'ai pas pu trouver d'informations précises sur ce sujet dans la source officielle consultée." Ne jamais inventer d'informations.
        """
        response = model.generate_content(prompt)
        print("✅ Réponse générée.")
        return response.text
    except Exception as e:
        print(f"Erreur Gemini : {e}")
        return "Une erreur est survenue lors de la génération de la réponse."

# --- ENDPOINT PRINCIPAL ---
@app.post("/process_query", response_model=AnswerResponse)
async def get_legal_answer_endpoint(request: QueryRequest):
    user_question = request.question
    user_country = request.country
    print("-" * 50)
    print(f"🌐 Nouvelle requête reçue | Pays : {user_country} | Question : {user_question}")
    print("-" * 50)

    optimized_question = get_contextual_query(user_question, user_country)
    source_url = search_for_official_sites(optimized_question, user_country)

    # --- Aiguillage intelligent PDF / Web ---
    scraped_content = None
    if source_url:
        if source_url.lower().endswith('.pdf'):
            print("📌 Type de source détecté : PDF")
            scraped_content = scrape_pdf_content(source_url)
        else:
            print("📌 Type de source détecté : Page web")
            scraped_content = scrape_website_content(source_url)

    final_answer = generate_answer_with_gemini(scraped_content, user_question, user_country)

    return AnswerResponse(answer=final_answer, source_url=source_url)
