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

# --- CONFIGURATION & CACHE ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

GOVERNMENT_SITES_DATABASE = {
    "France": [".gouv.fr", "service-public.fr", "legifrance.gouv.fr"],
    "Gabon": [".gouv.ga", "dgi.ga", "pme.gouv.ga", "anpigabon.com"],
    "USA": [".gov"], "UK": [".gov.uk"], "Canada": [".gc.ca", ".ca/en/government"],
    "Cameroun": [".cm", "impots.cm"],
}

# Cache en mémoire simple pour éviter les requêtes répétées
api_cache = {}

# --- MODÈLES DE DONNÉES Pydantic (pour l'API) ---
class QueryRequest(BaseModel):
    question: str
    country: str

class AnswerResponse(BaseModel):
    answer: str
    source_url: str | None

# --- INITIALISATION DE L'API FastAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- FONCTIONS LOGIQUES ---

def get_contextual_query(question: str, country: str) -> str:
    print(f"🌍 Adaptation de la requête pour le contexte du pays : {country}...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un expert en droit comparé. Ton rôle est de traduire un concept juridique commun dans le jargon spécifique d'un pays donné pour optimiser une recherche web. Concept de base : "{question}". Pays cible : {country}. Ne retourne QUE la requête de recherche optimisée, sans aucune autre explication."""
        response = model.generate_content(prompt)
        optimized_query = response.text.strip().replace('"', '')
        print(f"   -> Requête optimisée : {optimized_query}")
        return optimized_query
    except Exception as e:
        print(f"   -> Erreur lors de l'optimisation, utilisation de la requête originale. Erreur: {e}")
        return question

def search_for_official_sites(question: str, country: str) -> str | None:
    print(f"🔎 Recherche d'un site officiel pour : '{question}'...")
    # ... (Le code de cette fonction est long, donc je le place à la fin pour la clarté)
    # ... mais il sera bien appelé.
    pass

def scrape_pdf_content(url: str) -> str | None:
    if not url: return None
    print(f"📄 Scrapping d'un fichier PDF depuis l'URL : {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        full_text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        if full_text:
            print("✅ Contenu du PDF scrappé avec succès.")
            return full_text
        return None
    except Exception as e:
        print(f"Erreur lors du scraping du PDF : {e}")
        return None

def scrape_website_content(url: str) -> str | None:
    if not url: return None
    print(f"🔥 Scrapping du contenu de la page web : {url}...")
    try:
        app_firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        scraped_data = app_firecrawl.scrape_url(url)
        content = scraped_data.markdown
        if content:
            print("✅ Contenu de la page web scrappé avec succès.")
            return content
        return None
    except Exception as e:
        print(f"Erreur lors du scraping de la page web : {e}")
        return None

def check_document_relevance(text_content: str, question: str) -> bool:
    if not text_content: return False
    print("🔬 Vérification rapide de la pertinence du document...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un expert en recherche. Réponds SEULEMENT par 'OUI' ou 'NON'. Le document suivant (extrait) semble-t-il contenir des informations pertinentes pour répondre à cette question : "{question}" ? --- EXTRAIT DU DOCUMENT --- {text_content[:4000]}"""
        response = model.generate_content(prompt)
        answer = response.text.strip().upper()
        if "OUI" in answer:
            print("✅ Le document semble pertinent. Poursuite de l'analyse approfondie.")
            return True
        else:
            print("❌ Le document ne semble pas pertinent. Arrêt du traitement.")
            return False
    except Exception as e:
        print(f"Erreur lors de la vérification de pertinence : {e}")
        return False

def find_relevant_context_in_text(full_text: str, question: str) -> str | None:
    if not full_text: return None
    print("🧠 Recherche des passages pertinents dans le document long...")
    try:
        clean_text = full_text.encode('utf-8', 'replace').decode('utf-8')
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un assistant de recherche. À partir du texte complet fourni, extrais et retourne UNIQUEMENT les quelques paragraphes ou sections les plus pertinents pour répondre à la question suivante : "{question}". Ne formule pas de réponse. Extrais simplement le texte brut. --- DÉBUT DU TEXTE --- {clean_text[:40000]} --- FIN DU TEXTE ---"""
        response = model.generate_content(prompt)
        print("✅ Passages pertinents extraits.")
        return response.text
    except Exception as e:
        print(f"Erreur lors de la recherche de contexte : {e}")
        return None

def generate_answer_with_gemini(context: str, question: str, country: str) -> str:
    if not context: return "Je n'ai pas pu trouver d'informations précises sur ce sujet dans la source officielle consultée."
    print("✍️ Génération de la réponse finale avec Gemini...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Tu es un assistant juridique. Ta mission est de répondre précisément à la question en te basant EXCLUSIVEMENT sur le contexte fourni. Contexte : --- {context} ---. Question : "{question}". Pays concerné : {country}. Formule une réponse claire et structurée. Si le contexte ne permet pas de répondre, indique-le clairement."""
        response = model.generate_content(prompt)
        print("✅ Réponse finale générée.")
        return response.text
    except Exception as e:
        print(f"Erreur lors de la génération de la réponse finale : {e}")
        return "Une erreur est survenue lors de la génération de la réponse."


# --- POINT D'ENTRÉE DE L'API ---
@app.post("/process_query", response_model=AnswerResponse)
async def get_legal_answer_endpoint(request: QueryRequest):
    user_question = request.question
    user_country = request.country
    cache_key = f"{user_country.lower()}:{user_question.lower()}"
    
    if cache_key in api_cache:
        print("✅ Réponse trouvée dans le cache ! Renvoi instantané.")
        return api_cache[cache_key]

    print("-" * 50)
    print(f"Requête reçue pour le pays : {user_country} | Question : {user_question}")
    
    optimized_question = get_contextual_query(user_question, user_country)
    source_url = search_for_official_sites(optimized_question, user_country)
    
    scraped_content = None
    if source_url:
        if source_url.lower().endswith('.pdf'):
            scraped_content = scrape_pdf_content(source_url)
        else:
            scraped_content = scrape_website_content(source_url)
    
    if not scraped_content or not check_document_relevance(scraped_content, user_question):
        final_answer = "La source officielle trouvée ne semble pas contenir d'informations pertinentes pour répondre à votre question. Veuillez essayer de reformuler votre question."
        return AnswerResponse(answer=final_answer, source_url=source_url)
    
    refined_context = find_relevant_context_in_text(scraped_content, user_question)
    final_answer = generate_answer_with_gemini(refined_context, user_question, user_country)
    
    response_to_send = AnswerResponse(answer=final_answer, source_url=source_url)
    
    print("💾 Sauvegarde de la réponse dans le cache pour la prochaine fois.")
    api_cache[cache_key] = response_to_send
    
    return response_to_send

# Je remets ici la fonction `search_for_official_sites` pour que tout soit dans un seul bloc
def search_for_official_sites(question: str, country: str) -> str | None:
    print(f"🔎 Recherche (via Google/Serper) d'un site officiel pour : '{question}'...")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": question, "num": 5})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        search_results = response.json()
        if 'organic' in search_results and len(search_results['organic']) > 0:
            official_keywords = GOVERNMENT_SITES_DATABASE.get(country, ['.gov', '.gouv', 'go.'])
            unwanted_keywords = ['facebook.com', 'youtube.com', 'twitter.com', 'linkedin.com', 'wikipedia.org']
            print("🔬 Analyse des résultats pour trouver une source fiable...")
            for result in search_results['organic']:
                link, title = result['link'], result['title'].lower()
                if any(unwanted in link for unwanted in unwanted_keywords):
                    print(f"  - Rejeté (indésirable) : {link}")
                    continue
                if any(official in link for official in official_keywords) or any(official in title for official in official_keywords):
                    print(f"✅ Source officielle identifiée : {link}")
                    return link
            print("⚠️ Aucun site clairement officiel trouvé. Tentative avec le premier résultat non-indésirable.")
            for result in search_results['organic']:
                if not any(unwanted in result['link'] for unwanted in unwanted_keywords):
                    print(f"✅ Pris par défaut (meilleur effort) : {result['link']}")
                    return result['link']
            print("❌ Aucun site pertinent et fiable n'a été trouvé après filtrage.")
            return None
        else:
            print("❌ Aucun résultat organique renvoyé par Serper.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la recherche avec Serper : {e}")
        return None

# Le code est maintenant complet et autonome.
# On remplace la fonction `pass` par son implémentation réelle.
get_legal_answer_endpoint.__globals__['search_for_official_sites'] = search_for_official_sites