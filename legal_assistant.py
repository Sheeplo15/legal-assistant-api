import os
import requests
import json
import google.generativeai as genai
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from fastapi import FastAPI
from pydantic import BaseModel

# --- CHARGEMENT DES CLÉS ET CONFIGURATION ---
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
}

# --- FASTAPI SETUP ---
app = FastAPI()

# --- STRUCTURES DES DONNÉES ---
class QueryRequest(BaseModel):
    question: str
    country: str

class AnswerResponse(BaseModel):
    answer: str
    source_url: str | None

# --- FONCTIONS PRINCIPALES ---

def get_contextual_query(question: str, country: str) -> str:
    print(f"🌍 Adaptation de la requête pour le contexte du pays : {country}...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Tu es un expert en droit comparé. Ton rôle est de traduire un concept juridique commun
        dans le jargon spécifique d'un pays donné pour optimiser une recherche web.

        Concept de base : "{question}"
        Pays cible : {country}

        Reformule la question ci-dessus en une requête de recherche Google optimale en utilisant les termes les plus probables pour le pays cible.

        Ne retourne QUE la requête de recherche optimisée, sans aucune autre explication.
        """
        response = model.generate_content(prompt)
        optimized_query = response.text.strip().replace('"', '')
        print(f"   -> Requête optimisée : {optimized_query}")
        return optimized_query
    except Exception as e:
        print(f"   -> Erreur lors de l'optimisation: {e}")
        return question

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
                link = result['link']
                title = result['title'].lower()

                if any(unwanted in link for unwanted in unwanted_keywords):
                    print(f"  - Rejeté (indésirable) : {link}")
                    continue

                if any(official in link for official in official_keywords) or any(official in title for official in official_keywords):
                    print(f"✅ Source officielle identifiée : {link}")
                    return link

                print(f"  - Ignoré (non-officiel au premier abord) : {link}")

            print("⚠️ Aucun site clairement officiel trouvé. Tentative avec le premier résultat non-indésirable.")
            for result in search_results['organic']:
                link = result['link']
                if not any(unwanted in link for unwanted in unwanted_keywords):
                    print(f"✅ Pris par défaut (meilleur effort) : {link}")
                    return link

            print("❌ Aucun site pertinent et fiable n'a été trouvé après filtrage.")
            return None
        else:
            print("❌ Aucun résultat organique renvoyé par Serper.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la recherche avec Serper : {e}")
        return None

def scrape_website_content(url: str) -> str | None:
    if not url:
        return None
    print(f"🔥 Scrapping du contenu de l'URL : {url}...")
    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        scraped_data = app.scrape_url(url)
        content = scraped_data.markdown
        if content:
            print("✅ Contenu scrappé avec succès.")
            return content
        else:
            print("❌ Le scraping n'a retourné aucun contenu.")
            return None
    except Exception as e:
        print(f"Erreur lors du scraping avec Firecrawl : {e}")
        return None

def generate_answer_with_gemini(context: str, question: str, country: str) -> str:
    if not context:
        return "Je n'ai pas pu extraire d'informations de la source pour répondre à votre question."

    print("🧠 Génération de la réponse avec Gemini...")
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    Tu es un assistant juridique IA. Ta mission est de répondre de manière précise et factuelle
    à la question de l'utilisateur en te basant EXCLUSIVEMENT sur le texte source fourni ci-dessous,
    qui provient d'un site web gouvernemental.

    **Texte source scrappé :**
    ---
    {context}
    ---

    **Pays concerné :** {country}

    **Question de l'utilisateur :** "{question}"

    **Instructions :**
    1. Lis attentivement le texte source.
    2. Formule une réponse claire et structurée en français qui répond directement à la question.
    3. Ne mentionne PAS que tu te bases sur un "texte source fourni". Agis comme si tu connaissais l'information.
    4. Si l'information n'est PAS présente dans le texte source, indique clairement : "Je n'ai pas pu trouver d'informations précises sur ce sujet dans la source officielle consultée."
    5. Ne jamais inventer d'informations. La précision est cruciale.
    """

    try:
        response = model.generate_content(prompt)
        print("✅ Réponse générée.")
        return response.text
    except Exception as e:
        print(f"Erreur lors de la génération avec Gemini : {e}")
        return "Une erreur est survenue lors de la génération de la réponse."

# --- ENDPOINT PRINCIPAL ---
@app.post("/get-legal-answer", response_model=AnswerResponse)
async def get_legal_answer_endpoint(request: QueryRequest):
    user_question = request.question
    user_country = request.country

    print("-" * 50)
    print(f"Requête reçue pour le pays : {user_country}")
    print(f"Question initiale : {user_question}")
    print("-" * 50)

    optimized_question = get_contextual_query(user_question, user_country)
    source_url = search_for_official_sites(optimized_question, user_country)
    scraped_content = scrape_website_content(source_url)
    final_answer = generate_answer_with_gemini(scraped_content, user_question, user_country)

    return AnswerResponse(answer=final_answer, source_url=source_url)
