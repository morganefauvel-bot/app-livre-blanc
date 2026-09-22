import os
import random
import asyncio
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

# ------------------------------------------------------------------------------
# CONFIGURATION STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Converteo - Pipeline Livre Blanc Multi-Agents",
    page_icon="📚",
    layout="wide"
)

# Modèle par défaut : gemini-2.5-* est en cours de dépréciation (arrêt le
# 16 octobre 2026). gemini-3.5-flash est la version stable actuelle recommandée
# pour ce type de pipeline de génération de contenu. gemini-3.8-flash est le
# modèle le plus récent (GA depuis le 2 septembre 2026), plus capable sur les
# tâches longues/agentiques mais avec moins de recul en production.
DEFAULT_MODEL = "gemini-3.5-flash"
ALT_MODEL = "gemini-3.8-flash"

# ------------------------------------------------------------------------------
# FONCTIONS UTILITAIRES
# ------------------------------------------------------------------------------
def load_prompt(filename: str) -> str:
    path = os.path.join("prompts", filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    st.error(f"Fichier de prompt introuvable : {path}")
    return ""

def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

async def call_agent_with_retry(
    client: genai.Client,
    system_prompt: str,
    user_input: str,
    model: str,
    max_retries: int = 5,
) -> str:
    """Appelle l'API avec re-tentative (backoff exponentiel + jitter).

    Distingue 429 (quota) de 503 (surcharge temporaire) : le quota mérite
    un délai plus long car il ne se libère pas en quelques secondes.
    """
    base_delay = 2
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                )
            )
            return response.text
        except Exception as e:
            err_msg = str(e)
            is_last_attempt = attempt == max_retries - 1
            is_quota = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
            is_overloaded = "503" in err_msg or "UNAVAILABLE" in err_msg

            if not (is_quota or is_overloaded) or is_last_attempt:
                raise e

            # Quota épuisé -> délai plus long (le compteur se réinitialise
            # à la minute), surcharge temporaire -> backoff classique.
            multiplier = 3 if is_quota else 2
            delay = base_delay * (multiplier ** attempt) + random.uniform(0, 1)
            delay = min(delay, 60)  # on plafonne pour ne pas bloquer l'UI indéfiniment
            await asyncio.sleep(delay)  # await, pas time.sleep, sinon ça bloque tout l'event loop

async def run_pipeline(whitepaper_text: str, metadata: str, api_key: str, model: str, progress_callback):
    client = genai.Client(
        api_key=api_key,
        http_options=HttpOptions(api_version="v1")
    )

    prompt_pitch = load_prompt("agent_pitch.txt")
    prompt_linkedin = load_prompt("agent_linkedin.txt")
    prompt_seo = load_prompt("agent_seo_video.txt")

    # Agent 1 : Pitch
    progress_callback(25, "Agent 1/3 : Génération Pitch & LP...")
    input_1 = f"<livre_blanc>\n{whitepaper_text}\n</livre_blanc>\n<metadata>\n{metadata}\n</metadata>"
    result_pitch = await call_agent_with_retry(client, prompt_pitch, input_1, model)

    await asyncio.sleep(1)

    # Agent 2 : LinkedIn
    progress_callback(55, "Agent 2/3 : Kit LinkedIn...")
    input_2 = f"<livre_blanc>\n{whitepaper_text}\n</livre_blanc>\n<pitch_reference>\n{result_pitch}\n</pitch_reference>\n<metadata>\n{metadata}\n</metadata>"
    result_linkedin = await call_agent_with_retry(client, prompt_linkedin, input_2, model)

    await asyncio.sleep(1)

    # Agent 3 : SEO & Vidéo
    progress_callback(85, "Agent 3/3 : SEO & Script Vidéo...")
    input_3 = f"<livre_blanc>\n{whitepaper_text}\n</livre_blanc>"
    result_seo = await call_agent_with_retry(client, prompt_seo, input_3, model)

    return result_pitch, result_linkedin, result_seo

# ------------------------------------------------------------------------------
# INTERFACE STREAMLIT
# ------------------------------------------------------------------------------
st.title("📚 Assistant Diffusion Livre Blanc (Converteo)")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_env = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input("Clé API Gemini :", value=api_key_env, type="password").strip()
    model = st.selectbox(
        "Modèle :",
        [DEFAULT_MODEL, ALT_MODEL],
        help="gemini-3.5-flash : stable, éprouvé. gemini-3.8-flash : plus récent, plus capable sur les tâches longues/agentiques."
    )

col1, col2 = st.columns(2)
with col1:
    uploaded_pdf = st.file_uploader("📄 Livre blanc (PDF)", type=["pdf"])
with col2:
    authors = st.text_input("👥 Auteurs & Métiers", placeholder="Ex: Morgane Fauvel (Lead Consultant)")

if st.button("🚀 Générer le Kit de Diffusion", use_container_width=True):
    if not uploaded_pdf or not authors or not api_key:
        st.warning("Veuillez fournir le PDF, les auteurs et la clé API.")
    else:
        with st.spinner("Extraction du texte PDF..."):
            text = extract_pdf_text(uploaded_pdf)

        bar = st.progress(0)
        status = st.empty()

        def update_p(pct, msg):
            bar.progress(pct)
            status.info(msg)

        try:
            pitch, linkedin, seo = asyncio.run(
                run_pipeline(text, f"Auteurs: {authors}", api_key, model, update_p)
            )
            bar.progress(100)
            status.success("Génération terminée avec succès !")

            st.session_state['full_md'] = f"# KIT DE DIFFUSION\n\n## PARTIE 1 : PITCH & LP\n\n{pitch}\n\n---\n\n## PARTIE 2 : LINKEDIN\n\n{linkedin}\n\n---\n\n## PARTIE 3 : SEO & VIDÉO\n\n{seo}"
            st.session_state['pitch'] = pitch
            st.session_state['linkedin'] = linkedin
            st.session_state['seo'] = seo

        except Exception as e:
            status.error(f"Erreur : {e}")

# Affichage des résultats
if 'full_md' in st.session_state:
    st.markdown("---")
    st.download_button("📥 Télécharger le Rapport Markdown", data=st.session_state['full_md'], file_name="Rapport_Diffusion_Final.md")

    t1, t2, t3, t4 = st.tabs(["Rapport Global", "Pitch & LP", "LinkedIn", "SEO & Vidéo"])
    with t1: st.markdown(st.session_state['full_md'])
    with t2: st.markdown(st.session_state['pitch'])
    with t3: st.markdown(st.session_state['linkedin'])
    with t4: st.markdown(st.session_state['seo'])