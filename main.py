import os
import asyncio
from pypdf import PdfReader
from google import genai
from google.genai import types

# Initialisation du client SDK Gemini
client = genai.Client()

MODEL_NAME = "gemini-2.5-flash"

def load_prompt(filename: str) -> str:
    path = os.path.join("prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

async def run_subagent(agent_name: str, system_prompt: str, user_input: str) -> str:
    print(f"🚀 [Sous-agent : {agent_name}] Lancement de la génération...")
    response = await client.aio.models.generate_content(
        model=MODEL_NAME,
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        )
    )
    print(f"✅ [Sous-agent : {agent_name}] Terminé !")
    return response.text

async def orchestrate_whitepaper_pipeline(whitepaper_text: str, metadata: str):
    print("\n--- DÉBUT DU PIPELINE MULTI-AGENTS ---\n")
    
    prompt_pitch = load_prompt("agent_pitch.txt")
    prompt_linkedin = load_prompt("agent_linkedin.txt")
    prompt_seo = load_prompt("agent_seo_video.txt")

    # ÉTAPE 1 : Agent 1 (Pitch & LP)
    input_agent_1 = f"<livre_blanc>\n{whitepaper_text}\n</livre_blanc>\n<metadata>\n{metadata}\n</metadata>"
    result_pitch = await run_subagent("Agent 1 - Pitch & LP", prompt_pitch, input_agent_1)

    # ÉTAPE 2 & 3 EN PARALLÈLE
    input_agent_2 = f"<livre_blanc>\n{whitepaper_text}\n</livre_blanc>\n<pitch_reference>\n{result_pitch}\n</pitch_reference>\n<metadata>\n{metadata}\n</metadata>"
    input_agent_3 = f"<livre_blanc>\n{whitepaper_text}\n</livre_blanc>"

    task_linkedin = run_subagent("Agent 2 - LinkedIn & Internal", prompt_linkedin, input_agent_2)
    task_seo = run_subagent("Agent 3 - SEO & Vidéo", prompt_seo, input_agent_3)

    result_linkedin, result_seo = await asyncio.gather(task_linkedin, task_seo)

    # ASSEMBLAGE DU RAPPORT FINAL
    final_output = f"# KIT COMPLET DE DIFFUSION DU LIVRE BLANC\n\n"
    final_output += f"## PARTIE 1 : PITCH, LANDING PAGE & CORPO\n\n{result_pitch}\n\n---\n\n"
    final_output += f"## PARTIE 2 : KIT LINKEDIN & INTERNAL ADVOCACY\n\n{result_linkedin}\n\n---\n\n"
    final_output += f"## PARTIE 3 : DÉCLINAISONS SEO & SCRIPT VIDÉO\n\n{result_seo}\n"

    with open("Rapport_Diffusion_Final.md", "w", encoding="utf-8") as f:
        f.write(final_output)

    print("\n🎉 Pipeline terminé avec succès ! Le fichier 'Rapport_Diffusion_Final.md' a été généré.")

if __name__ == "__main__":
    print("=== ASSISTANT MULTI-AGENTS DIFFUSION LIVRE BLANC ===")
    
    # 1. Demande le nom du fichier PDF
    pdf_filename = input("📄 Indiquez le nom du fichier PDF (ex: livre_blanc.pdf) : ").strip()
    
    # 2. Demande les métadonnées (auteurs et métiers)
    metadata_input = input("👥 Indiquez les auteurs et leurs métiers (ex: Morgane Fauvel - Lead Consultant, Jean Dupont - Senior Partner) : ").strip()

    # Lecture du fichier PDF
    if not os.path.exists(pdf_filename):
        print(f"❌ Erreur : Le fichier '{pdf_filename}' n'a pas été trouvé dans le dossier du projet.")
    else:
        print(f"\n📖 Lecture du fichier PDF '{pdf_filename}' en cours...")
        reader = PdfReader(pdf_filename)
        whitepaper_content = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                whitepaper_content += text + "\n"

        # Lancement de l'orchestration
        asyncio.run(orchestrate_whitepaper_pipeline(whitepaper_content, metadata_input))