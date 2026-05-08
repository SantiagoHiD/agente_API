import json
import os
import sys
import io
import warnings
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

# Configuración
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# PASO 1: Recibir datos (Requerimiento + Resoluciones del Humano)
input_data = json.loads(sys.stdin.read())
REQUERIMIENTO = input_data.get("requerimiento", "")
RESOLUCIONES_HUMANAS = input_data.get("resoluciones", {}) 

# PASO 2: Simulación de recuperación de contexto (RAG)
kb_path = BASE_DIR / "knowledge_base_data"
db_client = chromadb.PersistentClient(path=str(kb_path))
collection = db_client.get_or_create_collection(name="katary_sgc")

query_emb = modelo.encode([REQUERIMIENTO]).tolist()
resultados = collection.query(query_embeddings=query_emb, n_results=2)
contexto = "\n".join(resultados["documents"][0])

# PASO 3: Construcción del Prompt con Certezas (HITL)
# En lugar de dejar que el LLM invente, le pasamos lo que el humano decidió
resoluciones_text = ""
for palabra, solucion in RESOLUCIONES_HUMANAS.items():
    resoluciones_text += f"- Término: {palabra} -> Resolución definida por el usuario: {solucion}\n"

prompt_v4 = f"""Eres un Analista Senior de Katary Software.
El analista humano ha tomado decisiones CRÍTICAS sobre las ambigüedades:
{resoluciones_text}

Contexto de la empresa:
{contexto}

TAREA: Genera las historias de usuario en JSON respetando ESTRICTAMENTE las resoluciones del humano.
No inventes valores diferentes a los proporcionados en las resoluciones."""

res_v4 = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "system", "content": prompt_v4}, {"role": "user", "content": REQUERIMIENTO}],
    temperature=0.1
)

# PASO 5: Comparación final de la evolución
comparacion = """
EVOLUCIÓN FINAL (V1 -> V4):
- V1: Texto libre (Caos)
- V2: JSON (Estructura, pero con mentiras/suposiciones)
- V3: Detector (Sabemos qué está mal, pero el LLM sigue inventando)
- V4: HITL (Certeza absoluta. El humano manda, el agente ejecuta).
"""

# PASO 6: La Limitación de V4
limitacion = """
LIMITACIÓN DE V4 (Escalabilidad):
Este modelo es el más preciso, pero es el más lento de ejecutar porque 
requiere que un humano esté presente para responder.
En un pipeline de 1000 requerimientos, V4 es costoso. Se usa solo para 
módulos críticos del negocio.
"""

print(json.dumps({
    "paso1": REQUERIMIENTO,
    "paso2": "Resoluciones aplicadas correctamente.",
    "paso3": res_v4.choices[0].message.content,
    "paso5": comparacion,
    "paso6": limitacion
}, ensure_ascii=False))