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

# Configuración para evitar errores de codificación
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

# Configurar paths
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# Inicialización de Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Capturar requerimiento del usuario (Paso 1)
REQUERIMIENTO = sys.stdin.read().strip()
if not REQUERIMIENTO:
    REQUERIMIENTO = "El sistema debe gestionar usuarios de forma segura y eficiente"

# --- PASO 2: Búsqueda de historias similares (Retrieval) ---
modelo = SentenceTransformer("all-MiniLM-L6-v2")
kb_path = BASE_DIR / "knowledge_base_data"
db_client = chromadb.PersistentClient(path=str(kb_path))
collection = db_client.get_or_create_collection(
    name="katary_sgc",
    metadata={"hnsw:space": "cosine"}
)

query_emb = modelo.encode([REQUERIMIENTO]).tolist()
resultados = collection.query(query_embeddings=query_emb, n_results=3)

historias_ref_list = []
contexto_kb = "## HISTORIAS DE REFERENCIA DEL SGC DE KATARY\n"
for i in range(len(resultados["ids"][0])):
    sim = 1 - resultados["distances"][0][i]
    texto = resultados["documents"][0][i]
    id_h = resultados["ids"][0][i]
    historias_ref_list.append(f"ID: {id_h} (Similitud: {sim:.2f})")
    contexto_kb += f"### Referencia {i+1}\n**Historia:** {texto}\n\n"

# --- PASO 3: Generación CON RAG ---
system_prompt = f"Eres un Analista Senior de Katary Software.\n{contexto_kb}"
res_rag = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": REQUERIMIENTO}],
    temperature=0.3
)
resultado_con_rag = res_rag.choices[0].message.content

# --- PASO 4: Generación SIN RAG ---
res_sin = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": REQUERIMIENTO}],
    temperature=0.3
)
resultado_sin_rag = res_sin.choices[0].message.content

# --- PASO 5: Comparación y Análisis ---
comparacion = f"""
ANÁLISIS DE RESPUESTAS:
- Sin RAG: {len(resultado_sin_rag)} caracteres. (Conocimiento general)
- Con RAG: {len(resultado_con_rag)} caracteres. (Alineado a Katary Software)
- Historias similares encontradas: {len(historias_ref_list)}
"""

# --- PASO 6: Descubrir la limitación ---
limitacion = """
LIMITACIÓN TÉCNICA DETECTADA:
La salida es TEXTO LIBRE (Natural Language). 
Problemas para automatización:
1. No se puede parsear por un programa de forma confiable.
2. El formato varía en cada ejecución.
3. No es apto para integraciones con otros agentes (Agente de Pruebas, etc).
"""

# Retorno final en JSON
print(json.dumps({
    "paso1": REQUERIMIENTO,
    "paso2": "\n".join(historias_ref_list),
    "paso3": resultado_con_rag,
    "paso4": resultado_sin_rag,
    "paso5": comparacion,
    "paso6": limitacion
}, ensure_ascii=False))