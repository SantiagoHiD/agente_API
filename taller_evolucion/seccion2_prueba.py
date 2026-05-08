import json
import os
import sys
import io
import uuid
import warnings
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

# Configuración de entorno
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# Inicialización
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# PASO 1: Recibir Historia de Usuario
REQUERIMIENTO = sys.stdin.read().strip()
if not REQUERIMIENTO:
    REQUERIMIENTO = "El sistema debe gestionar usuarios de forma segura y eficiente"

# PASO 2: Buscar historias similares (Retrieval)
kb_path = BASE_DIR / "knowledge_base_data"
db_client = chromadb.PersistentClient(path=str(kb_path))
collection = db_client.get_or_create_collection(name="katary_sgc")

query_emb = modelo.encode([REQUERIMIENTO]).tolist()
resultados = collection.query(query_embeddings=query_emb, n_results=3)

historias_ref_list = []
contexto_kb = "## HISTORIAS DE REFERENCIA\n"
for i in range(len(resultados["ids"][0])):
    sim = 1 - resultados["distances"][0][i]
    texto = resultados["documents"][0][i]
    historias_ref_list.append(f"ID: {resultados['ids'][0][i]} ({sim:.2f})")
    contexto_kb += f"- {texto}\n"

# PASO 3: Generación CON RAG (Formato JSON)
# Definimos el esquema que queremos que el LLM siga
prompt_json = f"""Eres un Analista Senior. Responde UNICAMENTE en formato JSON.
{contexto_kb}
Estructura:
{{ "user_stories": [ {{ "id": "US-001", "priority": "high", "story": "Como... quiero... para..." }} ] }}
Requerimiento: {REQUERIMIENTO}"""

res_rag = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "system", "content": prompt_json}],
    temperature=0.1 # Menor temperatura para mayor estabilidad en el JSON
)
resultado_con_rag = res_rag.choices[0].message.content

# PASO 4: Generación SIN RAG (Texto Libre)
res_sin = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": f"Escribe una historia de usuario para: {REQUERIMIENTO}"}],
    temperature=0.3
)
resultado_sin_rag = res_sin.choices[0].message.content

# PASO 5: Comparación y Análisis (Programático)
# Aquí es donde ocurre la magia del JSON: podemos contar cosas
try:
    datos_json = json.loads(resultado_con_rag.strip().replace('```json', '').replace('```', ''))
    num_historias = len(datos_json.get("user_stories", []))
    analisis = f"Análisis: Se detectaron {num_historias} historias en el JSON del RAG."
except:
    analisis = "Error: El RAG no devolvió un JSON válido para comparar."

# PASO 6: Descubrir la limitación (Suposiciones)
limitacion = """
LIMITACIÓN DETECTADA (Suposiciones):
Aunque el JSON es perfecto para las máquinas, el LLM 'supone' detalles.
Si el requerimiento dice 'seguro', el LLM asume 'AES-256' sin preguntar al cliente.
Esto genera software que el cliente quizás no pidió.
"""

# Salida final para la API
print(json.dumps({
    "paso1": REQUERIMIENTO,
    "paso2": "\n".join(historias_ref_list),
    "paso3": resultado_con_rag,
    "paso4": resultado_sin_rag,
    "paso5": analisis,
    "paso6": limitacion
}, ensure_ascii=False))