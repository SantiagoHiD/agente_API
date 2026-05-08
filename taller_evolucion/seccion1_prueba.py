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

# Configuración para evitar errores de codificación y warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

# Configurar paths
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# ============================================================
# PASO 1: Inicializar componentes e Historia de Usuario
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Recibimos el requerimiento desde Flask (stdin)
REQUERIMIENTO = sys.stdin.read().strip()
if not REQUERIMIENTO:
    REQUERIMIENTO = "El sistema debe gestionar usuarios de forma segura y eficiente"

# ============================================================
# PASO 2: Buscar historias similares (Retrieval)
# ============================================================
modelo = SentenceTransformer("all-MiniLM-L6-v2")
kb_path = BASE_DIR / "knowledge_base_data"
db_client = chromadb.PersistentClient(path=str(kb_path))
collection = db_client.get_or_create_collection(name="katary_sgc")

# Si la base está vacía, la poblamos (Tu bloque del 'if')
if collection.count() == 0:
    stories_path = BASE_DIR / "examples" / "knowledge_base" / "katary_stories.json"
    with open(stories_path, "r", encoding="utf-8") as f:
        stories = json.load(f)
    textos = [s["texto"] for s in stories]
    embeddings = modelo.encode(textos).tolist()
    collection.add(
        ids=[s["id"] for s in stories],
        embeddings=embeddings,
        documents=textos,
        metadatas=[{"dominio": s.get("dominio", "general"), "criterios": s.get("criterios", "")} for s in stories],
    )

# Realizamos la búsqueda
query_emb = modelo.encode([REQUERIMIENTO]).tolist()
resultados = collection.query(query_embeddings=query_emb, n_results=3)

historias_ref_list = [] # Esta es la variable que causaba el error
contexto_kb = "## HISTORIAS DE REFERENCIA DEL SGC DE KATARY\n"

# Tu bloque del 'for' para formatear resultados
for i in range(len(resultados["ids"][0])):
    sim = 1 - resultados["distances"][0][i]
    texto = resultados["documents"][0][i]
    id_h = resultados["ids"][0][i]
    historias_ref_list.append(f"ID: {id_h} (Similitud: {sim:.2f})")
    contexto_kb += f"### Referencia {i+1} [{id_h}]\n**Historia:** {texto}\n\n"

# ============================================================
# PASO 3: Generación CON RAG
# ============================================================
system_prompt = f"Eres un Analista Senior de Katary Software.\n{contexto_kb}"
res_rag = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": REQUERIMIENTO}],
    temperature=0.3
)
resultado_con_rag = res_rag.choices[0].message.content

# ============================================================
# PASO 4: Generación SIN RAG
# ============================================================
res_sin = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": REQUERIMIENTO}],
    temperature=0.3
)
resultado_sin_rag = res_sin.choices[0].message.content

# ============================================================
# PASO 5: Comparación y análisis
# ============================================================
comparacion = f"""
ANÁLISIS TÉCNICO:
- La versión CON RAG utiliza {len(historias_ref_list)} documentos de referencia.
- Longitud Con RAG: {len(resultado_con_rag)} caracteres vs Sin RAG: {len(resultado_sin_rag)}.
- El contexto de Katary Software permite generar criterios Given/When/Then más específicos.
"""

# ============================================================
# PASO 6: Descubrir la limitación
# ============================================================
limitacion = """
CONCLUSIÓN DEL TALLER:
El texto libre es excelente para humanos, pero imposible de procesar para máquinas.
Si quisiéramos automatizar pruebas, no podríamos extraer campos consistentes de aquí.
La solución: Salidas estructuradas (JSON), que veremos en la Sección 2.
"""

# ENVIAR TODO AL FRONTEND
print(json.dumps({
    "paso1": REQUERIMIENTO,
    "paso2": "\n".join(historias_ref_list),
    "paso3": resultado_con_rag,
    "paso4": resultado_sin_rag,
    "paso5": comparacion,
    "paso6": limitacion
}, ensure_ascii=False))