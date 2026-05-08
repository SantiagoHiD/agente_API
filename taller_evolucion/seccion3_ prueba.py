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

# Importar el detector de ambigüedades del proyecto
from src.ambiguity_detector import AmbiguityDetector

# Configuración de entorno
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# PASO 1: Requerimiento
REQUERIMIENTO = sys.stdin.read().strip()
if not REQUERIMIENTO:
    REQUERIMIENTO = "El sistema debe ser rápido y gestionar usuarios de forma segura"

# PASO 2: Detección Determinística de Ambigüedades (Pre-LLM)
detector = AmbiguityDetector()
analisis_ambiguedad = detector.analyze(REQUERIMIENTO)

ambiguedades_encontradas = []
for amb in analisis_ambiguedad["ambiguities"]:
    ambiguedades_encontradas.append(f"- [{amb['category']}] '{amb['word']}': {amb['reason']}")

# PASO 3: RAG (Búsqueda en ChromaDB)
kb_path = BASE_DIR / "knowledge_base_data"
db_client = chromadb.PersistentClient(path=str(kb_path))
collection = db_client.get_or_create_collection(name="katary_sgc")

query_emb = modelo.encode([REQUERIMIENTO]).tolist()
resultados = collection.query(query_embeddings=query_emb, n_results=2)
historias_ref = "\n".join([f"Ref: {doc[:100]}..." for doc in resultados["documents"][0]])

# PASO 4: Generación CON Guía de Ambigüedades
guia_text = json.dumps(analisis_ambiguedad["ambiguities"], indent=2)
prompt_v3 = f"""Eres un Analista Senior. 
Se han detectado estas ambigüedades en el requerimiento: {guia_text}
Usa estas historias de referencia: {historias_ref}

Tarea: Genera las historias de usuario resolviendo CADA ambigüedad con datos concretos.
Responde en formato JSON."""

res_v3 = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "system", "content": prompt_v3}, {"role": "user", "content": REQUERIMIENTO}],
    temperature=0.1
)
resultado_v3 = res_v3.choices[0].message.content

# PASO 5: Comparación (Diferencia entre Detector y LLM)
comparacion = f"""
ANÁLISIS V3:
- El detector encontró {len(analisis_ambiguedad['ambiguities'])} términos vagos.
- El LLM ahora sabe EXACTAMENTE qué palabras debe aclarar.
- Diferencia: En V2 el LLM adivinaba, en V3 el LLM recibe una lista de 'problemas' a corregir.
"""

# PASO 6: La Limitación (Human-in-the-loop)
limitacion = """
LIMITACIÓN DE LA SECCIÓN 3:
Aunque el detector marca el error, el LLM sigue inventando la solución.
Ejemplo: Si el detector dice que 'rápido' es ambiguo, el LLM inventa '2 segundos'.
¿Pero es eso lo que el cliente quería? 
Falta que el HUMANO valide la resolución. Eso es la Sección 4.
"""

# Salida JSON para el frontend
print(json.dumps({
    "paso1": REQUERIMIENTO,
    "paso2": "\n".join(ambiguedades_encontradas) if ambiguedades_encontradas else "No se detectaron ambigüedades críticas.",
    "paso3": resultado_v3,
    "paso4": "Generación técnica con guía de ambigüedades completada.",
    "paso5": comparacion,
    "paso6": limitacion
}, ensure_ascii=False))