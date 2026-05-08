"""Demo del Pipeline RAG + Groq para QualityAI.

Ejecutar desde la carpeta modulo1_requirements_refiner:
    python demo_rag_groq.py

Requisitos:
    pip install sentence-transformers chromadb groq python-dotenv
    Archivo .env con GROQ_API_KEY=tu-api-key
"""

import json
import os
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

# Silenciar warnings no críticos
warnings.filterwarnings("ignore", message=".*get_sentence_embedding_dimension.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

# Cargar variables de entorno (.env)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Error: No se encontró GROQ_API_KEY en el archivo .env")
    sys.exit(1)


# ============================================================
# PASO 1: Inicializar modelo de embeddings y ChromaDB
# ============================================================
def inicializar_kb():
    """Carga el modelo de embeddings y la base de conocimiento."""
    print("=" * 60)
    print("QUALITYAI — Pipeline RAG + Groq")
    print("=" * 60)

    # Modelo de embeddings
    print("\n⏳ Cargando modelo de embeddings...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    dims = modelo.get_embedding_dimension() if hasattr(modelo, 'get_embedding_dimension') else modelo.get_sentence_embedding_dimension()
    print(f"   ✅ Modelo cargado ({dims} dimensiones)")

    # ChromaDB persistente
    kb_path = Path(__file__).parent / "knowledge_base_data"
    client = chromadb.PersistentClient(path=str(kb_path))

    # Verificar si ya hay datos o hay que cargar
    collection = client.get_or_create_collection(
        name="katary_sgc",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        print("\n📚 Cargando historias del SGC de Katary en ChromaDB...")
        stories_path = Path(__file__).parent / "examples" / "knowledge_base" / "katary_stories.json"

        with open(stories_path, "r", encoding="utf-8") as f:
            stories = json.load(f)

        textos = [s["texto"] for s in stories]
        embeddings = modelo.encode(textos).tolist()

        collection.add(
            ids=[s["id"] for s in stories],
            embeddings=embeddings,
            documents=textos,
            metadatas=[{
                "dominio": s.get("dominio", "general"),
                "criterios": s.get("criterios", ""),
            } for s in stories],
        )
        print(f"   ✅ {collection.count()} historias indexadas")
    else:
        print(f"\n📚 Base de conocimiento existente: {collection.count()} historias")

    return modelo, collection


# ============================================================
# PASO 2: Buscar historias similares (Retrieval)
# ============================================================
def buscar_similares(modelo, collection, requerimiento, top_k=3):
    """Busca las historias más similares al requerimiento."""
    print(f"\n🔍 Buscando historias similares...")
    query_embedding = modelo.encode([requerimiento]).tolist()

    resultados = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    historias = []
    for i in range(len(resultados["ids"][0])):
        similitud = 1 - resultados["distances"][0][i]
        historia = {
            "id": resultados["ids"][0][i],
            "texto": resultados["documents"][0][i],
            "criterios": resultados["metadatas"][0][i].get("criterios", ""),
            "dominio": resultados["metadatas"][0][i].get("dominio", ""),
            "similitud": similitud,
        }
        historias.append(historia)
        emoji = "🟢" if similitud > 0.5 else "🟡" if similitud > 0.3 else "🔴"
        print(f"   {emoji} [{historia['id']}] Similitud: {similitud:.3f} — {historia['dominio']}")

    return historias


# ============================================================
# PASO 3: Construir prompt enriquecido (Augmented)
# ============================================================
def construir_prompt_rag(requerimiento, historias_similares):
    """Construye el prompt enriquecido con contexto de la KB."""
    contexto_kb = "## HISTORIAS DE REFERENCIA DEL SGC DE KATARY\n"
    contexto_kb += "Usa estas historias como modelo de calidad y profundidad:\n\n"

    for i, h in enumerate(historias_similares, 1):
        contexto_kb += f"### Referencia {i} [{h['id']}] (similitud: {h['similitud']:.2f})\n"
        contexto_kb += f"**Historia:** {h['texto']}\n"
        contexto_kb += f"**Criterios:** {h['criterios']}\n"
        contexto_kb += f"**Dominio:** {h['dominio']}\n\n"

    system_prompt = f"""Eres un Analista de Requerimientos Senior de Katary Software,
empresa colombiana con 19 años de experiencia y certificación CMMI-DEV Nivel 3.

Tu trabajo es transformar requerimientos ambiguos en historias de usuario profesionales
con criterios de aceptación Given/When/Then verificables.

REGLAS:
1. Formato: "Como [rol], quiero [acción], para que [beneficio]"
2. Criterios de aceptación en formato Given/When/Then con validaciones específicas
3. Incluir validaciones de datos (longitudes, formatos, rangos)
4. Incluir tiempos de respuesta esperados
5. Incluir manejo de errores y casos límite
6. Incluir al menos un caso negativo por cada caso positivo
7. Detectar y resolver ambigüedades con valores concretos
8. Seguir el nivel de calidad CMMI-DEV Nivel 3

{contexto_kb}"""

    user_message = f"""Transforma el siguiente requerimiento en historias de usuario
con el mismo nivel de calidad y detalle que las historias de referencia del SGC de Katary.

REQUERIMIENTO:
{requerimiento}

Genera las historias de usuario completas con criterios Given/When/Then detallados."""

    return system_prompt, user_message


# ============================================================
# PASO 4: Generar con Groq (Generation)
# ============================================================
def generar_con_groq(system_prompt, user_message):
    """Envía el prompt enriquecido a Groq y obtiene la respuesta."""
    print("\n🤖 Enviando a Groq (LLM)...")
    client = Groq(api_key=GROQ_API_KEY)

    respuesta = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=3000,
    )

    tokens_usados = respuesta.usage
    print(f"   ✅ Respuesta recibida ({tokens_usados.total_tokens} tokens)")

    return respuesta.choices[0].message.content


# ============================================================
# PASO 5: Generar SIN RAG para comparar
# ============================================================
def generar_sin_rag(requerimiento):
    """Genera una historia SIN contexto RAG para comparar."""
    client = Groq(api_key=GROQ_API_KEY)

    respuesta = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Eres un Analista de Requerimientos. Responde en español."},
            {"role": "user", "content": f"""Transforma este requerimiento en historias de usuario
con criterios de aceptación Given/When/Then.

REQUERIMIENTO: {requerimiento}"""},
        ],
        temperature=0.3,
        max_tokens=3000,
    )

    return respuesta.choices[0].message.content


# ============================================================
# PIPELINE COMPLETO
# ============================================================
def pipeline_rag(requerimiento):
    """Ejecuta el pipeline completo: Requisito → ChromaDB → RAG → Groq → Historia."""

    # Inicializar
    modelo, collection = inicializar_kb()

    # Buscar historias similares
    historias = buscar_similares(modelo, collection, requerimiento)

    # Construir prompt enriquecido
    system_prompt, user_message = construir_prompt_rag(requerimiento, historias)
    print(f"\n📦 Prompt construido: {len(system_prompt) + len(user_message)} caracteres")

    # Generar CON RAG
    resultado_rag = generar_con_groq(system_prompt, user_message)

    # Generar SIN RAG para comparar
    print("\n🔄 Generando versión SIN RAG para comparar...")
    resultado_sin_rag = generar_sin_rag(requerimiento)

    # Mostrar resultados
    print("\n")
    print("=" * 60)
    print("❌ RESULTADO SIN RAG (genérico)")
    print("=" * 60)
    print(resultado_sin_rag)

    print("\n\n")
    print("=" * 60)
    print("✅ RESULTADO CON RAG (informado por SGC de Katary)")
    print("=" * 60)
    print(resultado_rag)

    print("\n\n")
    print("=" * 60)
    print("💡 ANÁLISIS")
    print("=" * 60)
    print(f"Sin RAG: {len(resultado_sin_rag)} caracteres")
    print(f"Con RAG: {len(resultado_rag)} caracteres")
    print(f"Historias de referencia usadas: {len(historias)}")
    print("Compara el nivel de detalle en los criterios Given/When/Then")

    return resultado_rag


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    # Puedes cambiar este requerimiento por cualquier otro
    requerimiento = input("\n📝 Escribe un requerimiento (o Enter para usar el ejemplo):\n> ")

    if not requerimiento.strip():
        requerimiento = "Necesito un sistema de login seguro para la plataforma"

    print(f"\n📝 Requerimiento: \"{requerimiento}\"")
    pipeline_rag(requerimiento)
