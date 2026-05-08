"""
============================================================
TALLER: EVOLUCIÓN DEL AGENTE — Sección 3 de 4
============================================================
VERSIÓN: v3 — Detector de Ambigüedades (análisis pre-LLM)
CONCEPTO: Detección determinística vs generativa

En esta sección van a:
1. Ver cómo el detector de ambigüedades analiza el requerimiento ANTES del LLM
2. Entender la diferencia entre detección determinística y generativa
3. Ver las categorías IEEE 830 / ISO 25010 que usa el detector
4. Comparar la salida con v2: ¿el LLM resuelve mejor con esta guía?
5. Descubrir que el LLM sigue haciendo SUPOSICIONES

Ejecutar desde la carpeta modulo1_requirements_refiner:
    python taller_evolucion/taller_seccion3_ambiguity.py
============================================================
"""

import json
import os
import sys
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Configurar paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
import chromadb

from src.ambiguity_detector import AmbiguityDetector
from src.contract_a import (
    AcceptanceCriterion,
    AmbiguityResolution,
    RefinedRequirements,
    UserStory,
    Priority,
    StoryType,
)

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: No se encontro GROQ_API_KEY en el archivo .env")
    sys.exit(1)

# ============================================================
# MISMO REQUERIMIENTO QUE EN SECCIONES 1 Y 2
# ============================================================
REQUERIMIENTO = "El sistema debe gestionar usuarios de forma segura y eficiente, permitiendo el registro y autenticacion de usuarios"

print("=" * 70)
print("TALLER: EVOLUCION DEL AGENTE")
print("Seccion 3 de 4: Detector de Ambiguedades (analisis pre-LLM)")
print("=" * 70)
print(f"\nRequerimiento de prueba (el mismo de Secciones 1 y 2):")
print(f'  "{REQUERIMIENTO}"')
print(f"\nEn la seccion anterior vimos que el LLM hace SUPOSICIONES al resolver")
print(f"ambiguedades. Ahora vamos a detectarlas ANTES con reglas deterministicas.")
input("\nPresione Enter para iniciar...")


# ============================================================
# PASO 1: Inicializar el Detector de Ambigüedades
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 1: Inicializando el Detector de Ambiguedades")
print(f"{'─' * 70}")

detector = AmbiguityDetector()

print(f"\nEl detector usa REGLAS, no IA. Esto significa:")
print(f"  - Siempre detecta las MISMAS palabras (deterministico)")
print(f"  - No depende de temperatura, prompt ni modelo")
print(f"  - Si ejecutan 100 veces, el resultado es IDENTICO")
print(f"\nCategorias basadas en IEEE 830 e ISO 25010:")
print(f"  1. Terminos vagos: 'seguro', 'rapido', 'facil', 'eficiente'...")
print(f"  2. Requisitos incompletos: falta informacion critica")
print(f"  3. Ambiguedad de alcance: 'gestionar', 'manejar', 'controlar'...")
print(f"  4. Terminos subjetivos: 'amigable', 'intuitivo', 'moderno'...")
print(f"  5. Requisitos no medibles: no se puede verificar objetivamente")
print(f"  6. Falta de limites: sin rangos, longitudes o formatos definidos")

input("\nPresione Enter para analizar el requerimiento...")


# ============================================================
# PASO 2: Analizar ambigüedades del requerimiento
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 2: Analizando ambiguedades del requerimiento")
print(f"{'─' * 70}")
print(f'\nAnalizando: "{REQUERIMIENTO}"\n')

ambiguedades = detector.analyze(REQUERIMIENTO)

if ambiguedades:
    severity_count = {"alta": 0, "media": 0, "baja": 0}

    for i, amb in enumerate(ambiguedades, 1):
        severity_count[amb.severity] += 1
        if amb.severity == "alta":
            nivel = "ALTA"
        elif amb.severity == "media":
            nivel = "MEDIA"
        else:
            nivel = "BAJA"

        print(f"  Ambiguedad {i}:")
        print(f"    Palabra:    \"{amb.word}\"")
        print(f"    Categoria:  {amb.category.replace('_', ' ')}")
        print(f"    Severidad:  {nivel}")
        print(f"    Contexto:   {amb.context}")
        print(f"    Sugerencia: {amb.suggestion}")
        print()

    print(f"  RESUMEN: {len(ambiguedades)} ambiguedades encontradas")
    print(f"    Alta: {severity_count['alta']}")
    print(f"    Media: {severity_count['media']}")
    print(f"    Baja: {severity_count['baja']}")
else:
    print(f"  No se detectaron ambiguedades")

input("\nPresione Enter para ver como se inyectan en el prompt...")


# ============================================================
# PASO 3: Construir sección de ambigüedades para el prompt
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 3: Construyendo seccion de ambiguedades para el prompt")
print(f"{'─' * 70}")

if ambiguedades:
    seccion_ambiguedades = detector.build_prompt_section(ambiguedades)
    print(f"\nEsta seccion se INYECTA en el prompt del LLM para que")
    print(f"resuelva cada ambiguedad explicitamente:\n")
    print(f"{'─' * 40}")
    print(seccion_ambiguedades)
    print(f"{'─' * 40}")
    print(f"\n  Caracteres adicionales en el prompt: {len(seccion_ambiguedades)}")
else:
    seccion_ambiguedades = ""
    print(f"\n  Sin ambiguedades, el prompt es igual que en v2")

input("\nPresione Enter para probar con OTROS requerimientos...")


# ============================================================
# PASO 4: Ejercicio — Probar con otros requerimientos
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 4: EJERCICIO — Probar el detector con otros requerimientos")
print(f"{'─' * 70}")

ejemplos = [
    "El sistema debe ser rapido y facil de usar",
    "Necesito que el reporte se genere automaticamente con buena calidad",
    "El usuario debe poder realizar consultas de forma intuitiva y moderna",
    "El sistema debe soportar multiples usuarios simultaneamente de manera segura y confiable",
]

for j, ejemplo in enumerate(ejemplos, 1):
    print(f'\n  Ejemplo {j}: "{ejemplo}"')
    ambs = detector.analyze(ejemplo)
    if ambs:
        for a in ambs:
            nivel = "ALTA" if a.severity == "alta" else "MEDIA" if a.severity == "media" else "BAJA"
            print(f'    [{nivel}] "{a.word}" — {a.suggestion}')
    else:
        print(f"    Sin ambiguedades detectadas")

print(f"\n  Observen:")
print(f"    - El detector encuentra las MISMAS palabras siempre")
print(f"    - Si ejecutan otra vez, el resultado es identico (deterministico)")
print(f"    - Esto NO pasa con el LLM (generativo, no deterministico)")

input("\nPresione Enter para ejecutar el pipeline completo v3...")


# ============================================================
# PASO 5: Pipeline completo v3 (Ambiguedades + RAG + JSON + Contract A)
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 5: Ejecutando pipeline completo v3")
print(f"{'─' * 70}")
print(f"\nPipeline: Requerimiento -> Detector -> ChromaDB -> Prompt+Ambiguedades -> LLM -> JSON -> Contract A")

# 5a. Embeddings y ChromaDB
print(f"\n  5a. Inicializando embeddings y base de conocimiento...")
modelo = SentenceTransformer("all-MiniLM-L6-v2")

kb_path = BASE_DIR / "knowledge_base_data"
client = chromadb.PersistentClient(path=str(kb_path))
collection = client.get_or_create_collection(
    name="katary_sgc",
    metadata={"hnsw:space": "cosine"},
)

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

print(f"      Base de conocimiento: {collection.count()} historias")

# 5b. Buscar historias similares
print(f"  5b. Buscando historias similares...")
query_emb = modelo.encode([REQUERIMIENTO]).tolist()
resultados = collection.query(
    query_embeddings=query_emb,
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

historias = []
for i in range(len(resultados["ids"][0])):
    sim = 1 - resultados["distances"][0][i]
    historias.append({
        "id": resultados["ids"][0][i],
        "texto": resultados["documents"][0][i],
        "criterios": resultados["metadatas"][0][i].get("criterios", ""),
        "dominio": resultados["metadatas"][0][i].get("dominio", ""),
        "similitud": sim,
    })
    emoji = "ALTA" if sim > 0.5 else "MEDIA" if sim > 0.3 else "BAJA"
    print(f"      [{historias[-1]['id']}] Similitud: {sim:.3f} ({emoji})")

# 5c. Construir prompt con RAG + ambiguedades
print(f"  5c. Construyendo prompt con RAG + ambiguedades...")
contexto_kb = "## HISTORIAS DE REFERENCIA DEL SGC DE KATARY\n"
contexto_kb += "Usa estas historias como modelo de calidad y profundidad:\n\n"
for i, h in enumerate(historias, 1):
    contexto_kb += f"### Referencia {i} [{h['id']}] (similitud: {h['similitud']:.2f})\n"
    contexto_kb += f"**Historia:** {h['texto']}\n"
    contexto_kb += f"**Criterios:** {h['criterios']}\n\n"

# Combinar contexto RAG + ambiguedades detectadas
full_context = contexto_kb
if seccion_ambiguedades:
    full_context += "\n" + seccion_ambiguedades

system_prompt = f"""Eres un Analista de Requerimientos Senior de Katary Software (CMMI-DEV L3, 19 anos).
Transforma requerimientos ambiguos en historias de usuario estructuradas (IEEE 830 / ISO 25010).

{full_context}

## FORMATO JSON OBLIGATORIO
Responde SOLO con JSON valido, sin texto ni markdown. Estructura:
{{"project_context": "resumen", "user_stories": [
  {{"id": "US-001", "title": "min 10 chars", "story_type": "functional|non_functional|technical",
    "priority": "critical|high|medium|low", "as_a": "rol", "i_want": "accion", "so_that": "beneficio",
    "acceptance_criteria": [
      {{"id": "AC-001", "description": "min 20 chars", "given": "precondicion concreta",
        "when": "accion especifica", "then": "resultado verificable con tiempos",
        "test_data_examples": [{{"campo": "val", "expected": "resultado"}}],
        "is_negative_case": false, "boundary_values": ["min", "max"]}}],
    "business_rules": [], "dependencies": [], "ui_elements": [], "api_endpoints": [],
    "ambiguities_resolved": [
      {{"original_text": "texto ambiguo", "issue": "por que", "resolution": "valores concretos", "assumption_made": true}}]
  }}]}}

## REGLAS
1. IDs: US-001, AC-001 (3 digitos). ACs secuenciales globales
2. Cada criterio: given/when/then con datos concretos, min 2 test_data_examples
3. Por cada caso positivo, incluir 1 criterio negativo (is_negative_case: true)
4. Detectar y resolver ambiguedades con valores concretos en ambiguities_resolved
5. Responde SOLO JSON"""

user_message = f"""Analiza el siguiente requerimiento y transformalo en historias
de usuario con el nivel de calidad de las referencias del SGC de Katary.

REQUERIMIENTO:
{REQUERIMIENTO}"""

print(f"      System prompt: {len(system_prompt)} chars (RAG + ambiguedades)")

# 5d. Generar con LLM
print(f"  5d. Generando con Groq (llama-3.3-70b-versatile)...")
groq_client = Groq(api_key=GROQ_API_KEY)
respuesta = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    temperature=0.3,
    max_tokens=4000,
)

respuesta_raw = respuesta.choices[0].message.content
print(f"      Respuesta recibida ({respuesta.usage.total_tokens} tokens)")

# 5e. Parsear y validar
print(f"  5e. Parseando JSON y validando Contract A...")
text = respuesta_raw.strip()
if "```json" in text:
    text = text.split("```json", 1)[1]
    text = text.rsplit("```", 1)[0]
elif "```" in text:
    text = text.split("```", 1)[1]
    text = text.rsplit("```", 1)[0]

start = text.find("{")
end = text.rfind("}") + 1
datos = json.loads(text[start:end])

# Construir y validar Contract A
user_stories = []
ac_counter = 0
for story_data in datos.get("user_stories", []):
    criteria = []
    for ac_data in story_data.get("acceptance_criteria", []):
        ac_counter += 1
        criteria.append(AcceptanceCriterion(
            id=ac_data.get("id", f"AC-{ac_counter:03d}"),
            description=ac_data.get("description", ""),
            given=ac_data.get("given", ""),
            when=ac_data.get("when", ""),
            then=ac_data.get("then", ""),
            test_data_examples=ac_data.get("test_data_examples", []),
            is_negative_case=ac_data.get("is_negative_case", False),
            boundary_values=ac_data.get("boundary_values", []),
        ))

    ambiguities = []
    for amb_data in story_data.get("ambiguities_resolved", []):
        ambiguities.append(AmbiguityResolution(
            original_text=amb_data.get("original_text", ""),
            issue=amb_data.get("issue", ""),
            resolution=amb_data.get("resolution", ""),
            assumption_made=amb_data.get("assumption_made", False),
        ))

    try:
        story_type = StoryType(story_data.get("story_type", "functional"))
    except ValueError:
        story_type = StoryType.FUNCTIONAL
    try:
        priority = Priority(story_data.get("priority", "medium"))
    except ValueError:
        priority = Priority.MEDIUM

    user_stories.append(UserStory(
        id=story_data.get("id", f"US-{len(user_stories) + 1:03d}"),
        title=story_data.get("title", "Sin titulo"),
        story_type=story_type,
        priority=priority,
        as_a=story_data.get("as_a", ""),
        i_want=story_data.get("i_want", ""),
        so_that=story_data.get("so_that", ""),
        acceptance_criteria=criteria,
        business_rules=story_data.get("business_rules", []),
        dependencies=story_data.get("dependencies", []),
        ui_elements=story_data.get("ui_elements", []),
        api_endpoints=story_data.get("api_endpoints", []),
        ambiguities_resolved=ambiguities,
    ))

total_ambiguities = sum(len(s.ambiguities_resolved) for s in user_stories)
total_assumptions = sum(
    sum(1 for a in s.ambiguities_resolved if a.assumption_made)
    for s in user_stories
)

resultado = RefinedRequirements(
    pipeline_run_id=f"taller-{uuid.uuid4().hex[:8]}",
    agent_version="3.0.0",
    original_requirements_text=REQUERIMIENTO,
    project_context=datos.get("project_context", ""),
    user_stories=user_stories,
    total_ambiguities_found=total_ambiguities,
    total_assumptions_made=total_assumptions,
)

print(f"      Contract A validado exitosamente")

# Guardar
output_dir = BASE_DIR / "output"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "taller_seccion3_resultado.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(resultado.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)

print(f"      Archivo guardado: {output_file}")


# ============================================================
# PASO 6: Comparación v2 vs v3
# ============================================================
print(f"\n\n{'=' * 70}")
print("COMPARACION: v2 (sin detector) vs v3 (con detector)")
print(f"{'=' * 70}")

print(f"\n  Resultado v3:")
print(f"    Historias:     {len(resultado.user_stories)}")
total_ac = sum(len(s.acceptance_criteria) for s in resultado.user_stories)
total_neg = sum(
    sum(1 for ac in s.acceptance_criteria if ac.is_negative_case)
    for s in resultado.user_stories
)
print(f"    Criterios:     {total_ac} ({total_neg} negativos)")
print(f"    Ambiguedades:  {resultado.total_ambiguities_found}")
print(f"    Suposiciones:  {resultado.total_assumptions_made}")

# Cargar v2 si existe para comparar
v2_file = output_dir / "taller_seccion2_resultado.json"
if v2_file.exists():
    with open(v2_file, "r", encoding="utf-8") as f:
        v2_data = json.load(f)
    print(f"\n  Resultado v2 (de la Seccion anterior):")
    print(f"    Historias:     {len(v2_data.get('user_stories', []))}")
    v2_ac = sum(len(s.get("acceptance_criteria", [])) for s in v2_data.get("user_stories", []))
    print(f"    Criterios:     {v2_ac}")
    print(f"    Ambiguedades:  {v2_data.get('total_ambiguities_found', 'N/A')}")
    print(f"    Suposiciones:  {v2_data.get('total_assumptions_made', 'N/A')}")
    print(f"\n  Diferencia clave:")
    print(f"    v2: el LLM detecta Y resuelve (inconsistente, depende del prompt)")
    print(f"    v3: el detector detecta (consistente), el LLM resuelve (aun supone)")
else:
    print(f"\n  (Ejecuten la Seccion 2 primero para ver la comparacion)")


# ============================================================
# PASO 7: Descubrir la limitación
# ============================================================
print(f"\n\n{'=' * 70}")
print("EJERCICIO: El problema de las SUPOSICIONES persiste")
print(f"{'=' * 70}")
print(f"""
v3 mejoro la DETECCION de ambiguedades — ahora es deterministico y confiable.
Pero el PROBLEMA FUNDAMENTAL sigue: el LLM resuelve cada ambiguedad por SUPOSICION.

Miren las ambiguedades resueltas en su JSON:""")

for story in resultado.user_stories:
    for amb in story.ambiguities_resolved:
        supuesto = "SUPOSICION" if amb.assumption_made else "CONFIRMADO"
        print(f'  [{supuesto}] "{amb.original_text}" -> {amb.resolution}')

print(f"""
Preguntense:
  - La palabra "segura" fue resuelta correctamente?
  - El LLM asumio el nivel de seguridad que el CLIENTE realmente necesita?
  - Quien deberia resolver estas ambiguedades? El LLM o el analista humano?

ANALOGIA:
  Imaginen un arquitecto que lee los planos de una casa y encuentra
  ambiguedades: "cocina amplia", "buena iluminacion". En v2, el
  arquitecto inventa las medidas. En v3, un inspector senala las
  ambiguedades, pero el arquitecto sigue inventando. En v4, el
  inspector senala las ambiguedades y se las pregunta al CLIENTE.

CONCLUSION: Necesitamos que las ambiguedades detectadas se presenten
al ANALISTA para que las resuelva con su conocimiento del contexto.
Eso es Human-in-the-Loop. Eso es la Seccion 4.
""")


print(f"\n{'=' * 70}")
print("PREGUNTAS DE REFLEXION — Seccion 3")
print(f"{'=' * 70}")
print("""
Respondan en grupo:

  1. Que significa que el detector sea "deterministico"?
     Si lo ejecutan 5 veces, que cambia y que no cambia?

  2. Comparen las ambiguedades detectadas por el detector (PASO 2) con
     las que el LLM reporto en "ambiguities_resolved" del JSON.
     Son las mismas? El LLM encontro alguna que el detector no?

  3. Si el detector usa reglas fijas (~40 palabras), que pasa con
     ambiguedades que NO estan en su diccionario? Es una limitacion?
     Como se podria mejorar?

  4. El LLM sigue marcando "assumption_made": true. Que habria que
     cambiar en el pipeline para que sea false? Quien deberia decidir?

Cuando hayan respondido, ejecuten la Seccion 4:
  python taller_evolucion/taller_seccion4_hitl.py
""")
