"""
============================================================
TALLER: EVOLUCIÓN DEL AGENTE — Sección 4 de 4
============================================================
VERSIÓN: v4 — Human-in-the-Loop (HITL)
CONCEPTO: El analista valida, el LLM ejecuta

En esta sección van a:
1. Ver las ambigüedades detectadas y RESOLVERLAS ustedes mismos
2. Elegir entre: aceptar sugerencia, dar su propia resolución, o descartar
3. Comparar la salida: CERTEZAS (v4) vs SUPOSICIONES (v2/v3)
4. Entender cómo las decisiones del analista se propagan al pipeline
5. Comparar resultados entre grupos (misma entrada, diferentes decisiones)

Ejecutar desde la carpeta modulo1_requirements_refiner:
    python taller_evolucion/taller_seccion4_hitl.py
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
# MISMO REQUERIMIENTO QUE EN LAS 3 SECCIONES ANTERIORES
# ============================================================
REQUERIMIENTO = "El sistema debe gestionar usuarios de forma segura y eficiente, permitiendo el registro y autenticacion de usuarios"

print("=" * 70)
print("TALLER: EVOLUCION DEL AGENTE")
print("Seccion 4 de 4: Human-in-the-Loop (HITL)")
print("=" * 70)
print(f"\nRequerimiento de prueba (el mismo de las 3 secciones anteriores):")
print(f'  "{REQUERIMIENTO}"')
print(f"\nEn la seccion anterior vimos que el detector encuentra ambiguedades")
print(f"pero el LLM las resuelve por SUPOSICION. Ahora USTEDES van a decidir.")
print(f"\nCada grupo puede resolver las ambiguedades de forma diferente.")
print(f"Al final, compararemos los resultados entre grupos.")
input("\nPresione Enter para iniciar...")


# ============================================================
# PASO 1: Detectar ambigüedades (igual que Sección 3)
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 1: Detectando ambiguedades (detector deterministico)")
print(f"{'─' * 70}")

detector = AmbiguityDetector()
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
        print(f"  [{nivel}] \"{amb.word}\" — {amb.suggestion}")

    print(f"\n  Total: {len(ambiguedades)} ambiguedades")
    print(f"  Alta: {severity_count['alta']} | Media: {severity_count['media']} | Baja: {severity_count['baja']}")
else:
    print(f"  No se detectaron ambiguedades")

input("\nPresione Enter para iniciar la revision con el analista...")


# ============================================================
# PASO 2: HUMAN-IN-THE-LOOP — El analista resuelve
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 2: HUMAN-IN-THE-LOOP — Ustedes son el analista")
print(f"{'─' * 70}")
print(f"\nPara cada ambiguedad, tienen 3 opciones:")
print(f"  [1] Aceptar la sugerencia automatica")
print(f"  [2] Escribir su propia resolucion (basada en el contexto del proyecto)")
print(f"  [3] Descartar — 'no es ambiguo en este contexto'")
print()

resoluciones = []
aclaraciones = []

if ambiguedades:
    for i, amb in enumerate(ambiguedades, 1):
        if amb.severity == "alta":
            nivel = "ALTA"
        elif amb.severity == "media":
            nivel = "MEDIA"
        else:
            nivel = "BAJA"

        print(f"{'─' * 40}")
        print(f"Ambiguedad {i}/{len(ambiguedades)}: \"{amb.word}\" [{nivel}]")
        print(f"  Categoria:  {amb.category.replace('_', ' ')}")
        print(f"  Contexto:   {amb.context}")
        print(f"  Sugerencia: {amb.suggestion}")
        print()

        while True:
            opcion = input(f"  Que desean hacer? [1] Aceptar sugerencia  [2] Mi resolucion  [3] Descartar\n  > ").strip()

            if opcion == "1":
                resoluciones.append({
                    "word": amb.word,
                    "category": amb.category,
                    "analyst_resolution": amb.suggestion,
                    "status": "resolved",
                })
                aclaraciones.append(f"- \"{amb.word}\": {amb.suggestion}")
                print(f"  Sugerencia aceptada\n")
                break

            elif opcion == "2":
                custom = input(f"  Escriban su resolucion para \"{amb.word}\":\n  > ").strip()
                if custom:
                    resoluciones.append({
                        "word": amb.word,
                        "category": amb.category,
                        "analyst_resolution": custom,
                        "status": "resolved",
                    })
                    aclaraciones.append(f"- \"{amb.word}\": {custom}")
                    print(f"  Resolucion registrada\n")
                    break
                else:
                    print(f"  No escribieron nada. Intenten de nuevo.")

            elif opcion == "3":
                resoluciones.append({
                    "word": amb.word,
                    "category": amb.category,
                    "analyst_resolution": "",
                    "status": "dismissed",
                })
                print(f"  Descartada\n")
                break

            else:
                print(f"  Opcion no valida. Usen 1, 2 o 3.")

    # Resumen
    resueltas = sum(1 for r in resoluciones if r["status"] == "resolved")
    descartadas = sum(1 for r in resoluciones if r["status"] == "dismissed")

    print(f"\n{'─' * 40}")
    print(f"RESUMEN DE SU REVISION:")
    print(f"  Resueltas por el analista: {resueltas}")
    print(f"  Descartadas: {descartadas}")
    print(f"{'─' * 40}")

input("\nPresione Enter para ver como se inyectan las CERTEZAS en el prompt...")


# ============================================================
# PASO 3: Construir prompt con CERTEZAS (no suposiciones)
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 3: Construyendo prompt con las CERTEZAS del analista")
print(f"{'─' * 70}")

if resoluciones:
    # Usar build_resolved_prompt_section — certezas, no suposiciones
    seccion_certezas = detector.build_resolved_prompt_section(resoluciones)
    print(f"\nEsta seccion le dice al LLM: 'ESTAS son CERTEZAS, NO supongas'")
    print(f"\n{'─' * 40}")
    print(seccion_certezas)
    print(f"{'─' * 40}")
    print(f"\n  Diferencia clave:")
    print(f"    v3: build_prompt_section()         -> 'el LLM DEBE resolver estas ambiguedades'")
    print(f"    v4: build_resolved_prompt_section() -> 'DEBES usar estas definiciones como HECHOS'")
    print(f"\n  Resultado: en v4, assumption_made sera FALSE (certeza del analista)")
else:
    seccion_certezas = ""
    print(f"\n  Sin resoluciones del analista")

# Enriquecer requerimiento con aclaraciones
if aclaraciones:
    requerimiento_enriquecido = REQUERIMIENTO + "\n\nACLARACIONES DEL ANALISTA:\n"
    requerimiento_enriquecido += "\n".join(aclaraciones)
    print(f"\n  Requerimiento enriquecido con {len(aclaraciones)} aclaraciones")
else:
    requerimiento_enriquecido = REQUERIMIENTO

input("\nPresione Enter para ejecutar el pipeline completo v4...")


# ============================================================
# PASO 4: Pipeline completo v4 (HITL + Ambiguedades + RAG + JSON + Contract A)
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 4: Ejecutando pipeline completo v4")
print(f"{'─' * 70}")
print(f"\nPipeline: Requerimiento -> Detector -> ANALISTA -> ChromaDB -> RAG+Certezas -> LLM -> JSON -> Contract A")

# 4a. Embeddings y ChromaDB
print(f"\n  4a. Inicializando embeddings y base de conocimiento...")
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

# 4b. Buscar historias similares
print(f"  4b. Buscando historias similares...")
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

# 4c. Construir prompt con RAG + certezas del analista
print(f"  4c. Construyendo prompt con RAG + certezas...")
contexto_kb = "## HISTORIAS DE REFERENCIA DEL SGC DE KATARY\n"
contexto_kb += "Usa estas historias como modelo de calidad y profundidad:\n\n"
for i, h in enumerate(historias, 1):
    contexto_kb += f"### Referencia {i} [{h['id']}] (similitud: {h['similitud']:.2f})\n"
    contexto_kb += f"**Historia:** {h['texto']}\n"
    contexto_kb += f"**Criterios:** {h['criterios']}\n\n"

full_context = contexto_kb
if seccion_certezas:
    full_context += "\n" + seccion_certezas

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
      {{"original_text": "texto ambiguo", "issue": "por que", "resolution": "valores concretos", "assumption_made": false}}]
  }}]}}

## REGLAS
1. IDs: US-001, AC-001 (3 digitos). ACs secuenciales globales
2. Cada criterio: given/when/then con datos concretos, min 2 test_data_examples
3. Por cada caso positivo, incluir 1 criterio negativo (is_negative_case: true)
4. Resolver ambiguedades usando las DECISIONES DEL ANALISTA (assumption_made: false)
5. Responde SOLO JSON"""

user_message = f"""Analiza el siguiente requerimiento y transformalo en historias
de usuario con el nivel de calidad de las referencias del SGC de Katary.

REQUERIMIENTO:
{requerimiento_enriquecido}"""

print(f"      System prompt: {len(system_prompt)} chars")
print(f"      User message: {len(user_message)} chars")

# 4d. Generar con LLM
print(f"  4d. Generando con Groq (llama-3.3-70b-versatile)...")
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

# 4e. Parsear y validar
print(f"  4e. Parseando JSON y validando Contract A...")
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
    agent_version="4.0.0",
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
output_file = output_dir / "taller_seccion4_resultado.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(resultado.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)

print(f"      Archivo guardado: {output_file}")


# ============================================================
# PASO 5: Mostrar resultado detallado
# ============================================================
print(f"\n\n{'=' * 70}")
print("RESULTADO v4 — CON HUMAN-IN-THE-LOOP")
print(f"{'=' * 70}")

for story in resultado.user_stories:
    print(f"\n{'─' * 60}")
    print(f"  [{story.id}] {story.title}")
    print(f"  Tipo: {story.story_type.value} | Prioridad: {story.priority.value}")
    print(f"\n  Como: {story.as_a}")
    print(f"  Quiero: {story.i_want}")
    print(f"  Para que: {story.so_that}")

    print(f"\n  Criterios de aceptacion ({len(story.acceptance_criteria)}):")
    for ac in story.acceptance_criteria:
        neg = " [NEGATIVO]" if ac.is_negative_case else ""
        print(f"\n    [{ac.id}]{neg} {ac.description}")
        print(f"      GIVEN: {ac.given}")
        print(f"      WHEN:  {ac.when}")
        print(f"      THEN:  {ac.then}")

    if story.ambiguities_resolved:
        print(f"\n  Ambiguedades resueltas ({len(story.ambiguities_resolved)}):")
        for amb in story.ambiguities_resolved:
            if amb.assumption_made:
                etiqueta = "SUPOSICION"
            else:
                etiqueta = "CERTEZA"
            print(f'    [{etiqueta}] "{amb.original_text}" -> {amb.resolution}')


# ============================================================
# PASO 6: Comparación final — las 4 versiones
# ============================================================
print(f"\n\n{'=' * 70}")
print("COMPARACION FINAL — Las 4 versiones")
print(f"{'=' * 70}")

print(f"\n  v4 (esta ejecucion):")
print(f"    Historias:     {len(resultado.user_stories)}")
total_ac = sum(len(s.acceptance_criteria) for s in resultado.user_stories)
total_neg = sum(
    sum(1 for ac in s.acceptance_criteria if ac.is_negative_case)
    for s in resultado.user_stories
)
print(f"    Criterios:     {total_ac} ({total_neg} negativos)")
print(f"    Ambiguedades:  {resultado.total_ambiguities_found}")
print(f"    Suposiciones:  {resultado.total_assumptions_made}")
if resultado.total_assumptions_made == 0:
    print(f"    CERO SUPOSICIONES — todas resueltas por el analista")

# Cargar versiones anteriores si existen
for version, archivo in [("v2", "taller_seccion2_resultado.json"), ("v3", "taller_seccion3_resultado.json")]:
    vfile = output_dir / archivo
    if vfile.exists():
        with open(vfile, "r", encoding="utf-8") as f:
            vdata = json.load(f)
        print(f"\n  {version} (de la Seccion anterior):")
        print(f"    Historias:     {len(vdata.get('user_stories', []))}")
        v_ac = sum(len(s.get("acceptance_criteria", [])) for s in vdata.get("user_stories", []))
        print(f"    Criterios:     {v_ac}")
        print(f"    Suposiciones:  {vdata.get('total_assumptions_made', 'N/A')}")


# ============================================================
# RESUMEN EVOLUTIVO
# ============================================================
print(f"\n\n{'=' * 70}")
print("RESUMEN: EVOLUCION DEL AGENTE")
print(f"{'=' * 70}")
print(f"""
  Version 1 (Seccion 1): RAG basico
    + Base de conocimiento mejora la calidad
    - Salida en texto libre, imposible de parsear automaticamente

  Version 2 (Seccion 2): JSON + Contract A
    + Salida estructurada, acceso programatico
    + Validacion Pydantic como contrato entre agentes
    + Reintentos automaticos si el JSON es invalido
    - El LLM detecta ambiguedades de forma inconsistente
    - Todas las resoluciones son SUPOSICIONES

  Version 3 (Seccion 3): Detector de ambiguedades
    + Deteccion DETERMINISTICO pre-LLM (IEEE 830 / ISO 25010)
    + Siempre detecta las mismas palabras (consistente)
    + Se inyectan en el prompt para que el LLM las resuelva
    - El LLM sigue resolviendo por SUPOSICION
    - Nadie valida si las resoluciones son correctas

  Version 4 (Seccion 4): Human-in-the-Loop
    + El ANALISTA resuelve las ambiguedades (CERTEZAS)
    + El LLM usa las decisiones como HECHOS, no suposiciones
    + assumption_made: false -> el pipeline es confiable
    + Modo dual: interactivo o automatico

  Conclusion:
    Cada version resolvio una limitacion de la anterior.
    El agente final (v4) produce resultados CONFIABLES porque
    las decisiones criticas las toma un humano con contexto.
""")


# ============================================================
# ACTIVIDAD FINAL: Comparar entre grupos
# ============================================================
print(f"{'=' * 70}")
print("ACTIVIDAD FINAL: Comparar entre grupos")
print(f"{'=' * 70}")
print(f"""
Cada grupo resolvio las ambiguedades de forma diferente.
Comparen sus archivos JSON de salida:

  output/taller_seccion4_resultado.json

Preguntas para la discusion final:

  1. Que decisiones diferentes tomaron los grupos para "segura"?
     Alguna es "mas correcta" que otra? De que depende?

  2. Si cada grupo fuera un equipo de desarrollo diferente trabajando
     para el MISMO cliente, que pasaria si no se alinean las definiciones?

  3. Imaginen que este pipeline se ejecuta 100 veces al dia en produccion:
     v4 (interactivo) requiere un analista. Es escalable?
     Cuando usarian v3 (automatico) y cuando v4 (interactivo)?

  4. El agente final tiene: RAG + JSON + Contract A + Detector + HITL.
     Si tuvieran que quitar UNA de estas capas, cual quitarian?
     Cual es la mas critica y por que?

  5. Este agente es el Agente 1 del pipeline QualityAI. Su salida
     (Contract A) alimenta al Agente 2 (Test Architect). Si el
     Agente 1 produce suposiciones incorrectas, como afecta al
     Agente 2? Y al Agente 3 (Code Generator)? Y al Agente 4 (Executor)?

Felicitaciones por completar el taller!
""")
