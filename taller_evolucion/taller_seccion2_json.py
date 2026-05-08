"""
============================================================
TALLER: EVOLUCIÓN DEL AGENTE — Sección 2 de 4
============================================================
VERSIÓN: v2 — JSON + Contract A (salida estructurada)
CONCEPTO: El valor de la estructura y la validación

En esta sección van a:
1. Generar la MISMA salida que en Sección 1, pero ahora en JSON
2. Acceder a campos específicos programáticamente
3. Ver cómo Contract A (Pydantic) valida la estructura
4. Descubrir que el LLM hace SUPOSICIONES sin validar

Ejecutar desde la carpeta modulo1_requirements_refiner:
    python taller_evolucion/taller_seccion2_json.py
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
# MISMO REQUERIMIENTO QUE EN SECCIÓN 1
# ============================================================
REQUERIMIENTO = "El sistema debe gestionar usuarios de forma segura y eficiente, permitiendo el registro y autenticacion de usuarios"

print("=" * 70)
print("TALLER: EVOLUCION DEL AGENTE")
print("Seccion 2 de 4: JSON + Contract A (salida estructurada)")
print("=" * 70)
print(f"\nRequerimiento de prueba (el mismo de Seccion 1):")
print(f'  "{REQUERIMIENTO}"')
print(f"\nEn la seccion anterior vimos que el texto libre no sirve para")
print(f"un pipeline automatizado. Ahora vamos a pedirle JSON al LLM.")
input("\nPresione Enter para iniciar...")


# ============================================================
# PASO 1: Inicializar componentes (igual que Sección 1)
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 1: Inicializando componentes")
print(f"{'─' * 70}")

print("\nCargando modelo de embeddings...")
modelo = SentenceTransformer("all-MiniLM-L6-v2")
print("  Modelo cargado")

kb_path = BASE_DIR / "knowledge_base_data"
client = chromadb.PersistentClient(path=str(kb_path))
collection = client.get_or_create_collection(
    name="katary_sgc",
    metadata={"hnsw:space": "cosine"},
)

if collection.count() == 0:
    print("\nCargando historias del SGC de Katary...")
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
    print(f"  {collection.count()} historias indexadas")
else:
    print(f"\nBase de conocimiento existente: {collection.count()} historias")


# ============================================================
# PASO 2: Buscar historias similares (igual que Sección 1)
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 2: Buscando historias similares en ChromaDB")
print(f"{'─' * 70}")

query_emb = modelo.encode([REQUERIMIENTO]).tolist()
resultados = collection.query(
    query_embeddings=query_emb,
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

historias = []
for i in range(len(resultados["ids"][0])):
    sim = 1 - resultados["distances"][0][i]
    h = {
        "id": resultados["ids"][0][i],
        "texto": resultados["documents"][0][i],
        "criterios": resultados["metadatas"][0][i].get("criterios", ""),
        "dominio": resultados["metadatas"][0][i].get("dominio", ""),
        "similitud": sim,
    }
    historias.append(h)
    emoji = "ALTA" if sim > 0.5 else "MEDIA" if sim > 0.3 else "BAJA"
    print(f"  [{h['id']}] Similitud: {sim:.3f} ({emoji}) — Dominio: {h['dominio']}")

input("\nPresione Enter para ver el NUEVO prompt que pide JSON...")


# ============================================================
# PASO 3: Construir prompt con FORMATO JSON OBLIGATORIO
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 3: Construyendo prompt — AHORA PEDIMOS JSON")
print(f"{'─' * 70}")
print(f"\nDiferencia clave con Seccion 1:")
print(f"  Seccion 1: 'Genera las historias de usuario' → TEXTO LIBRE")
print(f"  Seccion 2: 'Responde SOLO con JSON valido' → ESTRUCTURA FIJA")

contexto_kb = "## HISTORIAS DE REFERENCIA DEL SGC DE KATARY\n"
contexto_kb += "Usa estas historias como modelo de calidad y profundidad:\n\n"
for i, h in enumerate(historias, 1):
    contexto_kb += f"### Referencia {i} [{h['id']}] (similitud: {h['similitud']:.2f})\n"
    contexto_kb += f"**Historia:** {h['texto']}\n"
    contexto_kb += f"**Criterios:** {h['criterios']}\n\n"

# --- ESTE ES EL PROMPT CLAVE DE v2: Le damos la ESTRUCTURA esperada ---
system_prompt = f"""Eres un Analista de Requerimientos Senior de Katary Software (CMMI-DEV L3, 19 anos).
Transforma requerimientos ambiguos en historias de usuario estructuradas.

{contexto_kb}

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

print(f"\nObserven el system prompt. Fijense en:")
print(f"  1. Le damos la ESTRUCTURA JSON completa como ejemplo")
print(f"  2. Usamos llaves dobles {{{{ }}}} para que Python no las interprete")
print(f"  3. Le decimos 'Responde SOLO JSON, sin texto ni markdown'")
print(f"  4. Definimos REGLAS especificas de formato")
print(f"\n  System prompt: {len(system_prompt)} caracteres")
print(f"  User message:  {len(user_message)} caracteres")

input("\nPresione Enter para generar con JSON...")


# ============================================================
# PASO 4: Generar con LLM pidiendo JSON
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 4: Generando historias en formato JSON")
print(f"{'─' * 70}")
print("\nEnviando a Groq (llama-3.3-70b-versatile)...")

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
print(f"  Respuesta recibida ({respuesta.usage.total_tokens} tokens)")

input("\nPresione Enter para parsear el JSON...")


# ============================================================
# PASO 5: Parsear JSON — extraer de la respuesta del LLM
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 5: Parseando JSON de la respuesta del LLM")
print(f"{'─' * 70}")

print(f"\nLa respuesta RAW del LLM puede venir 'sucia' — con texto antes/despues")
print(f"o envuelta en bloques ```json```. Necesitamos limpiarla.\n")

# Limpiar respuesta
text = respuesta_raw.strip()
if "```json" in text:
    text = text.split("```json", 1)[1]
    text = text.rsplit("```", 1)[0]
    print("  Se removio bloque ```json```")
elif "```" in text:
    text = text.split("```", 1)[1]
    text = text.rsplit("```", 1)[0]
    print("  Se removio bloque ```")

start = text.find("{")
end = text.rfind("}") + 1
if start == -1 or end == 0:
    print("  ERROR: No se encontro JSON valido en la respuesta")
    sys.exit(1)

json_str = text[start:end]
datos = json.loads(json_str)

print(f"  JSON parseado exitosamente")
print(f"  Historias encontradas: {len(datos.get('user_stories', []))}")
print(f"  Proyecto: {datos.get('project_context', 'N/A')[:80]}...")

input("\nPresione Enter para acceder a campos PROGRAMATICAMENTE...")


# ============================================================
# PASO 6: DEMOSTRACIÓN — Acceso programático a los datos
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 6: Acceso programatico a los datos")
print(f"{'─' * 70}")
print(f"\nEsto era IMPOSIBLE con el texto libre de la Seccion 1.")
print(f"Ahora podemos acceder a cualquier campo directamente:\n")

for i, story in enumerate(datos.get("user_stories", []), 1):
    print(f"  Historia {i}:")
    print(f"    ID:        {story.get('id', 'N/A')}")
    print(f"    Titulo:    {story.get('title', 'N/A')}")
    print(f"    Como:      {story.get('as_a', 'N/A')}")
    print(f"    Quiero:    {story.get('i_want', 'N/A')}")
    print(f"    Para que:  {story.get('so_that', 'N/A')}")
    print(f"    Prioridad: {story.get('priority', 'N/A')}")

    # Contar criterios
    criterios = story.get("acceptance_criteria", [])
    negativos = sum(1 for c in criterios if c.get("is_negative_case", False))
    print(f"    Criterios:  {len(criterios)} ({negativos} negativos)")

    # Primer criterio
    if criterios:
        ac = criterios[0]
        print(f"    Primer criterio:")
        print(f"      GIVEN: {ac.get('given', 'N/A')}")
        print(f"      WHEN:  {ac.get('when', 'N/A')}")
        print(f"      THEN:  {ac.get('then', 'N/A')}")

    # Ambiguedades
    ambigs = story.get("ambiguities_resolved", [])
    if ambigs:
        print(f"    Ambiguedades resueltas: {len(ambigs)}")
        for a in ambigs:
            supuesto = " [SUPOSICION]" if a.get("assumption_made", False) else ""
            print(f"      - \"{a.get('original_text', '')}\" -> {a.get('resolution', '')}{supuesto}")

    print()

input("Presione Enter para validar con Contract A...")


# ============================================================
# PASO 7: Validar contra Contract A (Pydantic)
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 7: Validando contra Contract A (Pydantic)")
print(f"{'─' * 70}")
print(f"\nContract A es el 'contrato' entre el Agente 1 (Requirements Refiner)")
print(f"y el Agente 2 (Test Architect). Define la ESTRUCTURA EXACTA que")
print(f"el Agente 2 espera recibir.\n")
print(f"Si el JSON no cumple el contrato, Pydantic lo rechaza.\n")

try:
    # Construir objetos Pydantic desde el JSON
    user_stories = []
    ac_counter = 0

    for story_data in datos.get("user_stories", []):
        criteria = []
        for ac_data in story_data.get("acceptance_criteria", []):
            ac_counter += 1
            criteria.append(
                AcceptanceCriterion(
                    id=ac_data.get("id", f"AC-{ac_counter:03d}"),
                    description=ac_data.get("description", ""),
                    given=ac_data.get("given", ""),
                    when=ac_data.get("when", ""),
                    then=ac_data.get("then", ""),
                    test_data_examples=ac_data.get("test_data_examples", []),
                    is_negative_case=ac_data.get("is_negative_case", False),
                    boundary_values=ac_data.get("boundary_values", []),
                )
            )

        ambiguities = []
        for amb_data in story_data.get("ambiguities_resolved", []):
            ambiguities.append(
                AmbiguityResolution(
                    original_text=amb_data.get("original_text", ""),
                    issue=amb_data.get("issue", ""),
                    resolution=amb_data.get("resolution", ""),
                    assumption_made=amb_data.get("assumption_made", False),
                )
            )

        try:
            story_type = StoryType(story_data.get("story_type", "functional"))
        except ValueError:
            story_type = StoryType.FUNCTIONAL

        try:
            priority = Priority(story_data.get("priority", "medium"))
        except ValueError:
            priority = Priority.MEDIUM

        user_stories.append(
            UserStory(
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
            )
        )

    # Calcular metricas
    total_ambiguities = sum(len(s.ambiguities_resolved) for s in user_stories)
    total_assumptions = sum(
        sum(1 for a in s.ambiguities_resolved if a.assumption_made)
        for s in user_stories
    )

    # Crear el modelo RefinedRequirements
    resultado = RefinedRequirements(
        pipeline_run_id=f"taller-{uuid.uuid4().hex[:8]}",
        agent_version="2.0.0",
        original_requirements_text=REQUERIMIENTO,
        project_context=datos.get("project_context", ""),
        user_stories=user_stories,
        total_ambiguities_found=total_ambiguities,
        total_assumptions_made=total_assumptions,
    )

    print("  VALIDACION EXITOSA — Contract A cumplido")
    print(f"\n  Resumen del Contract A validado:")
    print(f"    Historias:     {len(resultado.user_stories)}")
    total_ac = sum(len(s.acceptance_criteria) for s in resultado.user_stories)
    total_neg = sum(
        sum(1 for ac in s.acceptance_criteria if ac.is_negative_case)
        for s in resultado.user_stories
    )
    print(f"    Criterios:     {total_ac} ({total_neg} negativos)")
    print(f"    Ambiguedades:  {resultado.total_ambiguities_found}")
    print(f"    Suposiciones:  {resultado.total_assumptions_made}")

except ValidationError as e:
    print(f"  VALIDACION FALLIDA — {len(e.errors())} errores:")
    for err in e.errors()[:10]:
        print(f"    - {err['msg']}")
    print(f"\n  Cuando esto pasa, en v2 se reintenta: le enviamos los errores")
    print(f"  al LLM para que corrija y vuelva a generar.")

input("\nPresione Enter para guardar el JSON...")


# ============================================================
# PASO 8: Guardar JSON como archivo
# ============================================================
print(f"\n{'─' * 70}")
print("PASO 8: Guardando resultado como archivo JSON")
print(f"{'─' * 70}")

output_dir = BASE_DIR / "output"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "taller_seccion2_resultado.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(resultado.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)

print(f"\n  Archivo guardado: {output_file}")
print(f"  Tamano: {output_file.stat().st_size:,} bytes")


# ============================================================
# PASO 9: Descubrir la limitación — SUPOSICIONES
# ============================================================
print(f"\n\n{'=' * 70}")
print("EJERCICIO: Busquen las SUPOSICIONES")
print(f"{'=' * 70}")
print(f"""
Ahora la salida es JSON estructurado. Pueden acceder a cualquier campo
programaticamente. PERO hay un problema:

Busquen en el JSON el campo "ambiguities_resolved" y fijense en el campo
"assumption_made". Probablemente TODAS dicen: "assumption_made": true

Eso significa que el LLM SUPUSO como resolver cada ambiguedad:
  - "segura" -> probablemente asumio cifrado AES-256, 2FA, bloqueo por intentos
  - "eficiente" -> probablemente asumio tiempos < 2 segundos
  - Pero nadie le pregunto al cliente si eso era correcto!

PROBLEMA CONCRETO:
  Imaginen que "segura" para el cliente solo significaba "HTTPS + contrasena fuerte"
  Pero el LLM genero criterios para AES-256, 2FA, bloqueo por intentos...

  El Agente 2 (Test Architect) va a generar CASOS DE PRUEBA para esos criterios.
  El Agente 3 (Code Generator) va a generar CODIGO para esas pruebas.
  Al final, desarrollaron funcionalidad que NADIE PIDIO.

  Una suposicion incorrecta se propaga por todo el pipeline.
""")

# Mostrar las suposiciones encontradas
print(f"Las suposiciones en SU ejecucion:")
for story in resultado.user_stories:
    for amb in story.ambiguities_resolved:
        supuesto = "SUPOSICION" if amb.assumption_made else "CONFIRMADO"
        print(f'  [{supuesto}] "{amb.original_text}" -> {amb.resolution}')

print(f"\nCONCLUSION: Necesitamos que ALGUIEN detecte las ambiguedades ANTES")
print(f"de enviar al LLM. Pero no puede ser el LLM (es inconsistente).")
print(f"Necesitamos un detector DETERMINISTICO. Eso es la Seccion 3.")


print(f"\n\n{'=' * 70}")
print("PREGUNTAS DE REFLEXION — Seccion 2")
print(f"{'=' * 70}")
print("""
Respondan en grupo:

  1. Comparen la salida de Seccion 1 (texto libre) con la de Seccion 2 (JSON).
     Cual de las dos podria usar otro programa automaticamente? Por que?

  2. Que pasa si el LLM responde JSON invalido? Como lo resuelve v2?
     (Pista: miren el mecanismo de _call_llm_with_retry en agente_v2_json.py)

  3. Busquen "assumption_made" en su archivo JSON de salida.
     Cuantas suposiciones hizo el LLM? Alguna les parece incorrecta?

  4. Si el Agente 2 recibe una suposicion incorrecta y genera pruebas
     para ella, que impacto tiene en el proyecto? Cuanto cuesta corregirlo?

Cuando hayan respondido, ejecuten la Seccion 3:
  python taller_evolucion/taller_seccion3_ambiguity.py
""")
