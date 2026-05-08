"""Agente v2: RAG + Salida JSON estructurada + Validación Contract A.

Evolución de v1: misma base RAG pero ahora exigimos al LLM que responda
en JSON y validamos la estructura con Pydantic (Contract A).

¿Por qué esta mejora?
    En v1 la salida era texto libre. Eso significa que:
    - Cada ejecución produce un formato diferente
    - No se puede pasar automáticamente al Agente 2 (Test Architect)
    - No hay forma de verificar que la salida esté completa

    Ahora con Contract A (Pydantic), tenemos un "guardia de seguridad"
    que valida la estructura antes de dejarla pasar al siguiente agente.

Mejoras sobre v1:
    ✅ Salida JSON estructurada — formato consistente siempre
    ✅ Validación Pydantic — Contract A como contrato entre agentes
    ✅ Reintentos automáticos — si el JSON es inválido, le dice al LLM qué corregir
    ✅ Guardado automático — genera archivo JSON en carpeta output/

Limitaciones (se resuelven en v3):
    ❌ No detecta ambigüedades — "seguro", "rápido", "fácil" pasan sin análisis
    ❌ No hay interacción con el analista — el LLM decide todo solo

Pipeline: Requisito → ChromaDB → Prompt → Groq → JSON → Validación Contract A → Archivo

Ejecutar:
    python agente_v2_json.py
"""

import json
import os
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
import chromadb

# Importar Contract A (el contrato entre agentes)
sys.path.insert(0, str(Path(__file__).parent))
from src.contract_a import (
    AcceptanceCriterion,
    AmbiguityResolution,
    RefinedRequirements,
    UserStory,
    Priority,
    StoryType,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Error: No se encontró GROQ_API_KEY en el archivo .env")
    sys.exit(1)


# ============================================================
# SYSTEM PROMPT — Le pedimos JSON al LLM
# ============================================================
SYSTEM_PROMPT = """Eres un Analista de Requerimientos Senior de Katary Software (CMMI-DEV L3, 19 años).
Transforma requerimientos ambiguos en historias de usuario estructuradas.

{kb_context}

## FORMATO JSON OBLIGATORIO
Responde SOLO con JSON válido, sin texto ni markdown. Estructura:
{{"project_context": "resumen", "user_stories": [
  {{"id": "US-001", "title": "mín 10 chars", "story_type": "functional|non_functional|technical",
    "priority": "critical|high|medium|low", "as_a": "rol", "i_want": "acción", "so_that": "beneficio",
    "acceptance_criteria": [
      {{"id": "AC-001", "description": "mín 20 chars", "given": "precondición concreta",
        "when": "acción específica", "then": "resultado verificable con tiempos",
        "test_data_examples": [{{"campo": "val", "expected": "resultado"}}],
        "is_negative_case": false, "boundary_values": ["mín", "máx"]}}],
    "business_rules": [], "dependencies": [], "ui_elements": [], "api_endpoints": [],
    "ambiguities_resolved": [
      {{"original_text": "texto ambiguo", "issue": "por qué", "resolution": "valores concretos", "assumption_made": true}}]
  }}]}}

## REGLAS
1. IDs: US-001, AC-001 (3 dígitos). ACs secuenciales globales
2. Cada criterio: given/when/then con datos concretos, mín 2 test_data_examples
3. Por cada caso positivo, incluir 1 criterio negativo (is_negative_case: true)
4. Detectar y resolver ambigüedades con valores concretos en ambiguities_resolved
5. Responde SOLO JSON"""


# ============================================================
# CLASE DEL AGENTE v2
# ============================================================
class RequirementsRefinerAgent:
    """Agente v2: RAG + JSON + Contract A.

    Flujo:
        1. Recibe requerimiento en texto libre
        2. Busca historias similares en ChromaDB (RAG)
        3. Construye prompt pidiendo JSON
        4. Envía a Groq
        5. Valida contra Contract A (Pydantic)
        6. Si falla, reintenta con feedback de errores
        7. Retorna RefinedRequirements validado
    """

    def __init__(
        self,
        groq_api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        embedding_model: str = "all-MiniLM-L6-v2",
        kb_path: str = None,
        stories_path: str = None,
        max_retries: int = 3,
        temperature: float = 0.3,
    ):
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.temperature = temperature

        # Paths por defecto
        base_dir = Path(__file__).parent
        self.kb_path = kb_path or str(base_dir / "knowledge_base_data")
        self.stories_path = stories_path or str(
            base_dir / "examples" / "knowledge_base" / "katary_stories.json"
        )

        # Inicializar componentes
        self._init_embeddings(embedding_model)
        self._init_chromadb()

    def _init_embeddings(self, model_name: str):
        """Carga el modelo de embeddings."""
        print(f"⏳ Cargando modelo de embeddings: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        print(f"   ✅ Modelo cargado")

    def _init_chromadb(self):
        """Inicializa ChromaDB y carga historias si es necesario."""
        client = chromadb.PersistentClient(path=self.kb_path)
        self.collection = client.get_or_create_collection(
            name="katary_sgc",
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() == 0:
            self._load_stories()
        else:
            print(f"📚 Base de conocimiento: {self.collection.count()} historias")

    def _load_stories(self):
        """Carga historias desde JSON a ChromaDB."""
        print(f"📚 Indexando historias del SGC en ChromaDB...")
        with open(self.stories_path, "r", encoding="utf-8") as f:
            stories = json.load(f)

        textos = [s["texto"] for s in stories]
        embeddings = self.embedder.encode(textos).tolist()

        self.collection.add(
            ids=[s["id"] for s in stories],
            embeddings=embeddings,
            documents=textos,
            metadatas=[{
                "dominio": s.get("dominio", "general"),
                "criterios": s.get("criterios", ""),
            } for s in stories],
        )
        print(f"   ✅ {self.collection.count()} historias indexadas")

    # ----------------------------------------------------------
    # PASO 1: Búsqueda en KB (Retrieval)
    # ----------------------------------------------------------
    def _search_kb(self, requirement: str, top_k: int = 3) -> list[dict]:
        """Busca historias similares en ChromaDB."""
        query_emb = self.embedder.encode([requirement]).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        historias = []
        for i in range(len(results["ids"][0])):
            sim = 1 - results["distances"][0][i]
            historias.append({
                "id": results["ids"][0][i],
                "texto": results["documents"][0][i],
                "criterios": results["metadatas"][0][i].get("criterios", ""),
                "dominio": results["metadatas"][0][i].get("dominio", ""),
                "similitud": sim,
            })

        return historias

    # ----------------------------------------------------------
    # PASO 2: Construir prompt (Augmented)
    # ----------------------------------------------------------
    def _build_prompt(self, requirement: str, historias: list[dict]) -> tuple[str, str]:
        """Construye system prompt con contexto RAG."""
        contexto = "## HISTORIAS DE REFERENCIA DEL SGC DE KATARY\n"
        contexto += "Usa estas historias como modelo de calidad y profundidad:\n\n"

        for i, h in enumerate(historias, 1):
            contexto += f"### Referencia {i} [{h['id']}] (similitud: {h['similitud']:.2f})\n"
            contexto += f"**Historia:** {h['texto']}\n"
            contexto += f"**Criterios:** {h['criterios']}\n"
            contexto += f"**Dominio:** {h['dominio']}\n\n"

        system = SYSTEM_PROMPT.format(kb_context=contexto)

        user = (
            f"Analiza el siguiente requerimiento y transfórmalo en historias "
            f"de usuario con el nivel de calidad de las referencias del SGC de Katary.\n\n"
            f"REQUERIMIENTO:\n{requirement}"
        )

        return system, user

    # ----------------------------------------------------------
    # PASO 3: Llamar al LLM (Generation)
    # ----------------------------------------------------------
    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Envía el prompt a Groq y retorna la respuesta."""
        response = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=4000,
        )
        return response.choices[0].message.content

    def _call_llm_with_retry(self, system_prompt: str, user_message: str, errors: list[str]) -> str:
        """Reintenta la generación enviándole los errores de validación."""
        retry_message = (
            f"{user_message}\n\n"
            f"## CORRECCIONES REQUERIDAS\n"
            f"Tu respuesta anterior tuvo estos errores de validación:\n\n"
        )
        for i, err in enumerate(errors, 1):
            retry_message += f"{i}. {err}\n"

        retry_message += (
            "\nCorrige TODOS los errores y responde con el JSON completo corregido. "
            "SOLO JSON, sin texto adicional."
        )

        return self._call_llm(system_prompt, retry_message)

    # ----------------------------------------------------------
    # PASO 4: Parsear y validar JSON — NUEVO en v2
    # ----------------------------------------------------------
    def _extract_json(self, raw_response: str) -> dict:
        """Extrae JSON de la respuesta del LLM, limpiando texto extra."""
        text = raw_response.strip()

        # Remover bloques de markdown si el LLM los incluye
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.rsplit("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            text = text.rsplit("```", 1)[0]

        # Encontrar el JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No se encontró JSON válido en la respuesta del LLM")

        json_str = text[start:end]
        return json.loads(json_str)

    def _validate_contract_a(self, raw_json: dict, original_text: str, run_id: str) -> RefinedRequirements:
        """Valida el JSON contra Contract A (Pydantic)."""
        user_stories = []
        ac_counter = 0

        for story_data in raw_json.get("user_stories", []):
            # Criterios de aceptación
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

            # Ambigüedades resueltas (el LLM las detecta, pero por suposición)
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

            # Mapear enums
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
                    title=story_data.get("title", "Sin título"),
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

        if not user_stories:
            raise ValueError("El LLM no generó ninguna historia de usuario")

        # Métricas
        total_ambiguities = sum(len(s.ambiguities_resolved) for s in user_stories)
        total_assumptions = sum(
            sum(1 for a in s.ambiguities_resolved if a.assumption_made)
            for s in user_stories
        )

        return RefinedRequirements(
            pipeline_run_id=run_id,
            agent_version="2.0.0",
            original_requirements_text=original_text,
            project_context=raw_json.get("project_context", ""),
            user_stories=user_stories,
            total_ambiguities_found=total_ambiguities,
            total_assumptions_made=total_assumptions,
            coverage_notes=raw_json.get("coverage_notes"),
        )

    # ----------------------------------------------------------
    # PIPELINE COMPLETO
    # ----------------------------------------------------------
    def process(self, requirement: str, top_k: int = 3) -> RefinedRequirements:
        """Ejecuta el pipeline completo."""
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        print(f"\n{'=' * 60}")
        print(f"QUALITYAI — Agente v2: RAG + JSON + Contract A")
        print(f"Run ID: {run_id}")
        print(f"{'=' * 60}")
        print(f"\n📝 Requerimiento: \"{requirement}\"")

        # Paso 1: Buscar en KB
        print(f"\n🔍 Paso 1: Búsqueda en base de conocimiento...")
        historias = self._search_kb(requirement, top_k)
        for h in historias:
            emoji = "🟢" if h["similitud"] > 0.5 else "🟡" if h["similitud"] > 0.3 else "🔴"
            print(f"   {emoji} [{h['id']}] {h['similitud']:.3f} — {h['dominio']}")

        # Paso 2: Construir prompt
        print(f"\n📦 Paso 2: Construyendo prompt RAG...")
        system_prompt, user_message = self._build_prompt(requirement, historias)
        print(f"   Contexto: {len(system_prompt)} chars | Mensaje: {len(user_message)} chars")

        # Paso 3 y 4: Generar + Validar (con reintentos)
        last_errors = []
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt == 1:
                    print(f"\n🤖 Paso 3: Generando JSON con {self.model_name}...")
                    raw_response = self._call_llm(system_prompt, user_message)
                else:
                    print(f"\n🔄 Reintento {attempt}/{self.max_retries}: corrigiendo errores...")
                    raw_response = self._call_llm_with_retry(system_prompt, user_message, last_errors)

                print(f"   📄 Parseando JSON...")
                raw_json = self._extract_json(raw_response)

                stories_count = len(raw_json.get("user_stories", []))
                criteria_count = sum(
                    len(s.get("acceptance_criteria", []))
                    for s in raw_json.get("user_stories", [])
                )
                print(f"   Encontradas: {stories_count} historias, {criteria_count} criterios")

                print(f"   ✅ Paso 4: Validando contra Contract A...")
                result = self._validate_contract_a(raw_json, requirement, run_id)

                print(f"\n{'=' * 60}")
                print(f"✅ VALIDACIÓN EXITOSA (intento {attempt})")
                print(f"{'=' * 60}")
                print(f"   Historias generadas: {len(result.user_stories)}")
                total_criteria = sum(len(s.acceptance_criteria) for s in result.user_stories)
                print(f"   Criterios de aceptación: {total_criteria}")
                print(f"   Ambigüedades detectadas: {result.total_ambiguities_found}")
                print(f"   Suposiciones realizadas: {result.total_assumptions_made}")

                # ⚠️ Advertir sobre suposiciones
                if result.total_assumptions_made > 0:
                    print(f"\n⚠️ ADVERTENCIA: El LLM hizo {result.total_assumptions_made} suposiciones")
                    print(f"   Estas no fueron validadas por un analista humano")
                    print(f"   → Solución: ver agente_v3_ambiguity.py")

                return result

            except json.JSONDecodeError as e:
                last_errors = [f"JSON inválido: {str(e)}"]
                print(f"   ❌ Error de JSON: {e}")

            except ValidationError as e:
                last_errors = [err["msg"] for err in e.errors()]
                print(f"   ❌ Validación Contract A fallida ({len(last_errors)} errores):")
                for err in last_errors[:5]:
                    print(f"      - {err}")

            except ValueError as e:
                last_errors = [str(e)]
                print(f"   ❌ Error: {e}")

        raise RuntimeError(
            f"El LLM no generó un JSON válido después de {self.max_retries} intentos. "
            f"Errores: {last_errors}"
        )

    def process_and_save(self, requirement: str, output_path: str = None, top_k: int = 3) -> Path:
        """Ejecuta el pipeline y guarda el resultado como JSON."""
        result = self.process(requirement, top_k)

        if output_path is None:
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"contract_a_v2_{timestamp}.json"

        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)

        print(f"\n💾 Contract A guardado en: {output_path}")
        return output_path


# ============================================================
# VISUALIZACIÓN
# ============================================================
def mostrar_resultado(result: RefinedRequirements):
    """Muestra el resultado de forma legible."""
    print(f"\n{'=' * 60}")
    print(f"📋 HISTORIAS DE USUARIO GENERADAS")
    print(f"{'=' * 60}")

    for story in result.user_stories:
        print(f"\n{'─' * 60}")
        print(f"📌 [{story.id}] {story.title}")
        print(f"   Tipo: {story.story_type.value} | Prioridad: {story.priority.value}")
        print(f"\n   Como: {story.as_a}")
        print(f"   Quiero: {story.i_want}")
        print(f"   Para que: {story.so_that}")

        print(f"\n   📋 Criterios de aceptación ({len(story.acceptance_criteria)}):")
        for ac in story.acceptance_criteria:
            neg = " [NEGATIVO]" if ac.is_negative_case else ""
            print(f"\n   [{ac.id}]{neg} {ac.description}")
            print(f"      GIVEN: {ac.given}")
            print(f"      WHEN:  {ac.when}")
            print(f"      THEN:  {ac.then}")
            if ac.test_data_examples:
                print(f"      Datos de prueba: {ac.test_data_examples}")
            if ac.boundary_values:
                print(f"      Valores límite: {ac.boundary_values}")

        if story.ambiguities_resolved:
            print(f"\n   ⚠️ Ambigüedades resueltas ({len(story.ambiguities_resolved)}):")
            for amb in story.ambiguities_resolved:
                supuesto = " [SUPOSICIÓN]" if amb.assumption_made else ""
                print(f"      - \"{amb.original_text}\" → {amb.resolution}{supuesto}")

    print(f"\n{'=' * 60}")
    print(f"📊 RESUMEN")
    total_ac = sum(len(s.acceptance_criteria) for s in result.user_stories)
    total_neg = sum(
        sum(1 for ac in s.acceptance_criteria if ac.is_negative_case)
        for s in result.user_stories
    )
    print(f"   Historias: {len(result.user_stories)}")
    print(f"   Criterios: {total_ac} ({total_neg} negativos)")
    print(f"   Ambigüedades: {result.total_ambiguities_found}")
    print(f"   Suposiciones: {result.total_assumptions_made}")


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    agente = RequirementsRefinerAgent(groq_api_key=GROQ_API_KEY)

    requerimiento = input("\n📝 Escriba un requerimiento (o Enter para ejemplo):\n> ")
    if not requerimiento.strip():
        requerimiento = "Necesito un sistema de login seguro para la plataforma"

    try:
        resultado = agente.process(requerimiento)
        mostrar_resultado(resultado)
        archivo = agente.process_and_save(requerimiento)
    except RuntimeError as e:
        print(f"\n❌ {e}")
