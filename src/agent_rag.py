"""Agente 1 con RAG: Requirements Refiner potenciado con Base de Conocimiento.

Este agente es la versión avanzada del Requirements Refiner. Antes de llamar
al LLM, busca historias de usuario similares en la base de conocimiento
del SGC de Katary y las usa como contexto (RAG - Retrieval-Augmented Generation).

Diferencia vs agente básico:
- Básico: prompt fijo → LLM → historias genéricas
- RAG: búsqueda en KB → prompt enriquecido → LLM → historias informadas por experiencia Katary

Entrada: RequirementsInput (texto libre de requerimientos)
Salida: RefinedRequirements (Contract A)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from shared.base_agent import BaseAgent, AgentError
from shared.settings import Settings
from modulo1_requirements_refiner.src.contract_a import (
    AcceptanceCriterion,
    AmbiguityResolution,
    RefinedRequirements,
    UserStory,
)
from modulo1_requirements_refiner.src.knowledge_base import KnowledgeBase
from modulo1_requirements_refiner.src.prompts_rag import (
    build_system_prompt_with_rag,
    USER_MESSAGE_TEMPLATE,
)


class RequirementsInput(BaseModel):
    """Entrada del Agente 1: texto libre de requerimientos."""

    requirements_text: str = Field(
        ...,
        description="Documento de requerimientos en texto libre",
        min_length=20,
    )
    project_name: Optional[str] = Field(default=None)
    additional_context: Optional[str] = Field(default=None)


class RequirementsRefinerRAGAgent(BaseAgent[RequirementsInput, RefinedRequirements]):
    """Agente 1 con RAG: Refina requerimientos usando base de conocimiento institucional.

    Flujo:
    1. Recibe requerimiento en texto libre
    2. Busca historias similares en la base de conocimiento (embeddings + ChromaDB)
    3. Construye prompt enriquecido con las historias encontradas como referencia
    4. Llama al LLM (Ollama/Gemma o cloud) con el contexto institucional
    5. Valida la salida contra Contract A (Pydantic)
    6. Retorna historias de usuario con calidad CMMI-DEV L3
    """

    name = "requirements_refiner_rag"
    version = "0.2.0"
    description = (
        "Transforma requerimientos en texto libre → Historias de usuario "
        "con criterios de aceptación verificables, potenciado por base "
        "de conocimiento institucional (RAG)"
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        kb_top_k: int = 3,
    ):
        """
        Args:
            settings: configuración del LLM
            knowledge_base: base de conocimiento pre-inicializada (o None para crear una nueva)
            kb_top_k: cantidad de historias similares a incluir como contexto
        """
        super().__init__(settings)
        self.kb = knowledge_base
        self.kb_top_k = kb_top_k

    def _build_system_prompt(self) -> str:
        """Construye el system prompt base (sin RAG)."""
        return build_system_prompt_with_rag("")

    def _build_user_message(self, input_data: RequirementsInput) -> str:
        text = input_data.requirements_text
        if input_data.additional_context:
            text += f"\n\n## CONTEXTO ADICIONAL:\n{input_data.additional_context}"
        return USER_MESSAGE_TEMPLATE.format(requirements_text=text)

    def process(self, input_data: RequirementsInput) -> RefinedRequirements:
        """Procesa los requerimientos con RAG."""

        # 1. Buscar contexto en la base de conocimiento
        kb_context = ""
        if self.kb:
            kb_context = self.kb.build_context_for_prompt(
                input_data.requirements_text,
                top_k=self.kb_top_k,
            )
            from rich.console import Console
            console = Console()
            console.print(f"  [dim]RAG: {self.kb_top_k} historias similares encontradas en KB[/dim]")

        # 2. Construir prompt enriquecido con contexto de la KB
        system_prompt = build_system_prompt_with_rag(kb_context)
        user_message = self._build_user_message(input_data)

        # 3. Llamar al LLM
        raw_output = self.call_llm_json(system_prompt, user_message)

        # 4. Construir modelo de salida (mismo parsing que agente básico)
        user_stories = []
        ac_counter = 0

        for story_data in raw_output.get("user_stories", []):
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

            user_stories.append(
                UserStory(
                    id=story_data.get("id", f"US-{len(user_stories) + 1:03d}"),
                    title=story_data.get("title", "Sin título"),
                    story_type=story_data.get("story_type", "functional"),
                    priority=story_data.get("priority", "medium"),
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
            raise AgentError(
                self.name,
                "LLM no generó ninguna historia de usuario.",
            )

        total_ambiguities = sum(len(s.ambiguities_resolved) for s in user_stories)
        total_assumptions = sum(
            sum(1 for a in s.ambiguities_resolved if a.assumption_made)
            for s in user_stories
        )

        return RefinedRequirements(
            pipeline_run_id="",
            original_requirements_text=input_data.requirements_text,
            project_context=raw_output.get("project_context", ""),
            user_stories=user_stories,
            total_ambiguities_found=total_ambiguities,
            total_assumptions_made=total_assumptions,
            coverage_notes=raw_output.get("coverage_notes"),
        )
