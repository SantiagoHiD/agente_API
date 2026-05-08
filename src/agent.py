"""Agente 1: Requirements Refiner - Implementación principal.

Recibe texto libre de requerimientos y produce historias de usuario
estructuradas con criterios de aceptación verificables.

Entrada: str (texto libre de requerimientos)
Salida: RefinedRequirements (Contract A)
"""

from __future__ import annotations

from datetime import datetime
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
from modulo1_requirements_refiner.src.prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE


class RequirementsInput(BaseModel):
    """Entrada del Agente 1: texto libre de requerimientos."""

    requirements_text: str = Field(
        ...,
        description="Documento de requerimientos en texto libre",
        min_length=20,
    )
    project_name: Optional[str] = Field(default=None)
    additional_context: Optional[str] = Field(default=None)


class RequirementsRefinerAgent(BaseAgent[RequirementsInput, RefinedRequirements]):
    """Agente 1: Transforma requerimientos ambiguos en historias de usuario estructuradas."""

    name = "requirements_refiner"
    version = "0.1.0"
    description = "Transforma requerimientos en texto libre → Historias de usuario con criterios de aceptación verificables"

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__(settings)

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _build_user_message(self, input_data: RequirementsInput) -> str:
        text = input_data.requirements_text
        if input_data.additional_context:
            text += f"\n\n## CONTEXTO ADICIONAL:\n{input_data.additional_context}"
        return USER_MESSAGE_TEMPLATE.format(requirements_text=text)

    def process(self, input_data: RequirementsInput) -> RefinedRequirements:
        """Procesa los requerimientos y genera historias de usuario refinadas."""

        # 1. Llamar al LLM
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(input_data)
        raw_output = self.call_llm_json(system_prompt, user_message)

        # 2. Construir el modelo de salida
        user_stories = []
        ac_counter = 0

        for story_data in raw_output.get("user_stories", []):
            # Construir criterios de aceptación
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

            # Construir ambigüedades resueltas
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
                "LLM no generó ninguna historia de usuario. Revisa el requerimiento de entrada.",
            )

        # 3. Calcular métricas
        total_ambiguities = sum(
            len(story.ambiguities_resolved) for story in user_stories
        )
        total_assumptions = sum(
            sum(1 for a in story.ambiguities_resolved if a.assumption_made)
            for story in user_stories
        )

        return RefinedRequirements(
            pipeline_run_id="",  # Se asigna en el orquestador
            original_requirements_text=input_data.requirements_text,
            project_context=raw_output.get("project_context", ""),
            user_stories=user_stories,
            total_ambiguities_found=total_ambiguities,
            total_assumptions_made=total_assumptions,
            coverage_notes=raw_output.get("coverage_notes"),
        )
