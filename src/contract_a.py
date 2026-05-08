"""Contract A: Requirements Refiner → Test Architect.

Define el schema de salida del Agente 1 (Requirements Refiner).
Transforma requerimientos ambiguos en historias de usuario estructuradas
con criterios de aceptación verificables, específicos y sin ambigüedades.

Principios de diseño:
- Cada criterio de aceptación debe ser verificable por una máquina (testeable)
- Los criterios usan el patrón Given/When/Then implícito para facilitar Gherkin
- Se incluyen datos de ejemplo para que el Agente 2 genere escenarios concretos
- Se rastrean ambigüedades detectadas y cómo se resolvieron
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StoryType(str, Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    TECHNICAL = "technical"


class AcceptanceCriterion(BaseModel):
    """Un criterio de aceptación verificable y sin ambigüedad.

    Cada criterio debe poder traducirse directamente a un escenario Gherkin.
    """

    id: str = Field(
        ...,
        description="Identificador único del criterio (ej: AC-001)",
        pattern=r"^AC-\d{3}$",
    )
    description: str = Field(
        ...,
        description="Descripción clara del comportamiento esperado",
        min_length=20,
    )
    given: str = Field(
        ...,
        description="Precondición: estado inicial del sistema",
    )
    when: str = Field(
        ...,
        description="Acción: lo que el usuario/sistema hace",
    )
    then: str = Field(
        ...,
        description="Resultado esperado: lo que debe ocurrir",
    )
    test_data_examples: list[dict] = Field(
        default_factory=list,
        description="Datos de ejemplo para generar escenarios concretos",
    )
    is_negative_case: bool = Field(
        default=False,
        description="True si es un caso de prueba negativo (error esperado)",
    )
    boundary_values: list[str] = Field(
        default_factory=list,
        description="Valores límite a considerar en pruebas",
    )


class AmbiguityResolution(BaseModel):
    """Registro de una ambigüedad detectada en el requerimiento original."""

    original_text: str = Field(..., description="Texto original ambiguo")
    issue: str = Field(..., description="Por qué es ambiguo")
    resolution: str = Field(..., description="Cómo se resolvió la ambigüedad")
    assumption_made: bool = Field(
        default=False,
        description="True si la resolución fue por suposición (requiere validación)",
    )


class UserStory(BaseModel):
    """Historia de usuario completamente refinada y lista para testing."""

    id: str = Field(
        ...,
        description="Identificador único (ej: US-001)",
        pattern=r"^US-\d{3}$",
    )
    title: str = Field(..., description="Título conciso de la historia", min_length=10)
    story_type: StoryType = Field(default=StoryType.FUNCTIONAL)
    priority: Priority = Field(default=Priority.MEDIUM)

    # Formato estándar de historia de usuario
    as_a: str = Field(..., description="Rol del usuario (Como un...)")
    i_want: str = Field(..., description="Acción deseada (Quiero...)")
    so_that: str = Field(..., description="Beneficio esperado (Para que...)")

    # Criterios de aceptación - el corazón del contrato
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        ...,
        description="Lista de criterios de aceptación verificables",
        min_length=1,
    )

    # Contexto adicional para el Agente 2
    business_rules: list[str] = Field(
        default_factory=list,
        description="Reglas de negocio que afectan el comportamiento",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Dependencias con otras historias (IDs)",
    )
    ui_elements: list[str] = Field(
        default_factory=list,
        description="Elementos de UI involucrados (botones, formularios, etc.)",
    )
    api_endpoints: list[str] = Field(
        default_factory=list,
        description="Endpoints de API involucrados si aplica",
    )

    # Trazabilidad
    ambiguities_resolved: list[AmbiguityResolution] = Field(
        default_factory=list,
        description="Ambigüedades encontradas y cómo se resolvieron",
    )


class RefinedRequirements(BaseModel):
    """Contract A: Salida del Agente 1, entrada del Agente 2.

    Contiene todas las historias de usuario refinadas a partir
    del documento de requerimientos original.
    """

    # Metadata del pipeline
    pipeline_run_id: str = Field(..., description="ID único de la ejecución del pipeline")
    agent_name: str = Field(default="requirements_refiner")
    agent_version: str = Field(default="0.1.0")
    created_at: datetime = Field(default_factory=datetime.now)

    # Entrada original (para trazabilidad)
    original_requirements_text: str = Field(
        ...,
        description="Texto original del requerimiento tal como fue recibido",
    )

    # Salida refinada
    project_context: str = Field(
        ...,
        description="Resumen del contexto del proyecto extraído de los requerimientos",
    )
    user_stories: list[UserStory] = Field(
        ...,
        description="Historias de usuario refinadas",
        min_length=1,
    )

    # Métricas de calidad del refinamiento
    total_ambiguities_found: int = Field(default=0)
    total_assumptions_made: int = Field(default=0)
    coverage_notes: Optional[str] = Field(
        default=None,
        description="Notas sobre posibles gaps en la cobertura de requerimientos",
    )
