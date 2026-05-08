"""Detector de Ambigüedades para Requerimientos de Software.

Escanea el texto de un requerimiento buscando palabras y patrones ambiguos
según los estándares IEEE 830 e ISO 25010. El resultado se inyecta en el
prompt del LLM para que resuelva cada ambigüedad explícitamente.

Categorías de ambigüedad:
    1. Adjetivos vagos: "rápido", "seguro", "fácil", "eficiente"
    2. Verbos imprecisos: "gestionar", "administrar", "manejar" (sin definir acciones)
    3. Cuantificadores indefinidos: "varios", "algunos", "mucho"
    4. Requisitos no funcionales ocultos: palabras que implican rendimiento,
       seguridad, usabilidad, etc. sin métricas concretas
    5. Roles indefinidos: "usuario" sin especificar qué tipo

Referencia: IEEE 830 Sección 4.3 — Características de un buen requerimiento
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Ambiguity:
    """Una ambigüedad detectada en el requerimiento."""
    word: str                    # Palabra o frase ambigua encontrada
    category: str                # Categoría (adjetivo_vago, verbo_impreciso, etc.)
    ieee_830_violation: str      # Qué característica de IEEE 830 viola
    iso_25010_category: str      # Categoría ISO 25010 relacionada (si aplica)
    suggestion: str              # Sugerencia de cómo resolver la ambigüedad
    context: str                 # Fragmento del texto donde se encontró
    severity: str                # alta, media, baja


# ============================================================
# DICCIONARIO DE PALABRAS AMBIGUAS
# ============================================================
# Cada entrada tiene:
#   - category: tipo de ambigüedad
#   - ieee: qué propiedad de IEEE 830 viola
#   - iso: categoría ISO 25010 relacionada (vacío si no aplica)
#   - severity: gravedad
#   - suggestion: cómo resolverla

AMBIGUOUS_WORDS = {
    # --- Adjetivos vagos ---
    "rápido": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "alta",
        "suggestion": "Definir tiempo máximo en segundos (ej: 'responder en menos de 2 segundos')",
    },
    "rápida": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "alta",
        "suggestion": "Definir tiempo máximo en segundos (ej: 'responder en menos de 2 segundos')",
    },
    "rápidamente": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "alta",
        "suggestion": "Definir tiempo máximo en segundos",
    },
    "lento": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "media",
        "suggestion": "Definir umbral de tiempo que se considera inaceptable",
    },
    "seguro": {
        "category": "adjetivo_vago",
        "ieee": "no_ambiguo",
        "iso": "seguridad",
        "severity": "alta",
        "suggestion": "Especificar qué mecanismos de seguridad: cifrado, autenticación 2FA, bloqueo por intentos, etc.",
    },
    "segura": {
        "category": "adjetivo_vago",
        "ieee": "no_ambiguo",
        "iso": "seguridad",
        "severity": "alta",
        "suggestion": "Especificar qué mecanismos de seguridad concretos se requieren",
    },
    "fácil": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "usabilidad",
        "severity": "alta",
        "suggestion": "Definir métricas de usabilidad: máximo N clicks, completar tarea en menos de N minutos",
    },
    "intuitivo": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "usabilidad",
        "severity": "alta",
        "suggestion": "Definir criterios medibles: sin manual, primer uso exitoso en menos de N minutos",
    },
    "intuitiva": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "usabilidad",
        "severity": "alta",
        "suggestion": "Definir criterios medibles de facilidad de uso",
    },
    "eficiente": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "alta",
        "suggestion": "Definir métricas: tiempo de respuesta, uso de recursos, throughput",
    },
    "robusto": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "confiabilidad",
        "severity": "alta",
        "suggestion": "Definir: disponibilidad (99.9%), tolerancia a fallos, recuperación automática",
    },
    "robusta": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "confiabilidad",
        "severity": "alta",
        "suggestion": "Definir métricas de disponibilidad y tolerancia a fallos",
    },
    "adecuado": {
        "category": "adjetivo_vago",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar criterios concretos de qué es 'adecuado' en este contexto",
    },
    "adecuada": {
        "category": "adjetivo_vago",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar criterios concretos de qué es 'adecuada' en este contexto",
    },
    "amigable": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "usabilidad",
        "severity": "media",
        "suggestion": "Definir métricas de usabilidad concretas",
    },
    "óptimo": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "media",
        "suggestion": "Definir qué métricas deben optimizarse y a qué valores",
    },
    "óptima": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "media",
        "suggestion": "Definir métricas concretas de optimización",
    },
    "moderno": {
        "category": "adjetivo_vago",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "baja",
        "suggestion": "Especificar tecnologías, patrones o estándares concretos",
    },
    "moderna": {
        "category": "adjetivo_vago",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "baja",
        "suggestion": "Especificar tecnologías o estándares concretos",
    },
    "escalable": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "alta",
        "suggestion": "Definir: cuántos usuarios concurrentes, volumen de datos, crecimiento esperado",
    },
    "flexible": {
        "category": "adjetivo_vago",
        "ieee": "no_ambiguo",
        "iso": "mantenibilidad",
        "severity": "media",
        "suggestion": "Definir qué debe poder cambiarse sin modificar código",
    },
    "confiable": {
        "category": "adjetivo_vago",
        "ieee": "verificable",
        "iso": "confiabilidad",
        "severity": "alta",
        "suggestion": "Definir: porcentaje de disponibilidad, MTBF, tiempo máximo de recuperación",
    },

    # --- Verbos imprecisos ---
    "gestionar": {
        "category": "verbo_impreciso",
        "ieee": "completo",
        "iso": "",
        "severity": "alta",
        "suggestion": "Descomponer en acciones concretas: crear, editar, eliminar, listar, buscar, exportar",
    },
    "administrar": {
        "category": "verbo_impreciso",
        "ieee": "completo",
        "iso": "",
        "severity": "alta",
        "suggestion": "Descomponer en acciones concretas: crear, editar, desactivar, asignar roles",
    },
    "manejar": {
        "category": "verbo_impreciso",
        "ieee": "completo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar qué acciones concretas incluye 'manejar'",
    },
    "procesar": {
        "category": "verbo_impreciso",
        "ieee": "completo",
        "iso": "",
        "severity": "media",
        "suggestion": "Definir qué transformaciones o acciones ocurren durante el 'procesamiento'",
    },
    "controlar": {
        "category": "verbo_impreciso",
        "ieee": "completo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar qué se monitorea, qué acciones se toman, qué umbrales aplican",
    },
    "optimizar": {
        "category": "verbo_impreciso",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "media",
        "suggestion": "Definir qué métrica se mejora y cuál es el valor objetivo",
    },
    "mejorar": {
        "category": "verbo_impreciso",
        "ieee": "verificable",
        "iso": "",
        "severity": "media",
        "suggestion": "Definir qué se mejora, de qué valor a qué valor, cómo se mide",
    },

    # --- Cuantificadores indefinidos ---
    "varios": {
        "category": "cuantificador_indefinido",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar cantidad exacta o rango: 3, 5-10, mínimo 3",
    },
    "algunos": {
        "category": "cuantificador_indefinido",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar cantidad exacta o rango",
    },
    "muchos": {
        "category": "cuantificador_indefinido",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar cantidad exacta o rango",
    },
    "pocos": {
        "category": "cuantificador_indefinido",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "baja",
        "suggestion": "Especificar cantidad exacta o rango",
    },
    "suficiente": {
        "category": "cuantificador_indefinido",
        "ieee": "verificable",
        "iso": "",
        "severity": "media",
        "suggestion": "Definir el umbral mínimo concreto",
    },
    "gran cantidad": {
        "category": "cuantificador_indefinido",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar cantidad o rango: 1000+, más de 10000 registros",
    },

    # --- Roles indefinidos ---
    "usuario": {
        "category": "rol_indefinido",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar el rol: administrador, cliente, analista QA, desarrollador, visitante",
    },
    "usuarios": {
        "category": "rol_indefinido",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar qué tipos de usuarios y sus permisos",
    },

    # --- Temporalidad vaga ---
    "periódicamente": {
        "category": "temporalidad_vaga",
        "ieee": "verificable",
        "iso": "",
        "severity": "alta",
        "suggestion": "Definir frecuencia exacta: cada 5 minutos, diario a las 00:00, semanal los lunes",
    },
    "regularmente": {
        "category": "temporalidad_vaga",
        "ieee": "verificable",
        "iso": "",
        "severity": "alta",
        "suggestion": "Definir frecuencia exacta",
    },
    "frecuentemente": {
        "category": "temporalidad_vaga",
        "ieee": "verificable",
        "iso": "",
        "severity": "media",
        "suggestion": "Definir frecuencia o cantidad por período",
    },
    "en tiempo real": {
        "category": "temporalidad_vaga",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "alta",
        "suggestion": "Definir latencia máxima aceptable: menos de 500ms, menos de 1 segundo",
    },
    "inmediatamente": {
        "category": "temporalidad_vaga",
        "ieee": "verificable",
        "iso": "rendimiento",
        "severity": "media",
        "suggestion": "Definir tiempo máximo en milisegundos o segundos",
    },

    # --- Alcance indefinido ---
    "etc": {
        "category": "alcance_indefinido",
        "ieee": "completo",
        "iso": "",
        "severity": "alta",
        "suggestion": "Enumerar todos los elementos — 'etc.' oculta requerimientos",
    },
    "entre otros": {
        "category": "alcance_indefinido",
        "ieee": "completo",
        "iso": "",
        "severity": "alta",
        "suggestion": "Listar todos los elementos explícitamente",
    },
    "y demás": {
        "category": "alcance_indefinido",
        "ieee": "completo",
        "iso": "",
        "severity": "alta",
        "suggestion": "Enumerar todos los elementos del conjunto",
    },
    "similar": {
        "category": "alcance_indefinido",
        "ieee": "no_ambiguo",
        "iso": "",
        "severity": "media",
        "suggestion": "Especificar exactamente qué características debe compartir",
    },
}

# Mapeo de IEEE 830 violaciones a descripciones
IEEE_830_DESCRIPTIONS = {
    "no_ambiguo": "No ambiguo — debe tener una sola interpretación posible",
    "completo": "Completo — debe cubrir todos los casos sin dejar implícitos",
    "verificable": "Verificable — debe poder probarse con una métrica concreta",
    "trazable": "Trazable — debe poder rastrearse al origen",
}

# Mapeo de ISO 25010 categorías
ISO_25010_DESCRIPTIONS = {
    "rendimiento": "Rendimiento — eficiencia temporal y uso de recursos",
    "seguridad": "Seguridad — confidencialidad, integridad, autenticación",
    "usabilidad": "Usabilidad — facilidad de aprendizaje y operación",
    "confiabilidad": "Confiabilidad — madurez, disponibilidad, tolerancia a fallos",
    "mantenibilidad": "Mantenibilidad — modularidad, reusabilidad, modificabilidad",
}


class AmbiguityDetector:
    """Detecta ambigüedades en requerimientos de software según IEEE 830."""

    def __init__(self, custom_words: dict = None):
        """Inicializa el detector.

        Args:
            custom_words: diccionario adicional de palabras ambiguas
                         con el mismo formato que AMBIGUOUS_WORDS
        """
        self.words = dict(AMBIGUOUS_WORDS)
        if custom_words:
            self.words.update(custom_words)

    def analyze(self, requirement_text: str) -> list[Ambiguity]:
        """Analiza un requerimiento y detecta ambigüedades.

        Args:
            requirement_text: texto del requerimiento a analizar

        Returns:
            lista de ambigüedades detectadas, ordenadas por severidad
        """
        text_lower = requirement_text.lower()
        ambiguities = []
        found_positions = set()  # Evitar duplicados por posición

        for word, info in self.words.items():
            # Buscar la palabra en el texto (como palabra completa)
            pattern = r'\b' + re.escape(word.lower()) + r'\b'
            matches = re.finditer(pattern, text_lower)

            for match in matches:
                pos = match.start()
                if pos in found_positions:
                    continue
                found_positions.add(pos)

                # Extraer contexto (30 chars antes y después)
                ctx_start = max(0, pos - 30)
                ctx_end = min(len(requirement_text), pos + len(word) + 30)
                context = "..." + requirement_text[ctx_start:ctx_end] + "..."

                ambiguities.append(Ambiguity(
                    word=word,
                    category=info["category"],
                    ieee_830_violation=info["ieee"],
                    iso_25010_category=info.get("iso", ""),
                    suggestion=info["suggestion"],
                    context=context.strip(),
                    severity=info["severity"],
                ))

        # Ordenar: alta primero, luego media, luego baja
        severity_order = {"alta": 0, "media": 1, "baja": 2}
        ambiguities.sort(key=lambda a: severity_order.get(a.severity, 3))

        return ambiguities

    def build_prompt_section(self, ambiguities: list[Ambiguity]) -> str:
        """Genera la sección del prompt que indica al LLM qué ambigüedades resolver.

        Args:
            ambiguities: lista de ambigüedades detectadas

        Returns:
            texto formateado para inyectar en el prompt del LLM
        """
        if not ambiguities:
            return ""

        section = "## ⚠️ AMBIGÜEDADES DETECTADAS EN EL REQUERIMIENTO\n"
        section += "El análisis automático encontró las siguientes palabras ambiguas.\n"
        section += "DEBES resolver CADA UNA con valores concretos en tus historias de usuario.\n\n"

        for i, amb in enumerate(ambiguities, 1):
            ieee_desc = IEEE_830_DESCRIPTIONS.get(amb.ieee_830_violation, "")
            section += f"### Ambigüedad {i}: \"{amb.word}\" [{amb.severity.upper()}]\n"
            section += f"- **Categoría:** {amb.category.replace('_', ' ')}\n"
            section += f"- **Viola IEEE 830:** {ieee_desc}\n"
            if amb.iso_25010_category:
                iso_desc = ISO_25010_DESCRIPTIONS.get(amb.iso_25010_category, amb.iso_25010_category)
                section += f"- **ISO 25010:** {iso_desc}\n"
            section += f"- **Contexto:** {amb.context}\n"
            section += f"- **Acción requerida:** {amb.suggestion}\n\n"

        return section

    def build_resolved_prompt_section(self, resolutions: list[dict]) -> str:
        """Genera la sección del prompt cuando las ambigüedades YA fueron resueltas por el analista.

        A diferencia de build_prompt_section (que pide al LLM que resuelva),
        esta sección le INFORMA al LLM las decisiones confirmadas del analista.
        Esto elimina suposiciones — el LLM debe usar estos valores como hechos.

        Args:
            resolutions: lista de dicts con keys:
                - word: palabra ambigua
                - category: categoría de ambigüedad
                - analyst_resolution: texto de resolución del analista
                - status: "resolved" | "dismissed" (el analista dice que no es ambiguo)

        Returns:
            texto formateado para inyectar en el prompt del LLM
        """
        if not resolutions:
            return ""

        resolved = [r for r in resolutions if r["status"] == "resolved"]
        dismissed = [r for r in resolutions if r["status"] == "dismissed"]

        section = "## DECISIONES DEL ANALISTA SOBRE AMBIGÜEDADES\n"
        section += "Las siguientes ambigüedades fueron revisadas y resueltas por el analista.\n"
        section += "DEBES usar estas definiciones como HECHOS, NO como suposiciones.\n"
        section += "En ambiguities_resolved, marca assumption_made: false para todas.\n\n"

        for i, res in enumerate(resolved, 1):
            section += f"### Decisión {i}: \"{res['word']}\"\n"
            section += f"- **Categoría:** {res['category'].replace('_', ' ')}\n"
            section += f"- **Resolución confirmada:** {res['analyst_resolution']}\n\n"

        if dismissed:
            section += "### Términos aceptados como no ambiguos:\n"
            for d in dismissed:
                section += f"- \"{d['word']}\" — el analista confirma que es suficientemente claro\n"
            section += "\n"

        return section

    def generate_report(self, requirement_text: str) -> str:
        """Genera un reporte completo de ambigüedades para consola.

        Args:
            requirement_text: texto a analizar

        Returns:
            reporte formateado para imprimir
        """
        ambiguities = self.analyze(requirement_text)

        report = f"\n{'=' * 60}\n"
        report += "ANÁLISIS DE AMBIGÜEDADES (IEEE 830)\n"
        report += f"{'=' * 60}\n"
        report += f"\nTexto analizado:\n\"{requirement_text}\"\n"

        if not ambiguities:
            report += "\n✅ No se detectaron ambigüedades.\n"
            return report

        report += f"\n⚠️ {len(ambiguities)} ambigüedades detectadas:\n"

        severity_count = {"alta": 0, "media": 0, "baja": 0}
        for amb in ambiguities:
            severity_count[amb.severity] += 1

        report += f"   🔴 Alta: {severity_count['alta']}"
        report += f"   🟡 Media: {severity_count['media']}"
        report += f"   🟢 Baja: {severity_count['baja']}\n"

        for i, amb in enumerate(ambiguities, 1):
            emoji = "🔴" if amb.severity == "alta" else "🟡" if amb.severity == "media" else "🟢"
            report += f"\n{emoji} {i}. \"{amb.word}\" ({amb.category.replace('_', ' ')})\n"
            ieee_desc = IEEE_830_DESCRIPTIONS.get(amb.ieee_830_violation, "")
            report += f"   IEEE 830: {ieee_desc}\n"
            if amb.iso_25010_category:
                iso_desc = ISO_25010_DESCRIPTIONS.get(amb.iso_25010_category, "")
                report += f"   ISO 25010: {iso_desc}\n"
            report += f"   Contexto: {amb.context}\n"
            report += f"   💡 {amb.suggestion}\n"

        return report


# ============================================================
# SCRIPT DE DEMOSTRACIÓN
# ============================================================
if __name__ == "__main__":
    detector = AmbiguityDetector()

    ejemplos = [
        "El sistema debe ser rápido y seguro para los usuarios",
        "Se necesita gestionar el inventario de forma eficiente y escalable",
        "El usuario debe poder administrar sus datos de forma intuitiva y confiable",
        "El sistema debe procesar periódicamente los reportes, entre otros documentos",
        "Necesito un sistema de login seguro para la plataforma",
    ]

    for ejemplo in ejemplos:
        print(detector.generate_report(ejemplo))
        print()
