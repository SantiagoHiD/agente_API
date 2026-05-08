# Taller: Evolución del Agente — De RAG básico a Human-in-the-Loop

**Curso:** Calidad de Software y Pruebas Automatizadas — Universidad de Nariño
**Módulo 1:** Requirements Refiner

## Objetivo

Vivir la evolución del agente ejecutando las 4 versiones con el **mismo requerimiento**
y observando cómo cada versión resuelve una limitación de la anterior.

## Instrucciones

1. Ejecutar los 4 scripts **en orden** desde la carpeta `modulo1_requirements_refiner`
2. Usar el **mismo requerimiento** en todas las versiones (se proporciona por defecto)
3. Responder las preguntas de reflexión que aparecen al final de cada script
4. Comparar las salidas entre versiones

## Estructura

| Script | Versión | Concepto | Tiempo |
|--------|---------|----------|--------|
| `taller_seccion1_rag.py` | v1 | RAG básico — texto libre | 15 min |
| `taller_seccion2_json.py` | v2 | JSON + Contract A | 20 min |
| `taller_seccion3_ambiguity.py` | v3 | Detector de ambigüedades | 20 min |
| `taller_seccion4_hitl.py` | v4 | Human-in-the-Loop | 25 min |

**Tiempo total estimado: 80 minutos**

## Requisitos previos

- Python 3.10+ con venv activado
- Archivo `.env` con `GROQ_API_KEY`
- Dependencias instaladas: `pip install sentence-transformers chromadb groq python-dotenv pydantic`
- Base de conocimiento cargada en `knowledge_base_data/`

## Cómo ejecutar

```bash
cd modulo1_requirements_refiner
python taller_evolucion/taller_seccion1_rag.py
python taller_evolucion/taller_seccion2_json.py
python taller_evolucion/taller_seccion3_ambiguity.py
python taller_evolucion/taller_seccion4_hitl.py
```
