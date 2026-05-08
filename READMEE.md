# Módulo 1 — Requirements Refiner

**"De la ambigüedad a la precisión"**

Agente de IA que transforma requerimientos ambiguos de software en historias de usuario estructuradas con criterios de aceptación verificables. Este repositorio está diseñado para el **Curso de Calidad de Software y Pruebas Automatizadas — Universidad de Nariño**, y contiene el material para que los estudiantes vivan la evolución de un agente de IA desde un RAG básico hasta un flujo Human-in-the-Loop.

---

## Prerequisitos

- **Python 3.10 o superior**
- **Git** instalado
- **Cuenta en Groq** (gratuita) — para obtener tu API key en https://console.groq.com

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Juan-construsoft/qualityai-modulo1.git
cd qualityai-modulo1
```

### 2. Crear y activar un entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar la API key de Groq

Copia el archivo de ejemplo y edítalo con tu API key:

**Windows:**
```powershell
copy .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Luego abre `.env` y reemplaza el valor de ejemplo por tu API key real:
                
```
GROQ_API_KEY=gsk_tuApiKeyReal...
```

> **Cómo obtener la API key:** Crea una cuenta gratuita en [console.groq.com](https://console.groq.com), ve a *API Keys → Create API Key* y copia el valor.

---

## Estructura del repositorio

```
qualityai-modulo1/
├── agente_refiner.py          # Agente final consolidado (referencia)
├── agente_v1_rag.py           # v1 — RAG básico (texto libre)
├── agente_v2_json.py          # v2 — Salida JSON con Contract A
├── agente_v3_ambiguity.py     # v3 — Detector de ambigüedades
├── agente_v4_hitl.py          # v4 — Human-in-the-Loop
│
├── taller_evolucio