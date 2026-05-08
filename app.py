from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "API funcionando 🚀"

@app.route("/api/run", methods=["POST"])
def run_script():
    try:
        data = request.get_json()
        requerimiento = data.get("requerimiento", "")

        if not requerimiento:
            return jsonify({"error": "Requerimiento vacío"}), 400

        # Ruta absoluta al script
        basedir = os.path.abspath(os.path.dirname(__file__))
        script_path = os.path.join(basedir, "taller_evolucion", "taller_seccion1_rag.py")

        # FORZAR EL USO DEL VENV:
        # Esto busca el python.exe dentro de tu carpeta 'venv'
        venv_python = os.path.join(basedir, "venv", "Scripts", "python.exe")
        
        # Si no existe (por ejemplo en Linux), intenta la ruta de Linux
        if not os.path.exists(venv_python):
            venv_python = os.path.join(basedir, "venv", "bin", "python")

        # Si aún no existe, usamos el comando por defecto (pero lo ideal es el venv)
        python_executable = venv_python if os.path.exists(venv_python) else "python"

        result = subprocess.run(
            [python_executable, "-u", script_path],
            input=requerimiento,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            cwd=basedir,
            encoding="utf-8"
        )

        if result.returncode != 0:
            return jsonify({
                "error": "Error ejecutando script",
                "detalle": result.stderr
            }), 500

        return jsonify(json.loads(result.stdout.strip()))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Render asigna el puerto, usa 5000 si no existe
    app.run(host="0.0.0.0", port=port)