from flask import Flask, request, jsonify
from flask_cors import CORS

from taller_evolucion.taller_seccion1_rag import ejecutar

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "API funcionando 🚀"

@app.route("/api/run", methods=["POST"])
def run_script():
    try:
        data = request.get_json()
        requerimiento = data.get("requerimiento", "")

        if not requerimiento:
            return jsonify({"error": "Requerimiento vacío"}), 400

        resultado = ejecutar(requerimiento)

        return jsonify(resultado)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
