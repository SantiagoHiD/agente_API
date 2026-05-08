"""Explorador de la Base de Conocimiento ChromaDB.

Equivalente a hacer SELECT * FROM katary_sgc en una base relacional.

Ejecutar: python ver_kb.py
"""

import warnings
warnings.filterwarnings("ignore")

import chromadb
from pathlib import Path

# Conectar a la misma base de datos que usa el pipeline
kb_path = Path(__file__).parent / "knowledge_base_data"
client = chromadb.PersistentClient(path=str(kb_path))

# Obtener la colección
collection = client.get_collection("katary_sgc")

# ============================================================
# 1. Estadísticas generales (como un COUNT(*))
# ============================================================
print("=" * 60)
print("BASE DE CONOCIMIENTO — ChromaDB")
print("=" * 60)
print(f"\nTotal de historias: {collection.count()}")

# ============================================================
# 2. Ver TODOS los registros (como un SELECT *)
# ============================================================
datos = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

print(f"\n{'=' * 60}")
print("TODAS LAS HISTORIAS (SELECT *)")
print("=" * 60)

dominios = {}
for i in range(len(datos["ids"])):
    story_id = datos["ids"][i]
    texto = datos["documents"][i]
    metadata = datos["metadatas"][i]
    embedding = datos["embeddings"][i]

    dominio = metadata.get("dominio", "sin_dominio")
    dominios[dominio] = dominios.get(dominio, 0) + 1

    print(f"\n📄 [{story_id}]")
    print(f"   Dominio:  {dominio}")
    print(f"   Historia: {texto[:120]}...")
    print(f"   Criterios: {metadata.get('criterios', '')[:120]}...")
    print(f"   Embedding: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ... ] ({len(embedding)} dimensiones)")

# ============================================================
# 3. Resumen por dominio (como un GROUP BY dominio)
# ============================================================
print(f"\n{'=' * 60}")
print("HISTORIAS POR DOMINIO (GROUP BY dominio)")
print("=" * 60)
for dominio, count in sorted(dominios.items()):
    print(f"   {dominio}: {count} historias")

# ============================================================
# 4. Filtrar por dominio (como un WHERE dominio = 'autenticacion')
# ============================================================
print(f"\n{'=' * 60}")
print("FILTRO: WHERE dominio = 'autenticacion'")
print("=" * 60)

filtrados = collection.get(
    where={"dominio": "autenticacion"},
    include=["documents", "metadatas"]
)

for i in range(len(filtrados["ids"])):
    print(f"\n   [{filtrados['ids'][i]}] {filtrados['documents'][i][:100]}...")
