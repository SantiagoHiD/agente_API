"""Base de Conocimiento Vectorial para el Agente 1.

Usa embeddings (Hugging Face sentence-transformers) y ChromaDB para almacenar
y buscar historias de usuario similares del SGC de Katary.

Conceptos clave:
- Embedding: representación numérica del significado de un texto (vector de 384-1024 dimensiones)
- Similitud coseno: medida de qué tan similares son dos vectores (0 = nada, 1 = idénticos)
- RAG (Retrieval-Augmented Generation): buscar contexto relevante antes de llamar al LLM

Dependencias:
    pip install sentence-transformers chromadb

Uso:
    kb = KnowledgeBase()
    kb.load_from_json("examples/knowledge_base/katary_stories.json")
    similares = kb.search("El sistema debe gestionar inventario", top_k=3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    """Base de conocimiento vectorial para historias de usuario.

    Almacena historias de usuario como embeddings en ChromaDB.
    Permite búsqueda semántica: dado un requerimiento nuevo,
    encuentra las historias más similares en significado (no en palabras).
    """

    def __init__(
        self,
        persist_dir: str = "./knowledge_base_data",
        collection_name: str = "katary_user_stories",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Inicializa la base de conocimiento.

        Args:
            persist_dir: directorio donde se almacenan los datos de ChromaDB
            collection_name: nombre de la colección en ChromaDB
            embedding_model: modelo de Hugging Face para generar embeddings.
                Opciones recomendadas:
                - 'sentence-transformers/all-MiniLM-L6-v2' (80MB, rápido, OK con español)
                - 'intfloat/multilingual-e5-large' (1.1GB, excelente con español)
                - 'BAAI/bge-m3' (estado del arte multilingüe)
        """
        # Modelo de embeddings — se descarga automáticamente de Hugging Face
        # la primera vez. Después se usa desde cache local.
        print(f"Cargando modelo de embeddings: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # Base de datos vectorial — persiste en disco
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Historias de usuario del SGC de Katary Software",
                "source": "CMMI-DEV Nivel 3 — 19 años de experiencia",
            },
        )

        print(f"Base de conocimiento inicializada: {self.collection.count()} documentos")

    def add_story(
        self,
        story_id: str,
        texto: str,
        criterios: str = "",
        dominio: str = "general",
        proyecto: str = "",
        metadata_extra: Optional[dict] = None,
    ) -> None:
        """Agrega una historia de usuario a la base de conocimiento.

        Args:
            story_id: identificador único (ej: "SGC-US-001")
            texto: texto completo de la historia de usuario
            criterios: criterios de aceptación en formato libre o Given/When/Then
            dominio: categoría temática (ej: "autenticacion", "reportes", "inventario")
            proyecto: nombre del proyecto de origen
            metadata_extra: metadatos adicionales
        """
        # Generar embedding del texto
        embedding = self.embedder.encode(texto).tolist()

        # Preparar metadatos
        metadata = {
            "dominio": dominio,
            "criterios": criterios,
            "proyecto": proyecto,
        }
        if metadata_extra:
            metadata.update(metadata_extra)

        # Almacenar en ChromaDB
        self.collection.add(
            ids=[story_id],
            embeddings=[embedding],
            documents=[texto],
            metadatas=[metadata],
        )

    def load_from_json(self, filepath: str) -> int:
        """Carga historias de usuario desde un archivo JSON.

        El JSON debe tener la estructura:
        [
            {
                "id": "SGC-US-001",
                "texto": "Como [rol], quiero [acción], para que [beneficio]",
                "criterios": "GIVEN ... WHEN ... THEN ...",
                "dominio": "gestion_proyectos",
                "proyecto": "Katary360"
            },
            ...
        ]

        Args:
            filepath: ruta al archivo JSON

        Returns:
            cantidad de historias cargadas
        """
        with open(filepath, "r", encoding="utf-8") as f:
            stories = json.load(f)

        # Procesamiento en lote para eficiencia
        ids = [s["id"] for s in stories]
        textos = [s["texto"] for s in stories]
        metadatas = [
            {
                "dominio": s.get("dominio", "general"),
                "criterios": s.get("criterios", ""),
                "proyecto": s.get("proyecto", ""),
            }
            for s in stories
        ]

        # Generar todos los embeddings en una sola llamada (mucho más rápido)
        print(f"Generando embeddings para {len(textos)} historias...")
        embeddings = self.embedder.encode(textos, show_progress_bar=True).tolist()

        # Almacenar todo de una vez
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=textos,
            metadatas=metadatas,
        )

        print(f"Cargadas {len(stories)} historias. Total en KB: {self.collection.count()}")
        return len(stories)

    def search(self, query: str, top_k: int = 3, dominio: Optional[str] = None) -> list[dict]:
        """Busca las historias más similares a un requerimiento.

        Usa similitud coseno entre el embedding del query y los embeddings almacenados.

        Args:
            query: texto del requerimiento a buscar
            top_k: cantidad de resultados a retornar
            dominio: filtrar por dominio específico (opcional)

        Returns:
            lista de diccionarios con las historias más similares:
            [
                {
                    "id": "SGC-US-001",
                    "documento": "Como líder de proyecto...",
                    "criterios": "GIVEN ... WHEN ... THEN ...",
                    "dominio": "gestion_proyectos",
                    "similitud": 0.85
                }
            ]
        """
        # Generar embedding del query
        query_embedding = self.embedder.encode(query).tolist()

        # Preparar filtro por dominio si se especifica
        where_filter = {"dominio": dominio} if dominio else None

        # Buscar en ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

        # Formatear resultados
        historias = []
        for i in range(len(results["documents"][0])):
            # ChromaDB retorna distancia (0 = idéntico), convertimos a similitud
            similitud = 1 - results["distances"][0][i]
            historias.append(
                {
                    "id": results["ids"][0][i],
                    "documento": results["documents"][0][i],
                    "criterios": results["metadatas"][0][i].get("criterios", ""),
                    "dominio": results["metadatas"][0][i].get("dominio", ""),
                    "proyecto": results["metadatas"][0][i].get("proyecto", ""),
                    "similitud": round(similitud, 4),
                }
            )

        return historias

    def build_context_for_prompt(self, query: str, top_k: int = 3) -> str:
        """Busca historias similares y las formatea como contexto para el prompt del LLM.

        Este es el método que el Agente 1 con RAG llama antes de invocar al LLM.
        Retorna un string listo para insertar en el system prompt.

        Args:
            query: texto del requerimiento
            top_k: cantidad de ejemplos a incluir

        Returns:
            string con las historias formateadas como contexto
        """
        similares = self.search(query, top_k=top_k)

        if not similares:
            return "(No se encontraron historias similares en la base de conocimiento)"

        contexto = "## HISTORIAS DE REFERENCIA DEL SGC DE KATARY\n"
        contexto += "Las siguientes historias son de proyectos reales anteriores con calidad CMMI-DEV L3.\n"
        contexto += "Úsalas como referencia de estilo, profundidad y calidad.\n\n"

        for i, h in enumerate(similares, 1):
            contexto += f"### Referencia {i} (similitud: {h['similitud']:.0%})\n"
            contexto += f"**Historia:** {h['documento']}\n"
            if h["criterios"]:
                contexto += f"**Criterios de aceptación:** {h['criterios']}\n"
            if h["dominio"]:
                contexto += f"**Dominio:** {h['dominio']}\n"
            contexto += "\n"

        return contexto

    def stats(self) -> dict:
        """Retorna estadísticas de la base de conocimiento."""
        count = self.collection.count()

        # Obtener todos los metadatos para análisis
        if count > 0:
            all_data = self.collection.get()
            dominios = {}
            proyectos = {}
            for meta in all_data["metadatas"]:
                dom = meta.get("dominio", "sin_dominio")
                proy = meta.get("proyecto", "sin_proyecto")
                dominios[dom] = dominios.get(dom, 0) + 1
                proyectos[proy] = proyectos.get(proy, 0) + 1
        else:
            dominios = {}
            proyectos = {}

        return {
            "total_historias": count,
            "dominios": dominios,
            "proyectos": proyectos,
            "modelo_embeddings": self.embedder.get_sentence_embedding_dimension(),
        }


# ----- Script de demostración -----
if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Base de Conocimiento Vectorial para QualityAI")
    print("=" * 60)

    # 1. Crear base de conocimiento
    kb = KnowledgeBase(persist_dir="./demo_kb")

    # 2. Cargar historias de ejemplo
    kb.load_from_json("examples/knowledge_base/katary_stories.json")

    # 3. Buscar similares para un requerimiento nuevo
    nuevo_req = "El sistema debe permitir a los gerentes ver el progreso de sus proyectos"
    print(f"\nBuscando similares para: '{nuevo_req}'")
    print("-" * 60)

    resultados = kb.search(nuevo_req, top_k=3)
    for r in resultados:
        print(f"  [{r['similitud']:.0%}] {r['documento'][:100]}...")

    # 4. Generar contexto para el prompt
    print("\n\nContexto generado para el prompt del LLM:")
    print("-" * 60)
    print(kb.build_context_for_prompt(nuevo_req))

    # 5. Estadísticas
    print("\nEstadísticas de la base:")
    stats = kb.stats()
    print(f"  Total historias: {stats['total_historias']}")
    print(f"  Dimensiones del embedding: {stats['modelo_embeddings']}")
    print(f"  Dominios: {stats['dominios']}")
