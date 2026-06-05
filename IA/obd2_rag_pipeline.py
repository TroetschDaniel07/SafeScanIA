"""
╔══════════════════════════════════════════════════════════════════════╗
║     PIPELINE RAG — OBD2 + GEMMA 4 (offline, sin cloud)            ║
║     Proyecto: Jornada de Iniciación Científica                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  TÉCNICA: Retrieval-Augmented Generation (RAG)                     ║
║  Lewis et al. (2020). NeurIPS. arXiv:2005.11401                    ║
║                                                                      ║
║  COMPONENTES:                                                        ║
║    1. Embeddings semánticos  → sentence-transformers               ║
║    2. Índice vectorial        → FAISS (Facebook AI)                ║
║    3. LLM local offline       → Gemma 4 vía llama.cpp             ║
║    4. Exportación             → JSON listo para app móvil          ║
╚══════════════════════════════════════════════════════════════════════╝

INSTALACIÓN (ejecutar una vez en tu Mac):
    pip install sentence-transformers faiss-cpu pandas tqdm
    pip install llama-cpp-python

    # Descargar Gemma 4 E4B Q4 (recomendado para Mac x86, ~2.5 GB):
    pip install huggingface_hub
    huggingface-cli download bartowski/gemma-4-4b-it-GGUF \
        --include "*Q4_K_M*" \
        --local-dir ./models/gemma4

USO:
    # Paso 1: construir índice vectorial (una sola vez)
    python3 obd2_rag_pipeline.py --paso 1

    # Paso 2: generar explicaciones para todos los códigos (tarda ~1-2h)
    python3 obd2_rag_pipeline.py --paso 2

    # Paso 3: probar con un código específico
    python3 obd2_rag_pipeline.py --paso 3 --codigo P0171

    # Paso 4: exportar todo a JSON para la app móvil
    python3 obd2_rag_pipeline.py --paso 4
"""

import os
import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("OBD2-RAG")


# ══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN CENTRAL
# Cambia estas rutas según tu entorno
# ══════════════════════════════════════════════════════════════════════

CONFIG = {
    # Ruta al CSV de códigos DTC
    "csv_path": "Powertrain_Codes.csv",

    # Modelo de embeddings: liviano, multilingüe, corre 100% local
    # Referencia: Reimers & Gurevych (2019). arXiv:1908.10084
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

    # Ruta al GGUF de Gemma 4 (descargado con huggingface-cli)
    # Cambia el nombre del archivo si decsargaste otra cuantización
    "gguf_path": "./models/gemma4/gemma-4-E4B-it-Q4_K_M.gguf",

    # Archivos de salida que genera este pipeline
    "faiss_index_path":   "obd2_faiss.index",
    "metadata_path":      "obd2_metadata.json",
    "explicaciones_path": "obd2_explicaciones.json",  # para la app móvil

    # Parámetros del LLM
    "n_ctx":        2048,   # ventana de contexto (tokens)
    "n_threads":    4,      # hilos CPU (ajustar según tu Mac)
    "temperature":  0.2,    # baja temperatura = respuestas más consistentes
    "max_tokens":   300,    # longitud máxima de la respuesta

    # Top-K para recuperación FAISS (cuántos candidatos buscar)
    "top_k": 3,
}


# ══════════════════════════════════════════════════════════════════════
# PASO 1 — CARGAR CSV Y CONSTRUIR ÍNDICE VECTORIAL
# ══════════════════════════════════════════════════════════════════════

def paso1_construir_indice():
    """
    Carga el CSV de códigos DTC, genera embeddings semánticos
    de cada descripción técnica y construye un índice FAISS.

    POR QUÉ EMBEDDINGS:
    Permiten búsqueda semántica en lugar de búsqueda exacta.
    Si el ELM327 devuelve un código desconocido o similar,
    el sistema recupera el código más parecido semánticamente.

    Referencia:
        Reimers, N. & Gurevych, I. (2019).
        Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
        EMNLP 2019. arXiv:1908.10084

    POR QUÉ FAISS:
        Johnson, J., Douze, M., & Jégou, H. (2019).
        Billion-scale similarity search with GPUs.
        IEEE Transactions on Big Data. arXiv:1702.08734
    """
    log.info("PASO 1 — Construyendo índice vectorial")

    # ── Cargar y limpiar CSV
    df = pd.read_csv(CONFIG["csv_path"])
    df["Code"] = df["Code"].str.strip().str.upper()
    df["Condition Description"] = df["Condition Description"].str.strip()
    df = df.dropna(subset=["Code", "Condition Description"])
    log.info(f"CSV cargado: {len(df):,} códigos DTC")

    # ── Preparar textos para embedding
    # Concatenamos código + descripción para riqueza semántica
    textos = [
        f"{row['Code']}: {row['Condition Description']}"
        for _, row in df.iterrows()
    ]

    # ── Cargar modelo de embeddings (se descarga automáticamente la 1ª vez)
    log.info(f"Cargando modelo de embeddings: {CONFIG['embedding_model']}")
    modelo_emb = SentenceTransformer(CONFIG["embedding_model"])

    # ── Generar embeddings (vectores numéricos de 384 dimensiones)
    log.info("Generando embeddings (puede tardar ~2 minutos)...")
    embeddings = modelo_emb.encode(
        textos,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # normalizar para similitud coseno
    )
    log.info(f"Embeddings generados: shape={embeddings.shape}")

    # ── Construir índice FAISS (IndexFlatIP = producto interno = coseno con normalización)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    log.info(f"Índice FAISS construido: {index.ntotal:,} vectores")

    # ── Guardar índice y metadatos
    faiss.write_index(index, CONFIG["faiss_index_path"])
    log.info(f"Índice guardado: {CONFIG['faiss_index_path']}")

    metadata = df.to_dict(orient="records")
    with open(CONFIG["metadata_path"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log.info(f"Metadatos guardados: {CONFIG['metadata_path']}")

    log.info("✅ PASO 1 completado")
    return index, metadata, modelo_emb


# ══════════════════════════════════════════════════════════════════════
# RECUPERACIÓN — buscar código en el índice
# ══════════════════════════════════════════════════════════════════════

def recuperar_dtc(query: str, index, metadata: list, modelo_emb,
                  top_k: int = None) -> list[dict]:
    """
    PASO RETRIEVAL del patrón RAG.

    Dado un código DTC o texto de consulta, busca los registros
    más similares semánticamente en el índice FAISS.

    Referencia (patrón RAG completo):
        Lewis, P. et al. (2020). Retrieval-Augmented Generation for
        Knowledge-Intensive NLP Tasks. NeurIPS. arXiv:2005.11401

    Args:
        query: código DTC (ej. "P0171") o texto libre
        top_k: cuántos candidatos retornar

    Returns:
        lista de dicts con código, descripción y score de similitud
    """
    if top_k is None:
        top_k = CONFIG["top_k"]

    # Normalizar query
    query_limpia = query.strip().upper()

    # 1. Búsqueda exacta primero (si el código existe literalmente)
    exacto = [m for m in metadata if m["Code"] == query_limpia]
    if exacto:
        log.info(f"Código {query_limpia} encontrado exactamente en la base de datos")
        return [{"codigo": exacto[0]["Code"],
                 "sistema": exacto[0]["Trouble Code System"],
                 "descripcion_tecnica": exacto[0]["Condition Description"],
                 "score": 1.0,
                 "tipo_match": "exacto"}]

    # 2. Búsqueda semántica si no hay match exacto
    log.info(f"Código {query_limpia} no encontrado exactamente → búsqueda semántica")
    q_vec = modelo_emb.encode([query_limpia], normalize_embeddings=True)
    scores, indices = index.search(q_vec.astype("float32"), top_k)

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        m = metadata[idx]
        resultados.append({
            "codigo":            m["Code"],
            "sistema":           m["Trouble Code System"],
            "descripcion_tecnica": m["Condition Description"],
            "score":             float(score),
            "tipo_match":        "semantico"
        })

    return resultados


# ══════════════════════════════════════════════════════════════════════
# GENERACIÓN — construir prompt y llamar a Gemma 4
# ══════════════════════════════════════════════════════════════════════

def construir_prompt(codigo: str, descripcion_tecnica: str,
                     sensores: dict = None) -> str:
    """
    PASO AUGMENTED del patrón RAG.

    Construye el prompt enriquecido con el contexto recuperado.
    Aplica técnicas de prompt engineering documentadas:
    - Rol específico (persona pattern)
    - Contexto verificable del CSV (fuente SAE J1979)
    - Restricción de formato JSON estricto
    - Definición explícita de clases de urgencia

    Referencias:
        White, J. et al. (2023). A Prompt Pattern Catalog.
        arXiv:2302.11382

        Wei, J. et al. (2022). Chain-of-Thought Prompting.
        NeurIPS 2022. arXiv:2201.11903
    """
    contexto_sensores = ""
    if sensores:
        contexto_sensores = "\nDATOS DE SENSORES ADICIONALES:\n"
        etiquetas = {
            "rpm":        "RPM del motor",
            "coolant":    "Temperatura del refrigerante (°C)",
            "speed":      "Velocidad (km/h)",
            "maf":        "Flujo de aire MAF (g/s)",
            "throttle":   "Posición mariposa (%)",
            "intake_temp":"Temperatura aire admisión (°C)",
        }
        for k, v in sensores.items():
            label = etiquetas.get(k, k)
            contexto_sensores += f"- {label}: {v}\n"

    return f"""<start_of_turn>user
Eres un asistente de diagnóstico vehicular especializado en explicar fallas de automóviles a conductores sin conocimientos técnicos. Tu tarea es traducir información técnica a lenguaje claro y accionable.

INFORMACIÓN DEL CÓDIGO DE FALLA (fuente: estándar SAE J1979):
- Código OBD2: {codigo}
- Descripción técnica oficial: {descripcion_tecnica}
{contexto_sensores}
INSTRUCCIONES:
1. Explica en 2-3 oraciones simples qué está fallando en el auto, usando analogías cotidianas si ayuda.
2. Determina la urgencia con exactamente uno de estos tres valores:
   - "verde": el conductor puede seguir manejando normalmente
   - "amarillo": debe visitar el taller en los próximos días, no es emergencia
   - "rojo": debe detener el vehículo de forma segura de inmediato
3. Da una acción concreta y simple que el conductor pueda hacer ahora.
4. Responde ÚNICAMENTE con este JSON, sin texto adicional:

{{
  "codigo": "{codigo}",
  "urgencia": "verde|amarillo|rojo",
  "titulo": "frase de máximo 5 palabras",
  "explicacion": "explicación clara para el conductor",
  "accion": "qué debe hacer el conductor ahora mismo"
}}
<end_of_turn>
<start_of_turn>model
"""


def cargar_gemma4():
    """
    Carga Gemma 4 E4B cuantizado vía llama-cpp-python.

    Por qué llama.cpp en Mac x86_64:
    LiteRT (antes TensorFlow Lite) requiere arquitectura ARM para
    aceleración en hardware. En Mac con Intel (x86_64), llama.cpp
    es la alternativa que corre en CPU puro con buen rendimiento.

    Referencia:
        Gerganov, G. (2023). llama.cpp: Efficient LLM inference in C++.
        GitHub. https://github.com/ggerganov/llama.cpp
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        log.error("llama-cpp-python no está instalado.")
        log.error("Instala con: pip install llama-cpp-python")
        raise

    if not os.path.exists(CONFIG["gguf_path"]):
        log.error(f"Modelo no encontrado: {CONFIG['gguf_path']}")
        log.error("Descarga Gemma 4 E4B con:")
        log.error("  huggingface-cli download bartowski/gemma-4-4b-it-GGUF \\")
        log.error("      --include '*Q4_K_M*' --local-dir ./models/gemma4")
        raise FileNotFoundError(f"GGUF no encontrado: {CONFIG['gguf_path']}")

    log.info(f"Cargando Gemma 4 desde: {CONFIG['gguf_path']}")
    log.info("Esto puede tardar 20-60 segundos la primera vez...")

    llm = Llama(
        model_path=CONFIG["gguf_path"],
        n_ctx=CONFIG["n_ctx"],
        n_threads=CONFIG["n_threads"],
        n_gpu_layers=0,     # CPU puro (x86_64 sin Metal/CUDA)
        verbose=False,
    )
    log.info("✅ Gemma 4 cargado en memoria")
    return llm


def generar_explicacion(llm, prompt: str) -> dict:
    """
    PASO GENERATION del patrón RAG.

    Llama a Gemma 4 con el prompt enriquecido y parsea la respuesta JSON.

    Args:
        llm:    instancia de Llama (Gemma 4 cargado)
        prompt: texto del prompt construido en construir_prompt()

    Returns:
        dict con urgencia, titulo, explicacion, accion
    """
    respuesta = llm(
        prompt,
        max_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"],
        stop=["<end_of_turn>", "<start_of_turn>"],
        echo=False,
    )

    texto = respuesta["choices"][0]["text"].strip()

    # Limpiar markdown si el modelo lo incluye
    if "```" in texto:
        partes = texto.split("```")
        for parte in partes:
            parte = parte.strip()
            if parte.startswith("json"):
                parte = parte[4:].strip()
            if parte.startswith("{"):
                texto = parte
                break

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        log.warning("Respuesta no es JSON válido, intentando extraer...")
        # Intentar extraer JSON del texto si hay texto extra
        inicio = texto.find("{")
        fin = texto.rfind("}") + 1
        if inicio != -1 and fin > inicio:
            try:
                return json.loads(texto[inicio:fin])
            except Exception:
                pass
        # Respuesta de fallback
        return {
            "urgencia":    "amarillo",
            "titulo":      "Revisión necesaria",
            "explicacion": "Se detectó una falla en el vehículo. Se recomienda revisión.",
            "accion":      "Lleva el vehículo a un taller para diagnóstico.",
        }


# ══════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL DE INFERENCIA
# (punto de entrada para el backend de tus compañeros)
# ══════════════════════════════════════════════════════════════════════

class OBD2_RAG:
    """
    Sistema RAG completo para diagnóstico OBD2 offline.

    Carga los componentes una sola vez en __init__ y los reutiliza
    para cada llamada a diagnosticar(), optimizando el rendimiento.

    Uso desde el backend:
        rag = OBD2_RAG()
        resultado = rag.diagnosticar("P0171")
        resultado = rag.diagnosticar("P0301", sensores={"rpm": 650, "coolant": 95})
    """

    def __init__(self):
        log.info("Iniciando sistema OBD2-RAG...")

        # Cargar índice vectorial y metadatos
        if not os.path.exists(CONFIG["faiss_index_path"]):
            log.warning("Índice no encontrado. Ejecuta primero el Paso 1.")
            log.warning("  python3 obd2_rag_pipeline.py --paso 1")
            raise FileNotFoundError("Ejecuta el Paso 1 primero.")

        self.index = faiss.read_index(CONFIG["faiss_index_path"])
        with open(CONFIG["metadata_path"], encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Cargar modelo de embeddings
        self.modelo_emb = SentenceTransformer(CONFIG["embedding_model"])

        # Cargar Gemma 4 vía llama.cpp
        self.llm = cargar_gemma4()

        # Caché en memoria para no llamar al LLM dos veces con el mismo código
        self._cache = {}

        log.info("✅ Sistema OBD2-RAG listo")

    def diagnosticar(self, codigo_dtc: str, sensores: dict = None) -> dict:
        """
        Diagnóstica un código DTC y retorna explicación en lenguaje simple.

        Args:
            codigo_dtc: código OBD2 (ej. "P0171", "p0301")
            sensores:   dict opcional con lecturas del ELM327:
                        {
                          "rpm": 650,
                          "coolant": 95,
                          "speed": 0,
                          "maf": 1.2,
                          "throttle": 14,
                          "intake_temp": 45
                        }

        Returns:
            {
              "codigo":      "P0171",
              "urgencia":    "amarillo",
              "titulo":      "Mezcla pobre de combustible",
              "explicacion": "...",
              "accion":      "...",
              "encontrado":  True,
              "tipo_match":  "exacto"
            }
        """
        codigo_dtc = codigo_dtc.strip().upper()

        # Verificar caché (evitar llamadas redundantes al LLM)
        cache_key = f"{codigo_dtc}_{str(sorted(sensores.items()) if sensores else [])}"
        if cache_key in self._cache:
            log.info(f"Retornando resultado en caché para {codigo_dtc}")
            return self._cache[cache_key]

        # ── RETRIEVAL: buscar en índice vectorial
        candidatos = recuperar_dtc(
            codigo_dtc, self.index, self.metadata, self.modelo_emb
        )

        if not candidatos:
            return {
                "codigo":      codigo_dtc,
                "urgencia":    "amarillo",
                "titulo":      "Código no reconocido",
                "explicacion": f"El código {codigo_dtc} no está en nuestra base de datos.",
                "accion":      "Lleva el vehículo a un taller con escáner diagnóstico.",
                "encontrado":  False,
                "tipo_match":  "ninguno",
            }

        # Tomar el candidato más relevante
        mejor = candidatos[0]

        # ── AUGMENTED: construir prompt con contexto recuperado
        prompt = construir_prompt(
            mejor["codigo"],
            mejor["descripcion_tecnica"],
            sensores
        )

        # ── GENERATION: Gemma 4 genera la explicación
        log.info(f"Llamando a Gemma 4 para {codigo_dtc}...")
        t0 = time.time()
        resultado_ia = generar_explicacion(self.llm, prompt)
        t1 = time.time()
        log.info(f"Gemma 4 respondió en {t1-t0:.1f}s")

        # Construir respuesta final
        resultado = {
            "codigo":      codigo_dtc,
            "urgencia":    resultado_ia.get("urgencia", "amarillo"),
            "titulo":      resultado_ia.get("titulo", "Falla detectada"),
            "explicacion": resultado_ia.get("explicacion", "Falla detectada en el vehículo."),
            "accion":      resultado_ia.get("accion", "Consulta con un mecánico."),
            "encontrado":  True,
            "tipo_match":  mejor["tipo_match"],
        }

        self._cache[cache_key] = resultado
        return resultado

    def diagnosticar_multiples(self, codigos: list[str],
                                sensores: dict = None) -> list[dict]:
        """
        Diagnostica múltiples códigos DTC activos al mismo tiempo.
        Retorna la lista ordenada de mayor a menor urgencia (rojo primero).
        """
        orden = {"rojo": 0, "amarillo": 1, "verde": 2}
        resultados = []
        for codigo in codigos:
            r = self.diagnosticar(codigo, sensores)
            resultados.append(r)
            time.sleep(0.3)
        resultados.sort(key=lambda x: orden.get(x["urgencia"], 1))
        return resultados


# ══════════════════════════════════════════════════════════════════════
# PASO 2 — GENERAR EXPLICACIONES PARA TODOS LOS CÓDIGOS
# ══════════════════════════════════════════════════════════════════════

def paso2_generar_todas_explicaciones():
    """
    Genera explicaciones en lenguaje simple para TODOS los códigos
    del CSV usando Gemma 4 y las guarda en un archivo JSON.

    Este archivo JSON es lo que tus compañeros integran en la app:
    es un lookup instantáneo sin necesidad de correr ningún LLM
    en el teléfono del usuario.

    Estrategia:
    - Si ya existe el JSON de explicaciones, retoma desde donde quedó
    - Guarda cada 50 códigos para no perder progreso
    """
    log.info("PASO 2 — Generando explicaciones para todos los códigos DTC")

    # Cargar componentes
    index = faiss.read_index(CONFIG["faiss_index_path"])
    with open(CONFIG["metadata_path"], encoding="utf-8") as f:
        metadata = json.load(f)
    modelo_emb = SentenceTransformer(CONFIG["embedding_model"])
    llm = cargar_gemma4()

    # Cargar progreso previo si existe
    explicaciones = {}
    if os.path.exists(CONFIG["explicaciones_path"]):
        with open(CONFIG["explicaciones_path"], encoding="utf-8") as f:
            explicaciones = json.load(f)
        log.info(f"Retomando progreso: {len(explicaciones)} códigos ya procesados")

    pendientes = [m for m in metadata if m["Code"] not in explicaciones]
    log.info(f"Códigos pendientes: {len(pendientes):,}")

    for i, registro in enumerate(tqdm(pendientes, desc="Procesando códigos")):
        codigo = registro["Code"]
        prompt = construir_prompt(codigo, registro["Condition Description"])

        try:
            resultado = generar_explicacion(llm, prompt)
            resultado["codigo"] = codigo
            resultado["descripcion_tecnica"] = registro["Condition Description"]
            explicaciones[codigo] = resultado
        except Exception as e:
            log.warning(f"Error procesando {codigo}: {e}")
            explicaciones[codigo] = {
                "codigo":      codigo,
                "urgencia":    "amarillo",
                "titulo":      "Falla detectada",
                "explicacion": f"Error al procesar código {codigo}. Consulta con mecánico.",
                "accion":      "Lleva el vehículo a un taller para diagnóstico.",
                "descripcion_tecnica": registro["Condition Description"],
            }

        # Guardar progreso cada 50 códigos
        if (i + 1) % 50 == 0:
            with open(CONFIG["explicaciones_path"], "w", encoding="utf-8") as f:
                json.dump(explicaciones, f, ensure_ascii=False, indent=2)
            log.info(f"Progreso guardado: {len(explicaciones)} códigos")

    # Guardar archivo final
    with open(CONFIG["explicaciones_path"], "w", encoding="utf-8") as f:
        json.dump(explicaciones, f, ensure_ascii=False, indent=2)

    log.info(f"✅ PASO 2 completado: {len(explicaciones):,} códigos procesados")
    log.info(f"   Archivo generado: {CONFIG['explicaciones_path']}")


# ══════════════════════════════════════════════════════════════════════
# PASO 3 — PROBAR CON UN CÓDIGO ESPECÍFICO
# ══════════════════════════════════════════════════════════════════════

def paso3_probar_codigo(codigo: str, sensores: dict = None):
    """
    Prueba el pipeline completo con un código DTC específico.
    Muestra el resultado en pantalla de forma legible.
    """
    log.info(f"PASO 3 — Probando código: {codigo}")

    rag = OBD2_RAG()
    resultado = rag.diagnosticar(codigo, sensores)

    colores = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}
    icono = colores.get(resultado["urgencia"], "⚪")

    print("\n" + "═"*60)
    print(f"  DIAGNÓSTICO: {resultado['codigo']}")
    print("═"*60)
    print(f"  {icono}  {resultado['urgencia'].upper()} — {resultado['titulo']}")
    print(f"\n  {resultado['explicacion']}")
    print(f"\n  → ACCIÓN: {resultado['accion']}")
    print(f"\n  Encontrado: {resultado['encontrado']} ({resultado['tipo_match']})")
    print("═"*60)

    return resultado


# ══════════════════════════════════════════════════════════════════════
# PASO 4 — EXPORTAR PARA LA APP MÓVIL
# ══════════════════════════════════════════════════════════════════════

def paso4_exportar_para_app():
    """
    Exporta los archivos que necesitan integrar tus compañeros:

    1. obd2_explicaciones.json  → lookup rápido por código DTC
                                  (no requiere LLM en el teléfono)
    2. obd2_faiss.index         → índice vectorial para búsqueda semántica
                                  (por si quieren búsqueda flexible)
    3. obd2_metadata.json       → metadatos de todos los códigos

    ESTRATEGIA DE DESPLIEGUE RECOMENDADA PARA LA APP:
    - El JSON de explicaciones se embebe en la app como asset estático
    - El teléfono hace un lookup O(1) por código: sin LLM, sin internet
    - Solo se llama al sistema RAG (online/Python) para códigos nuevos
      o no encontrados en el JSON local
    """
    if not os.path.exists(CONFIG["explicaciones_path"]):
        log.error("No existe el archivo de explicaciones. Ejecuta primero el Paso 2.")
        return

    with open(CONFIG["explicaciones_path"], encoding="utf-8") as f:
        explicaciones = json.load(f)

    # Estadísticas por urgencia
    urgencias = {"verde": 0, "amarillo": 0, "rojo": 0, "desconocido": 0}
    for v in explicaciones.values():
        u = v.get("urgencia", "desconocido")
        urgencias[u] = urgencias.get(u, 0) + 1

    print("\n" + "═"*60)
    print("  EXPORTACIÓN PARA APP MÓVIL")
    print("═"*60)
    print(f"  Total códigos procesados: {len(explicaciones):,}")
    print(f"  🟢 Verde    (puede seguir): {urgencias['verde']:,}")
    print(f"  🟡 Amarillo (ir al taller): {urgencias['amarillo']:,}")
    print(f"  🔴 Rojo     (detener ya!):  {urgencias['rojo']:,}")
    print(f"\n  Archivos para tus compañeros:")
    print(f"  ✅ {CONFIG['explicaciones_path']}  ← principal (lookup offline)")
    print(f"  ✅ {CONFIG['faiss_index_path']}      ← índice semántico opcional")
    print(f"  ✅ {CONFIG['metadata_path']}        ← metadatos completos")
    print("═"*60)

    log.info("✅ PASO 4 completado")


# ══════════════════════════════════════════════════════════════════════
# EJEMPLO DE ENTRADA/SALIDA DOCUMENTADO
# ══════════════════════════════════════════════════════════════════════

EJEMPLO_IO = """
EJEMPLO DE ENTRADA Y SALIDA
════════════════════════════

ENTRADA:
    codigo_dtc = "P0171"
    sensores = {
        "rpm":         1200,
        "coolant":     88,
        "speed":       60,
        "maf":         4.1,
        "throttle":    22,
        "intake_temp": 35
    }

PROCESO INTERNO (RAG):
    1. RETRIEVAL → CSV lookup: P0171 = "System Too Lean (Bank 1)"
    2. AUGMENTED → prompt enriquecido con descripción técnica + sensores
    3. GENERATION → Gemma 4 genera explicación en español simple

SALIDA:
    {
      "codigo":      "P0171",
      "urgencia":    "amarillo",
      "titulo":      "Mezcla pobre de combustible",
      "explicacion": "Tu motor está recibiendo muy poco combustible o
                      demasiado aire. Puede ser porque el filtro de aire
                      está sucio, hay una fuga de aire, o el sensor de
                      oxígeno está fallando.",
      "accion":      "Puedes seguir manejando con cuidado, pero agenda
                      una revisión esta semana. Revisa el filtro de aire
                      como primera medida.",
      "encontrado":  True,
      "tipo_match":  "exacto"
    }
"""


# ══════════════════════════════════════════════════════════════════════
# REFERENCIAS CIENTÍFICAS COMPLETAS
# ══════════════════════════════════════════════════════════════════════

REFERENCIAS = """
REFERENCIAS CIENTÍFICAS — Pipeline RAG OBD2
════════════════════════════════════════════

[1] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V.,
    Goyal, N., ... & Kiela, D. (2020).
    Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
    NeurIPS 2020. arXiv:2005.11401
    https://arxiv.org/abs/2005.11401
    → Fundamento del patrón RAG: Retrieval + Augmented + Generation.

[2] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... (2023).
    Retrieval-Augmented Generation for Large Language Models: A Survey.
    arXiv:2312.10997. https://arxiv.org/abs/2312.10997
    → Survey del estado del arte en RAG, justifica la arquitectura.

[3] Reimers, N. & Gurevych, I. (2019).
    Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
    EMNLP 2019. arXiv:1908.10084
    https://arxiv.org/abs/1908.10084
    → Fundamento del modelo de embeddings paraphrase-multilingual-MiniLM.

[4] Johnson, J., Douze, M., & Jégou, H. (2019).
    Billion-scale similarity search with GPUs.
    IEEE Transactions on Big Data. arXiv:1702.08734
    https://arxiv.org/abs/1702.08734
    → Fundamento del índice FAISS para búsqueda vectorial eficiente.

[5] White, J., Fu, Q., Hays, S., Sandborn, M., Olea, C., ... (2023).
    A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT.
    arXiv:2302.11382. https://arxiv.org/abs/2302.11382
    → Justifica las técnicas de diseño de prompt (rol, formato JSON, etc.).

[6] Google DeepMind. (2026). Gemma 4: Technical Report.
    https://ai.google.dev/gemma/docs/core
    → Modelo LLM utilizado para generación de texto offline.

[7] Gerganov, G. (2023). llama.cpp: Efficient LLM Inference in C++.
    GitHub. https://github.com/ggerganov/llama.cpp
    → Runtime de inferencia local que permite correr Gemma 4 sin GPU.

[8] SAE International. (2012). SAE J1979: E/E Diagnostic Test Modes.
    https://www.sae.org/standards/content/j1979_201202/
    → Estándar que define los códigos DTC y PIDs usados en el CSV.

[9] Michailidis, E. T., Panagiotopoulou, A., & Papadakis, A. (2025).
    A Review of OBD-II-Based Machine Learning Applications for
    Sustainable, Efficient, Secure, and Safe Vehicle Driving.
    Sensors, 25(13), 4057. https://doi.org/10.3390/s25134057
    → Revisión científica reciente de IA aplicada a diagnóstico OBD2.

[10] Wei, J., Wang, X., Schuurmans, D., Bosma, M., ... (2022).
     Chain-of-Thought Prompting Elicits Reasoning in LLMs.
     NeurIPS 2022. arXiv:2201.11903
     https://arxiv.org/abs/2201.11903
     → Justifica el uso de instrucciones paso a paso en el prompt.
"""


# ══════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline RAG para diagnóstico OBD2 con Gemma 4"
    )
    parser.add_argument("--paso", type=int, choices=[1, 2, 3, 4],
                        help="Paso a ejecutar (1=índice, 2=generar, 3=probar, 4=exportar)")
    parser.add_argument("--codigo", type=str, default="P0171",
                        help="Código DTC para el Paso 3 (default: P0171)")
    parser.add_argument("--referencias", action="store_true",
                        help="Mostrar referencias científicas")
    parser.add_argument("--ejemplo", action="store_true",
                        help="Mostrar ejemplo de entrada/salida")

    args = parser.parse_args()

    if args.referencias:
        print(REFERENCIAS)

    if args.ejemplo:
        print(EJEMPLO_IO)

    if args.paso == 1:
        paso1_construir_indice()

    elif args.paso == 2:
        paso2_generar_todas_explicaciones()

    elif args.paso == 3:
        # Ejemplo con sensores opcionales
        sensores_ejemplo = {
            "rpm":         1200,
            "coolant":     88,
            "speed":       60,
            "maf":         4.1,
            "throttle":    22,
            "intake_temp": 35,
        }
        paso3_probar_codigo(args.codigo, sensores_ejemplo)

    elif args.paso == 4:
        paso4_exportar_para_app()

    else:
        print("\nUso:")
        print("  python3 obd2_rag_pipeline.py --paso 1   # Construir índice vectorial")
        print("  python3 obd2_rag_pipeline.py --paso 2   # Generar todas las explicaciones")
        print("  python3 obd2_rag_pipeline.py --paso 3 --codigo P0171   # Probar un código")
        print("  python3 obd2_rag_pipeline.py --paso 4   # Exportar para app móvil")
        print("  python3 obd2_rag_pipeline.py --referencias")
        print("  python3 obd2_rag_pipeline.py --ejemplo")