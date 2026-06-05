#!/bin/bash
# download_models.sh - Descarga los modelos necesarios

echo "📦 Descargando Gemma 4 E2B (2B parámetros, más ligero)..."
pip install huggingface_hub

huggingface-cli download bartowski/gemma-4-2b-it-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ./models/gemma4

echo "✅ Modelo descargado en ./models/gemma4/"