from llama_cpp import Llama
import os

# Ruta al modelo
model_path = "./models/gemma4/gemma-4-E4B-it-Q4_K_M.gguf"

print(f"📁 Verificando archivo: {model_path}")
print(f"📏 Tamaño: {os.path.getsize(model_path) / 1e9:.2f} GB")

print("\n🚗 Cargando modelo (esto puede tardar 1-2 minutos)...")

try:
    llm = Llama(
        model_path=model_path,
        n_ctx=512,  # Contexto pequeño para probar
        n_threads=2,
        n_gpu_layers=0,
        verbose=True,
    )
    print("✅ Modelo cargado exitosamente!")
    
    # Probar una inferencia simple
    respuesta = llm("Hola, responde SOLO 'OK'", max_tokens=10)
    print(f"📝 Respuesta: {respuesta['choices'][0]['text']}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔧 Posibles soluciones:")
    print("   1. pip install --upgrade llama-cpp-python")
    print("   2. Usar el archivo Q8_0 en lugar de Q4_K_M")
    print("   3. Reinstalar: CMAKE_ARGS='-DLLAMA_METAL=OFF' pip install llama-cpp-python --force-reinstall")