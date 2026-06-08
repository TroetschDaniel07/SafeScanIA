from llama_cpp import Llama
import os
import json

# Configuración
MODEL_PATH = "./models/gemma4/gemma-4-E4B-it-Q4_K_M.gguf"

class AsistenteOBD2:
    def __init__(self):
        print("🚗 Inicializando asistente de diagnóstico automotriz...")
        
        # Verificar modelo
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No encuentro el modelo en: {MODEL_PATH}")
        
        print(f"📁 Modelo encontrado: {os.path.getsize(MODEL_PATH) / 1e9:.2f} GB")
        
        # Cargar modelo
        print("⏳ Cargando Gemma 4 (puede tardar 1-2 minutos)...")
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,        # Contexto más largo para conversación
            n_threads=4,       # Ajusta según tu Mac
            n_gpu_layers=0,    # CPU puro
            verbose=False,     # Cambia a True para ver logs
        )
        print("✅ ¡Modelo listo! Puedes hacer preguntas sobre tu auto.\n")
        
        # Historial de conversación
        self.historial = []
    
    def diagnosticar(self, consulta_usuario: str) -> str:
        """
        Consulta a Gemma 4 sobre problemas del auto
        """
        # Construir prompt con rol específico (igual que en tu pipeline)
        prompt = f"""<start_of_turn>user
Eres un asistente de diagnóstico vehicular EXPERTO y AMIGABLE. 
Tu trabajo es ayudar a conductores SIN conocimientos técnicos.

REGLAS IMPORTANTES:
1. Responde SIEMPRE en español, claro y sencillo
2. Usa analogías cotidianas (ej: "es como cuando respiras con un tapabocas")
3. Si no sabes algo, dilo honestamente
4. Prioriza la seguridad del conductor
5. Sé conciso pero completo (máximo 4-5 oraciones)

CONSULTA DEL CONDUCTOR:
{consulta_usuario}

RESPONDE de forma natural, como un mecánico amigable explicando a un cliente:
<end_of_turn>
<start_of_turn>model
"""
        
        # Generar respuesta
        respuesta = self.llm(
            prompt,
            max_tokens=300,
            temperature=0.7,    # Más creativo para conversación
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False,
        )
        
        texto = respuesta["choices"][0]["text"].strip()
        
        # Guardar en historial
        self.historial.append({"pregunta": consulta_usuario, "respuesta": texto})
        
        return texto
    
    def chat(self):
        """
        Modo conversación interactiva
        """
        print("═"*60)
        print("  🔧 ASISTENTE OBD2 - Diagnóstico vehicular")
        print("  💡 Pregúntame sobre luces, ruidos, olores o comportamientos raros")
        print("  🚪 Escribe 'salir', 'exit' o 'quit' para terminar")
        print("═"*60)
        print()
        
        while True:
            try:
                consulta = input("\n🚗 Tú: ").strip()
                
                if consulta.lower() in ['salir', 'exit', 'quit', 'q']:
                    print("\n👋 ¡Cuídate! Recuerda: ante la duda, visita al mecánico.")
                    break
                
                if not consulta:
                    continue
                
                print("\n🤖 Asistente: ", end="", flush=True)
                respuesta = self.diagnosticar(consulta)
                print(respuesta)
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

# ═══════════════════════════════════════════════════════════
# USO DEL PROGRAMA
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Opción 1: Pregunta única
    print("¿Qué modo quieres?")
    print("1. Pregunta única")
    print("2. Chat interactivo")
    
    modo = input("\nElige (1 o 2): ").strip()
    
    asistente = AsistenteOBD2()
    
    if modo == "1":
        consulta = input("\n🚗 ¿Qué te pasa en el auto? ")
        respuesta = asistente.diagnosticar(consulta)
        print(f"\n🤖 {respuesta}")
    
    else:
        asistente.chat()