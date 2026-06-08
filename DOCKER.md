# Dockerización mínima para el proyecto

Instrucciones rápidas para construir y ejecutar la imagen localmente.

1) Build (imagen local)

```bash
docker build -t modeloia:latest .
```

2) Ejecutar montando la carpeta `models` (recomendado — los modelos son grandes)

```bash
docker run --rm -it -v $(pwd)/models:/app/models modeloia:latest
```

3) Con Docker Compose

```bash
docker-compose up --build
```

Notas:
- El script `IA/preguntar_modelo.py` busca el modelo en `./models/gemma4/...`. Monta la carpeta `models` en `/app/models` como en los ejemplos.
- `llama-cpp-python` puede requerir compilación nativa; la imagen instala herramientas de compilación.
- Los modelos `.gguf` son grandes; evita copiarlos dentro de la imagen y móntalos como volumen.
