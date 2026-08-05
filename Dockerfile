# MiniMax H3 (Ref2VA) para RunPod Serverless.
#
# Base: el worker oficial de RunPod para ComfyUI. No hay handler propio: ese
# worker ya resuelve la cola, el base64 y los errores, y lo mantiene RunPod.
#
# Nodos: los NATIVOS de ComfyUI. H3 tiene soporte de fabrica desde 0.30.0, asi
# que no hace falta ningun custom node.
#
# POR QUE NO EL PLUGIN DE RUNNINGHUB
#
# Se probo y funciona, pero exige los pesos sin podar del repositorio de
# Gluttony10: 47 GB solo el DiT. En una tarjeta de 48 GB eso obliga a offload
# por capas, que mantiene el modelo entero en RAM del sistema; RunPod mata el
# contenedor por pasarse de su limite de memoria:
#
#   memory_required=56838265372  pins_required=46997542600
#   unhealthy container: triggered memory limits (OOM)
#
# Comfy-Org publica la MISMA particion Ref2VA podada en 20.97 GB. Con ella el
# modelo entra entero en VRAM y desaparecen el offload, el anclaje en RAM y,
# de paso, el arbol de configuraciones en models/diffusers que el plugin
# necesitaba.

FROM runpod/worker-comfyui:5.8.6-base

# Triton compila en tiempo de ejecucion los kernels de cuantizacion
# int8_convrot y necesita un compilador de C y las cabeceras de Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential python3-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ComfyUI 0.30.1: los nodos nativos de H3 aparecen en 0.30.0.
#
# CRITICO: se instala en /opt/venv, que es el entorno que realmente ejecuta el
# worker (ver ENV PATH de la imagen base). En /comfyui/.venv hay otro que no se
# usa; instalar ahi deja el runtime con paquetes viejos y los pesos
# int8_convrot fallan en silencio.
RUN cd /comfyui && \
    git fetch --depth 1 origin refs/tags/v0.30.1 && \
    git checkout FETCH_HEAD && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Los requirements suben torch a una version cuya rueda por defecto es cu130,
# y la flota de RunPod va con driver 570.x (CUDA 12.8), donde cu130 no arranca.
# Se fuerza cu126, que si corre sobre 12.8.
RUN /opt/venv/bin/pip install --no-cache-dir --force-reinstall \
        torch==2.12.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

RUN /opt/venv/bin/pip install --no-cache-dir hf_transfer

# Los pesos (42.5 GB) no van en la imagen: ningun runner de CI puede exportar
# una capa de ese tamano. Los baja este script al arrancar.
COPY start-h3.sh /start-h3.sh
RUN chmod +x /start-h3.sh

# Comprobaciones en build. Cada una corresponde a algo que ya fallo en
# ejecucion, donde diagnosticarlo cuesta un arranque de worker entero.
RUN test -x /start.sh && \
    /opt/venv/bin/python -c "from huggingface_hub import snapshot_download; print('descarga de modelos OK')" && \
    grep -q "MiniMaxH3ReferenceToVideo" /comfyui/comfy_extras/nodes_minimax_h3.py && \
    echo "nodo nativo MiniMaxH3ReferenceToVideo presente" && \
    bash -n /start-h3.sh && echo "start-h3.sh OK"

CMD ["/start-h3.sh"]
