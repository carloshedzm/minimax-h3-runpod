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

# torch con cu130. NO bajar esto a cu126.
#
# comfy_kitchen trae tres backends de operaciones cuantizadas — cuda, triton y
# eager — y los dos rapidos EXIGEN cu130. Con cu126 el arranque avisa:
#
#   WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations
#   backend cuda:   'disabled': True
#   backend triton: 'disabled': True
#   backend eager:  'disabled': False
#
# Y todo pasa por `eager`, la ruta ingenua: 348 s para 5 s de video a 480p.
# Los kernels que se quedan fuera son justo los que este modelo necesita
# (int8_linear, convrot_w4a4_linear, scaled_mm_*).
#
# La version anterior usaba cu126 por un aviso de otro worker sobre el driver
# de la flota de RunPod. Pero los chequeos de arranque de nuestros workers
# reportan CUDA 13.0, asi que ese aviso no aplicaba aqui.
#
# IMPORTANTE: el endpoint debe fijar "Minimum CUDA version" en 13.0. En un
# host con 12.8 esta rueda no arranca.
ARG CANAL_TORCH=cu130
RUN /opt/venv/bin/pip install --no-cache-dir --force-reinstall \
        torch==2.12.0 torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${CANAL_TORCH}" && \
    /opt/venv/bin/python -c "import torch; print('torch', torch.__version__)"

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
