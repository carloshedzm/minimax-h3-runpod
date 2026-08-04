# MiniMax H3 (Ref2VA) para RunPod Serverless.
#
# Base: el worker oficial de RunPod para ComfyUI. No escribimos handler
# propio — ese worker ya resuelve la cola, el base64 y los errores, y lo
# mantiene RunPod.
#
# Nodos: el plugin de RunningHub, socio de MiniMax. Se elige sobre los nodos
# nativos de ComfyUI porque trae las tres rutas del modelo (T2VA, FL2VA y
# Ref2VA) con offload por capas, lo que permite correr en 24 GB.
#
# Usamos REF2VA y no FL2VA a proposito. Segun la documentacion del plugin,
# un keyframe de FL2VA "ocupa una posicion real de fotograma" — la foto es
# el fotograma 0 y el modelo se aleja de ella — mientras que una referencia
# de Ref2VA "dirige la identidad en vez de convertirse en un fotograma".
# Para videos de una persona concreta hace falta lo segundo.

FROM runpod/worker-comfyui:5.8.6-base

# Triton compila en tiempo de ejecucion los kernels de cuantizacion
# int8_convrot, y para eso necesita un compilador de C y las cabeceras de
# Python. Sin esto falla al cargar los pesos, con un error que no menciona
# la causa.
#
# ffmpeg/ffprobe los exige el plugin para preparar los medios de Ref2VA.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential python3-dev ffmpeg aria2 && \
    rm -rf /var/lib/apt/lists/*

# ComfyUI fijado a una version conocida. El plugin trae su propio runtime de
# H3, asi que no dependemos de los nodos nativos, pero si de CreateVideo y
# SaveVideo, que son de ComfyUI moderno.
#
# CRITICO: se instala en /opt/venv, que es el entorno que realmente ejecuta
# el worker (ver ENV PATH de la imagen base). En /comfyui/.venv hay otro
# entorno que no se usa: instalar ahi deja el runtime con paquetes viejos y
# los pesos int8_convrot fallan en silencio.
RUN cd /comfyui && \
    git fetch --depth 1 origin refs/tags/v0.30.1 && \
    git checkout FETCH_HEAD && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# El plugin de RunningHub. Se fija al commit para que un build futuro no
# traiga otra version sin que sepamos por que cambio el resultado.
ARG PLUGIN_REF=main
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3.git && \
    cd ComfyUI_RH_MinMaxH3 && \
    git checkout "$PLUGIN_REF" && \
    git log -1 --format='PLUGIN %h %ad' --date=short && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Los requirements del plugin arrastran torch a una version cuya rueda por
# defecto es cu130. La flota de RunPod va con driver 570.x (CUDA 12.8), donde
# cu130 no arranca. Se fuerza cu126, que si corre sobre 12.8.
#
# Va DESPUES de instalar el plugin a proposito: si fuera antes, su pip lo
# volveria a subir.
RUN /opt/venv/bin/pip install --no-cache-dir --force-reinstall \
        torch==2.12.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

# Los pesos NO van en la imagen: son 85 GB y ningun runner de CI puede
# exportar una capa de ese tamano. Los baja este script al arrancar, o los
# lee del network volume si hay uno montado.
COPY start-h3.sh /start-h3.sh
RUN chmod +x /start-h3.sh

# Acelera la descarga de los pesos. Es opcional: si faltara, el script de
# arranque reintenta por la via lenta.
RUN /opt/venv/bin/pip install --no-cache-dir hf_transfer

# Comprobaciones en build. Cada una corresponde a algo que ya fallo en
# ejecucion, donde el diagnostico cuesta un arranque de worker entero.
RUN test -f /comfyui/custom_nodes/ComfyUI_RH_MinMaxH3/minimax_h3_nodes/nodes.py && \
    test -x /start.sh && \
    /opt/venv/bin/python -c "import transformers, accelerate, sentencepiece, einops; print('dependencias del plugin OK')" && \
    /opt/venv/bin/python -c "from huggingface_hub import snapshot_download; print('descarga de modelos OK')" && \
    bash -n /start-h3.sh && echo "start-h3.sh OK"

CMD ["/start-h3.sh"]
