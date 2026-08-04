#!/usr/bin/env bash
set -euo pipefail

# Descarga los pesos de MiniMax H3 y cede el control al arranque oficial
# del worker.
#
# POR QUE NO VAN DENTRO DE LA IMAGEN
#
# Son 85 GB. Ningun runner de CI gratuito puede exportar una capa asi: el
# paso de export de buildkit necesita alrededor del doble del contenido, y
# el archivo mas grande por si solo son 47 GB.
#
# El coste es que el primer arranque de cada worker descarga. Se amortigua
# de dos formas: con FlashBoot, que congela el proceso ya cargado en vez de
# matarlo, y con un network volume, que los deja bajados una sola vez para
# todos los workers.

MODELOS_HF="${MODELOS_HF:-Gluttony10/MiniMax-H3-INT8-CONVROT}"

# El plugin busca los pesos en <models>/MiniMax-H3, y ese nombre viaja en el
# workflow como `model_root`. No cambiarlo sin cambiar tambien el cliente.
NOMBRE_RAIZ="MiniMax-H3"
DESTINO_LOCAL="/comfyui/models/${NOMBRE_RAIZ}"

if [ -d /runpod-volume ]; then
    # Con volumen de red, los pesos sobreviven al worker y se comparten.
    DESTINO="/runpod-volume/models/${NOMBRE_RAIZ}"
    mkdir -p "$DESTINO"
    mkdir -p "$(dirname "$DESTINO_LOCAL")"
    # ComfyUI mira su propia carpeta, asi que se enlaza. Se rehace el enlace
    # en cada arranque por si el worker anterior dejo uno roto.
    rm -rf "$DESTINO_LOCAL"
    ln -s "$DESTINO" "$DESTINO_LOCAL"
    echo "[h3] pesos en el network volume: $DESTINO"
else
    DESTINO="$DESTINO_LOCAL"
    mkdir -p "$DESTINO"
    echo "[h3] sin network volume; los pesos se bajan a este worker"
fi

# Solo la particion Ref2VA. La de FL2VA son otros 47 GB que no usamos: con
# ella la foto seria el fotograma 0 en vez de gobernar la identidad.
#
# hf reanuda descargas a medias y verifica, asi que un worker que muriera a
# mitad no deja un archivo truncado que el siguiente daria por bueno.
FALTA=0
for f in \
    "MiniMax-H3-Ref2VA-int8_convrot.safetensors" \
    "qwen3-vl-32b-int8_convrot.safetensors" \
    "MiniMax-H3-video_vae.safetensors" \
    "MiniMax-H3-audio_vae.safetensors" ; do
    if [ ! -s "$DESTINO/$f" ]; then
        echo "[h3] falta $f"
        FALTA=1
    fi
done

if [ "$FALTA" = "1" ]; then
    echo "[h3] descargando ~85 GB desde $MODELOS_HF (la primera vez tarda)"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    /opt/venv/bin/hf download "$MODELOS_HF" \
        --local-dir "$DESTINO" \
        --exclude "*FL2VA*" \
        || {
            # hf_transfer acelera mucho pero es mas fragil ante cortes de
            # red. Si falla, se reintenta por la via lenta antes de darse
            # por vencido: mejor tardar que perder el arranque entero.
            echo "[h3] fallo la descarga rapida; reintento sin hf_transfer"
            HF_HUB_ENABLE_HF_TRANSFER=0 /opt/venv/bin/hf download "$MODELOS_HF" \
                --local-dir "$DESTINO" --exclude "*FL2VA*"
        }
else
    echo "[h3] pesos ya presentes, no se descarga nada"
fi

echo "[h3] contenido de $DESTINO:"
ls -la "$DESTINO" || true
du -sh "$DESTINO" 2>/dev/null || true

echo "[h3] cediendo el control al arranque oficial del worker"
exec /start.sh
