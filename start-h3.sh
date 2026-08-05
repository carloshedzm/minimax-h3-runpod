#!/usr/bin/env bash
set -euo pipefail

# Sin esto, cualquier fallo mata el script en silencio: RunPod solo dice
# "Job processing failed" y los logs quedan sin una linea que lo explique.
trap 'echo "[h3] FALLO en la linea $LINENO (codigo $?). El worker no arranca." >&2' ERR

# Descarga los pesos de MiniMax H3 y cede el control al arranque oficial del
# worker.
#
# Van aqui y no dentro de la imagen porque son 42.5 GB: ningun runner de CI
# gratuito puede exportar una capa asi. El coste es que el primer arranque de
# cada worker descarga, pero se midio en unos 70 segundos para 80 GB, asi que
# no compensa montar un network volume solo por eso.

REPO="${REPO_PESOS:-Comfy-Org/MiniMax-H3}"

# Los nodos nativos leen de las carpetas estandar de ComfyUI. No hace falta
# ningun arbol de configuraciones aparte: eso era una exigencia del plugin de
# RunningHub, que ya no usamos.
D=/comfyui/models
mkdir -p "$D/diffusion_models" "$D/text_encoders" "$D/vae"

# Solo la particion Ref2VA y solo la variante PODADA.
#
#   ref2va_pruned_int8_convrot   20.97 GB   <- esta
#   ref2va_int8_convrot          34.04 GB
#   ref2va_bf16                  66.28 GB
#
# La podada conserva unicamente lo que Ref2VA necesita, y con 21 GB el modelo
# entra entero en una tarjeta de 48 GB. Las otras obligan a offload por capas,
# que mantiene el modelo en RAM del sistema y hace que RunPod mate el
# contenedor por exceso de memoria.
ARCHIVOS=(
  "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "vae/minimax_h3_video_vae_fp16.safetensors"
  "vae/minimax_h3_audio_vae_fp32.safetensors"
)

FALTA=0
for f in "${ARCHIVOS[@]}"; do
    [ -s "$D/$f" ] || { echo "[h3] falta $f"; FALTA=1; }
done

if [ "$FALTA" = "1" ]; then
    # Se usa la API de Python y no el ejecutable de linea de comandos porque su
    # nombre depende de la version: en huggingface-hub < 1.0 se llama
    # `huggingface-cli` y a partir de 1.0 `hf`. La imagen base fija
    # "huggingface-hub<1.0", asi que `hf` no existe.
    REPO="$REPO" DESTINO="$D" ARCHIVOS="${ARCHIVOS[*]}" /opt/venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ["REPO"]
destino = os.environ["DESTINO"]
patrones = os.environ["ARCHIVOS"].split()

for rapido in (True, False):
    # hf_transfer multiplica la velocidad pero es mas fragil ante cortes de
    # red. Se intenta con el y se repite sin el.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if rapido else "0"
    try:
        print(f"[h3] descargando 42.5 GB de {repo}", flush=True)
        snapshot_download(
            repo_id=repo,
            local_dir=destino,
            allow_patterns=patrones,
            max_workers=8,
        )
        break
    except Exception as e:
        if rapido:
            print(f"[h3] fallo con hf_transfer ({e}); reintento sin el", flush=True)
        else:
            raise
print("[h3] descarga terminada", flush=True)
PY
else
    echo "[h3] pesos ya presentes, no se descarga nada"
fi

echo "[h3] modelos:"
for f in "${ARCHIVOS[@]}"; do
    if [ -s "$D/$f" ]; then
        printf '  %8s  %s\n' "$(du -h "$D/$f" | cut -f1)" "$f"
    else
        echo "  AUSENTE  $f"
    fi
done

echo "[h3] cediendo el control al arranque oficial del worker"
exec /start.sh
