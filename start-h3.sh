#!/usr/bin/env bash
set -euo pipefail

# Sin esto, cualquier fallo mata el script en silencio: RunPod solo dice
# "Job processing failed" y los logs quedan sin una linea que explique por
# que. Nos costo un arranque entero averiguar que faltaba un ejecutable.
trap 'echo "[h3] FALLO en la linea $LINENO (codigo $?). El worker no arranca." >&2' ERR

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

# El plugin necesita DOS cosas y no una:
#
#   models/MiniMax-H3/           los tensores, en archivos sueltos
#   models/diffusers/MiniMax-H3/ la arquitectura: model_index.json, configs,
#                                tokenizer y preprocessor_config
#
# Su README lo dice sin rodeos: "los pesos convertidos por si solos no bastan
# para cargar el modelo". La primera version de este script solo bajaba los
# pesos, y el cargador fallaba con "Could not find MiniMax H3 partition
# 'ref2va' ... inspected no child model_index.json files".
#
# De la segunda solo hacen falta los archivos pequenos: los tensores ya
# estan en la primera. Excluyendo los .safetensors son 145 MB en vez de
# cientos de GB.
UPSTREAM_HF="${UPSTREAM_HF:-MiniMaxAI/MiniMax-H3}"

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

# La arquitectura va en models/diffusers/<raiz>, junto a los pesos pero en
# otro arbol. El plugin busca ahi los model_index.json de cada particion.
DIFFUSERS="/comfyui/models/diffusers/${NOMBRE_RAIZ}"
mkdir -p "$DIFFUSERS"

# Se usa la API de Python y no el ejecutable de linea de comandos porque su
# nombre depende de la version: en huggingface-hub < 1.0 se llama
# `huggingface-cli` y a partir de 1.0 `hf`. La imagen base fija
# explicitamente "huggingface-hub<1.0", asi que `hf` no existe.
#
# snapshot_download reanuda y verifica lo ya bajado, asi que llamarlo en
# cada arranque es barato: si los archivos estan, no descarga nada.
DESTINO="$DESTINO" DIFFUSERS="$DIFFUSERS" \
MODELOS_HF="$MODELOS_HF" UPSTREAM_HF="$UPSTREAM_HF" \
/opt/venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

def bajar(repo, destino, ignorar, que):
    # hf_transfer multiplica la velocidad pero es mas fragil ante cortes de
    # red, y puede no estar instalado. Se intenta con el y se repite sin el.
    for rapido in (True, False):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if rapido else "0"
        try:
            print(f"[h3] {que}: {repo} -> {destino}", flush=True)
            snapshot_download(
                repo_id=repo,
                local_dir=destino,
                ignore_patterns=ignorar,
                max_workers=8,
            )
            return
        except Exception as e:
            if rapido:
                print(f"[h3] fallo con hf_transfer ({e}); reintento sin el", flush=True)
            else:
                raise

# 1) Los tensores. Solo la particion Ref2VA: la de FL2VA son otros 47 GB que
#    no usamos, porque con ella la foto seria el fotograma 0 en vez de
#    gobernar la identidad.
bajar(os.environ["MODELOS_HF"], os.environ["DESTINO"], ["*FL2VA*"], "pesos (~85 GB)")

# 2) La arquitectura. Se excluyen los .safetensors porque los tensores ya
#    vienen del paso anterior: quedan 145 MB de configs y tokenizer en vez de
#    cientos de GB.
bajar(
    os.environ["UPSTREAM_HF"],
    os.environ["DIFFUSERS"],
    ["*.safetensors", "assets/*", "docs/*", "scripts/*"],
    "configs (~145 MB)",
)
print("[h3] descargas terminadas", flush=True)
PY

echo "[h3] pesos en $DESTINO:"
du -sh "$DESTINO" 2>/dev/null || true
echo "[h3] arquitectura en $DIFFUSERS:"
ls -1 "$DIFFUSERS" 2>/dev/null | head || true
test -f "$DIFFUSERS/Ref2VA/model_index.json" \
  && echo "[h3] Ref2VA/model_index.json presente" \
  || echo "[h3] AVISO: falta Ref2VA/model_index.json — el cargador fallara"

echo "[h3] cediendo el control al arranque oficial del worker"
exec /start.sh
