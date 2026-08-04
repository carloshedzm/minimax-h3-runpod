# MiniMax H3 (Ref2VA) en RunPod Serverless

Worker para [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) usando
la ruta **Ref2VA**, montado sobre dos piezas que ya existen y se mantienen:

- [`runpod/worker-comfyui`](https://github.com/runpod-workers/worker-comfyui) —
  el worker oficial de RunPod. Resuelve la cola, el base64 y los errores.
- [`ComfyUI_RH_MinMaxH3`](https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3) —
  el plugin de RunningHub, socio de MiniMax. Trae el runtime completo de H3
  con offload por capas.

Aquí no hay handler propio ni conversor de workflows: esas dos cosas ya están
resueltas río arriba.

## Por qué Ref2VA y no FL2VA

H3 puede partir de una foto de dos maneras, y **no son intercambiables**:

| | FL2VA | Ref2VA |
|---|---|---|
| Qué es la foto | el fotograma 0 del vídeo | una referencia sin posición |
| Efecto | el vídeo arranca ahí y se aleja | **gobierna la identidad todo el clip** |
| Resolución | debe coincidir con el lienzo | puede ser la suya |
| Audio de referencia | no | sí, encadenado |

Para vídeos de una persona concreta hace falta Ref2VA. La mayoría de imágenes
públicas traen solo FL2VA, que es la mitad del tamaño pero el camino
equivocado para este uso.

## Pesos

Se descargan **al arrancar el worker**, no en el build: son 85 GB y ningún
runner de CI gratuito puede exportar una capa de ese tamaño.

| Archivo | Tamaño |
|---|---|
| `MiniMax-H3-Ref2VA-int8_convrot.safetensors` | 47.00 GB |
| `qwen3-vl-32b-int8_convrot.safetensors` | 27.14 GB |
| `MiniMax-H3-video_vae.safetensors` | 10.42 GB |
| `MiniMax-H3-audio_vae.safetensors` | 0.61 GB |

De [`Gluttony10/MiniMax-H3-INT8-CONVROT`](https://huggingface.co/Gluttony10/MiniMax-H3-INT8-CONVROT).
La partición FL2VA (otros 47 GB) se excluye a propósito.

**Con un network volume montado en `/runpod-volume`** los pesos se bajan una
sola vez y se comparten entre workers. Sin él, cada worker frío los descarga.

## Configuración del endpoint

| | |
|---|---|
| Imagen | `ghcr.io/<usuario>/minimax-h3-ref2va:latest` |
| GPU | 48 GB cómodo · 24 GB posible con offload por capas |
| Container disk | **280 GB** — tiene que caber la imagen más los 85 GB de pesos |
| Execution timeout | 1800 s |
| FlashBoot | activado |

## Límites reales

- **768p.** El lado corto se fija en 768 y el área máxima es 768×1344. El *2K*
  que anuncia MiniMax lo produce `H3-Regenerate-2K`, un módulo que **no
  liberaron** — su propia ficha lo dice: *"debido a la complejidad del
  sistema, este módulo todavía no está open-source"*. Cualquier montaje local
  que prometa 2K está escalando 768p.
- 5 a 15 segundos a 24 fps.
- Audio estéreo de 32 kHz generado junto al vídeo, no doblado después.

## Detalles que costaron encontrarse

- **El venv que ejecuta es `/opt/venv`**, no `/comfyui/.venv`. Instalar en el
  segundo deja el runtime con paquetes viejos y los pesos `int8_convrot`
  fallan sin decir por qué.
- **torch se fuerza a cu126.** Los requirements lo suben a una versión cuya
  rueda por defecto es cu130, y la flota de RunPod va con driver 570.x
  (CUDA 12.8), donde cu130 no arranca.
- **Triton compila los kernels `int8_convrot` en ejecución**, así que hacen
  falta `build-essential` y `python3-dev` en la imagen.
- **ffmpeg y ffprobe** los exige el plugin para preparar los medios de Ref2VA.

Los tres primeros están documentados por quienes ya pusieron H3 en RunPod
([vincezh2000](https://github.com/vincezh2000/minimax-h3-comfyui-serverless),
[dooglex-rg](https://github.com/dooglex-rg/h3-worker)); el cuarto lo pide el
propio plugin.
