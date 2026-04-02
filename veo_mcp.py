"""
Servidor MCP para geração de vídeos com Veo 3.1 via Google AI Studio (Gemini API)

Instalação das dependências:
    pip install mcp google-genai google-cloud-storage

Uso:
    Defina a variável de ambiente GEMINI_API_KEY com sua chave do Google AI Studio.
    Para upload no GCS, configure as credenciais do GCP:
        export GOOGLE_APPLICATION_CREDENTIALS="/caminho/para/service_account.json"
    Execute: python veo_mcp_server.py
"""

import asyncio
import os
import time
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from google import genai
from google.genai import types


# ──────────────────────────────────────────────
# Inicialização
# ──────────────────────────────────────────────

app = Server("veo-video-generator")

def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Variável de ambiente GEMINI_API_KEY não definida. "
            "Obtenha sua chave em https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


# ──────────────────────────────────────────────
# Definição das ferramentas (tools)
# ──────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_video",
            description=(
                "Gera um vídeo de 8 segundos usando o modelo Veo 3.1 do Google. "
                "Suporta geração por texto, por imagem inicial/final, ou por imagens de referência. "
                "Retorna o caminho do arquivo .mp4 salvo localmente."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Descrição em texto do vídeo a ser gerado.",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Descrição do que NÃO deve aparecer no vídeo (opcional).",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["16:9", "9:16"],
                        "description": "Proporção do vídeo. Padrão: 16:9 (paisagem). Use 9:16 para retrato/vertical.",
                        "default": "16:9",
                    },
                    "resolution": {
                        "type": "string",
                        "enum": ["720p", "1080p", "4k"],
                        "description": "Resolução do vídeo. Padrão: 720p.",
                        "default": "720p",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "enum": [4, 6, 8],
                        "description": "Duração do vídeo em segundos (4, 6 ou 8). Padrão: 8.",
                        "default": 8,
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Caminho completo onde o vídeo .mp4 será salvo. Padrão: ./output_video.mp4",
                        "default": "./output_video.mp4",
                    },
                    "model_variant": {
                        "type": "string",
                        "enum": [
                            "veo-3.1-generate-preview",
                            "veo-3.1-fast-generate-preview",
                            "veo-3.1-lite-generate-preview",
                        ],
                        "description": (
                            "Variante do modelo Veo 3.1. "
                            "Padrão: veo-3.1-generate-preview (máxima qualidade). "
                            "Fast: mais rápido e barato. Lite: mais econômico ainda."
                        ),
                        "default": "veo-3.1-generate-preview",
                    },
                    "first_frame_image_path": {
                        "type": "string",
                        "description": "Caminho para imagem usada como primeiro frame do vídeo (opcional).",
                    },
                    "last_frame_image_path": {
                        "type": "string",
                        "description": "Caminho para imagem usada como último frame do vídeo (opcional).",
                    },
                    "reference_image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de até 3 caminhos de imagens de referência para guiar o conteúdo do vídeo (opcional).",
                        "maxItems": 3,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Semente para geração determinística (opcional). Valor entre 0 e 4294967295.",
                    },
                    "gcs_bucket_path": {
                        "type": "string",
                        "description": (
                            "Caminho do bucket GCS onde o vídeo será salvo após a geração (opcional). "
                            "Formatos aceitos: 'meu-bucket', 'meu-bucket/pasta/', 'gs://meu-bucket/pasta/video.mp4'. "
                            "Requer GOOGLE_APPLICATION_CREDENTIALS configurado. "
                            "Se informado, o retorno incluirá o URI gs:// completo do arquivo."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="extend_video",
            description=(
                "Estende um vídeo previamente gerado com Veo. "
                "Cada extensão adiciona até 8 segundos, continuando a partir do último segundo do clipe anterior."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Descrição do que deve acontecer na extensão do vídeo.",
                    },
                    "video_path": {
                        "type": "string",
                        "description": "Caminho para o arquivo .mp4 a ser estendido.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Caminho onde o vídeo estendido será salvo. Padrão: ./extended_video.mp4",
                        "default": "./extended_video.mp4",
                    },
                    "model_variant": {
                        "type": "string",
                        "enum": [
                            "veo-3.1-generate-preview",
                            "veo-3.1-fast-generate-preview",
                        ],
                        "default": "veo-3.1-generate-preview",
                    },
                    "gcs_bucket_path": {
                        "type": "string",
                        "description": (
                            "Caminho do bucket GCS onde o vídeo estendido será salvo (opcional). "
                            "Formatos aceitos: 'meu-bucket', 'meu-bucket/pasta/', 'gs://meu-bucket/pasta/video.mp4'. "
                            "Requer GOOGLE_APPLICATION_CREDENTIALS configurado."
                        ),
                    },
                },
                "required": ["prompt", "video_path"],
            },
        ),
    ]


# ──────────────────────────────────────────────
# Handlers das ferramentas
# ──────────────────────────────────────────────

def _load_image_as_part(image_path: str) -> types.Part:
    """Carrega uma imagem do disco e retorna como types.Part."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    ext = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")
    
    with open(path, "rb") as f:
        image_bytes = f.read()
    
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def _poll_operation(client: genai.Client, operation) -> object:
    """Aguarda a conclusão de uma operação de geração de vídeo."""
    print("Aguardando geração do vídeo (pode levar 2-5 minutos)...")
    poll_count = 0
    while not operation.done:
        time.sleep(15)
        poll_count += 1
        print(f"  [{poll_count * 15}s] Ainda gerando...")
        operation = client.operations.get(operation)
    return operation


def _upload_to_gcs(local_path: str, gcs_bucket_path: str) -> str:
    """
    Faz upload de um arquivo local para o Google Cloud Storage.

    Args:
        local_path:      Caminho local do arquivo (ex: /tmp/video.mp4)
        gcs_bucket_path: Destino no GCS. Aceita dois formatos:
                         - Apenas bucket:   "meu-bucket"
                         - Bucket + pasta:  "meu-bucket/pasta/" ou "meu-bucket/pasta/nome.mp4"
                         - Com prefixo gs:  "gs://meu-bucket/pasta/"

    Returns:
        URI completa no GCS: "gs://meu-bucket/pasta/video.mp4"
    """
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        raise ImportError(
            "Pacote google-cloud-storage não instalado. "
            "Execute: pip install google-cloud-storage"
        )

    path_clean = gcs_bucket_path.removeprefix("gs://")

    if "/" in path_clean:
        bucket_name, blob_prefix = path_clean.split("/", 1)
        if blob_prefix.endswith("/") or "." not in Path(blob_prefix).name:
            blob_name = blob_prefix.rstrip("/") + "/" + Path(local_path).name
        else:
            blob_name = blob_prefix
    else:
        bucket_name = path_clean
        blob_name = Path(local_path).name

    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Upload concluído: {gcs_uri}")
    return gcs_uri


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    
    # ── generate_video ────────────────────────
    if name == "generate_video":
        prompt             = arguments["prompt"]
        negative_prompt    = arguments.get("negative_prompt")
        aspect_ratio       = arguments.get("aspect_ratio", "16:9")
        resolution         = arguments.get("resolution", "720p")
        duration_seconds   = arguments.get("duration_seconds", 8)
        output_path        = arguments.get("output_path", "./output_video.mp4")
        model_variant      = arguments.get("model_variant", "veo-3.1-generate-preview")
        first_frame_path   = arguments.get("first_frame_image_path")
        last_frame_path    = arguments.get("last_frame_image_path")
        ref_image_paths    = arguments.get("reference_image_paths", [])
        seed               = arguments.get("seed")
        gcs_bucket_path    = arguments.get("gcs_bucket_path")

        try:
            client = get_client()

            # Monta a config
            config_kwargs: dict = {
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "number_of_videos": 1,
                "duration_seconds": duration_seconds,
            }
            if negative_prompt:
                config_kwargs["negative_prompt"] = negative_prompt
            if seed is not None:
                config_kwargs["seed"] = seed

            # Imagens de referência (até 3)
            if ref_image_paths:
                config_kwargs["reference_images"] = [
                    types.ReferenceImage(reference_image=_load_image_as_part(p))
                    for p in ref_image_paths[:3]
                ]

            config = types.GenerateVideosConfig(**config_kwargs)

            # Frames específicos
            generate_kwargs: dict = {
                "model": model_variant,
                "prompt": prompt,
                "config": config,
            }
            if first_frame_path:
                generate_kwargs["image"] = _load_image_as_part(first_frame_path)
            if last_frame_path:
                generate_kwargs["last_frame"] = _load_image_as_part(last_frame_path)

            # Inicia a geração (operação assíncrona na API)
            operation = client.models.generate_videos(**generate_kwargs)
            operation = _poll_operation(client, operation)

            # Salva o vídeo localmente
            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(output_path)

            # Upload para GCS (se solicitado)
            gcs_uri = None
            if gcs_bucket_path:
                gcs_uri = _upload_to_gcs(output_path, gcs_bucket_path)

            result = (
                f"✅ Vídeo gerado com sucesso!\n"
                f"📁 Salvo localmente em: {output_path}\n"
                f"🎬 Modelo: {model_variant}\n"
                f"📐 Resolução: {resolution} | Proporção: {aspect_ratio} | Duração: {duration_seconds}s"
            )
            if gcs_uri:
                result += f"\n☁️ URI no GCS: {gcs_uri}"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Erro ao gerar vídeo: {str(e)}")]

    # ── extend_video ──────────────────────────
    elif name == "extend_video":
        prompt        = arguments["prompt"]
        video_path    = arguments["video_path"]
        output_path   = arguments.get("output_path", "./extended_video.mp4")
        model_variant = arguments.get("model_variant", "veo-3.1-generate-preview")
        gcs_bucket_path = arguments.get("gcs_bucket_path")

        try:
            client = get_client()

            video_file = Path(video_path)
            if not video_file.exists():
                return [TextContent(type="text", text=f"❌ Vídeo não encontrado: {video_path}")]

            with open(video_file, "rb") as f:
                video_bytes = f.read()

            video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")

            operation = client.models.generate_videos(
                model=model_variant,
                prompt=prompt,
                video=video_part,
            )
            operation = _poll_operation(client, operation)

            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(output_path)

            # Upload para GCS (se solicitado)
            gcs_uri = None
            if gcs_bucket_path:
                gcs_uri = _upload_to_gcs(output_path, gcs_bucket_path)

            result = (
                f"✅ Vídeo estendido com sucesso!\n"
                f"📁 Salvo localmente em: {output_path}\n"
                f"🎬 Modelo: {model_variant}"
            )
            if gcs_uri:
                result += f"\n☁️ URI no GCS: {gcs_uri}"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Erro ao estender vídeo: {str(e)}")]

    return [TextContent(type="text", text=f"❌ Ferramenta desconhecida: {name}")]


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())