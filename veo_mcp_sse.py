"""
Servidor MCP para geração de vídeos com Veo 3.1 via Google AI Studio (Gemini API)
com transporte SSE (HTTP) para deploy remoto (GKE).

Versão ASGI Pura (Async): Máxima estabilidade para conexões SSE e Kubernetes.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool

from google import genai
from google.genai import types

load_dotenv()

# ──────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────

PORT = int(os.getenv("PORT", "8080"))

# Timeout máximo de polling: 30 minutos (120 polls × 15s)
MAX_POLL_COUNT = 120

app_mcp = Server("veo-video-generator")


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Variável de ambiente GEMINI_API_KEY não definida. "
            "Obtenha sua chave em https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


# ──────────────────────────────────────────────
# Utilitários Assíncronos
# ──────────────────────────────────────────────

async def _load_image_resource(image_path: str) -> types.Part:
    """Carrega uma imagem de disco local ou gs:// URI e retorna como types.Part."""
    if image_path.startswith("gs://"):
        ext = Path(image_path).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        return types.Part.from_uri(file_uri=image_path, mime_type=mime_map.get(ext, "image/jpeg"))

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    ext = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    def sync_read():
        with open(path, "rb") as f:
            return f.read()

    data = await asyncio.get_event_loop().run_in_executor(None, sync_read)
    return types.Part.from_bytes(data=data, mime_type=mime_type)


async def _upload_to_gcs_async(local_path: str, gcs_bucket_path: str) -> str:
    """Faz upload de um arquivo local para o GCS de forma assíncrona."""
    from google.cloud import storage

    def sync_upload():
        path_clean = gcs_bucket_path.removeprefix("gs://")
        if "/" in path_clean:
            parts = path_clean.split("/", 1)
            bucket_name = parts[0]
            blob_prefix = parts[1]
            # Se o prefixo termina com / ou não tem extensão, tratamos como pasta
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

    return await asyncio.get_event_loop().run_in_executor(None, sync_upload)


async def _poll_operation_async(client: genai.Client, operation) -> object:
    """
    Aguarda a conclusão de uma operação de geração de vídeo de forma assíncrona.
    Timeout máximo: MAX_POLL_COUNT × 15s (padrão: 30 minutos).
    """
    print("Aguardando geração do vídeo (pode levar 2-10 minutos)...")
    poll_count = 0

    while not operation.done:
        if poll_count >= MAX_POLL_COUNT:
            raise TimeoutError(
                f"Timeout: a geração do vídeo excedeu {MAX_POLL_COUNT * 15 // 60} minutos. "
                "Tente novamente ou use um modelo mais rápido (fast/lite)."
            )
        await asyncio.sleep(15)  # Não bloqueia o event loop
        poll_count += 1
        elapsed = poll_count * 15
        print(f"  [{elapsed}s / {MAX_POLL_COUNT * 15}s] Ainda gerando...")
        # Atualiza o status da operação (síncrono, mas rápido — mantido em executor)
        operation = await asyncio.get_event_loop().run_in_executor(
            None, lambda op=operation: client.operations.get(op)
        )

    return operation


# ──────────────────────────────────────────────
# Schema de Ferramentas
# ──────────────────────────────────────────────

@app_mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_video",
            description=(
                "Gera um vídeo usando o modelo Veo 3.1 do Google. "
                "Suporta geração por texto, por imagem inicial/final, ou por imagens de referência. "
                "Retorna o caminho do arquivo .mp4 salvo localmente e, opcionalmente, o URI no GCS."
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
                        "description": "Resolução do vídeo. Padrão: 1080p.",
                        "default": "1080p",
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
                        "description": "Caminho local ou gs:// URI para a imagem do primeiro frame (opcional).",
                    },
                    "last_frame_image_path": {
                        "type": "string",
                        "description": "Caminho local ou gs:// URI para a imagem do último frame (opcional).",
                    },
                    "reference_image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de até 3 caminhos (local ou gs://) de imagens de referência (opcional).",
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
                        "description": "Caminho local para o arquivo .mp4 a ser estendido.",
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
# Handlers das Ferramentas
# ──────────────────────────────────────────────

@app_mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    print(f"DEBUG: Chamando ferramenta '{name}' com argumentos {arguments}")
    client = get_client()

    try:
        # ── generate_video ────────────────────────────────────────────────────
        if name == "generate_video":
            prompt           = arguments["prompt"]
            negative_prompt  = arguments.get("negative_prompt")
            aspect_ratio     = arguments.get("aspect_ratio", "16:9")
            resolution       = arguments.get("resolution", "720p")
            duration_seconds = arguments.get("duration_seconds", 8)
            output_path      = arguments.get("output_path", "./output_video.mp4")
            model_variant    = arguments.get("model_variant", "veo-3.1-generate-preview")
            first_frame_path = arguments.get("first_frame_image_path")
            last_frame_path  = arguments.get("last_frame_image_path")
            ref_image_paths  = arguments.get("reference_image_paths", [])
            seed             = arguments.get("seed")
            gcs_bucket_path  = arguments.get("gcs_bucket_path")

            # Monta a config da geração
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

            # Imagens de referência (até 3) — carregadas de forma assíncrona
            if ref_image_paths:
                ref_parts = [await _load_image_resource(p) for p in ref_image_paths[:3]]
                config_kwargs["reference_images"] = [
                    types.ReferenceImage(reference_image=part) for part in ref_parts
                ]

            config = types.GenerateVideosConfig(**config_kwargs)

            # Kwargs da chamada de geração
            generate_kwargs: dict = {
                "model": model_variant,
                "prompt": prompt,
                "config": config,
            }
            if first_frame_path:
                generate_kwargs["image"] = await _load_image_resource(first_frame_path)
            if last_frame_path:
                generate_kwargs["last_frame"] = await _load_image_resource(last_frame_path)

            # Inicia a geração em executor (chamada síncrona/bloqueante)
            print(f"DEBUG: Iniciando geração com modelo '{model_variant}'...")
            operation = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.models.generate_videos(**generate_kwargs)
            )

            # Polling assíncrono — não bloqueia o event loop
            operation = await _poll_operation_async(client, operation)

            # Download e salvamento em executor (síncrono)
            generated_video = operation.response.generated_videos[0]
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            def sync_download_and_save():
                client.files.download(file=generated_video.video)
                generated_video.video.save(str(output_file))

            await asyncio.get_event_loop().run_in_executor(None, sync_download_and_save)

            # Upload para GCS (se solicitado)
            gcs_uri = None
            if gcs_bucket_path:
                gcs_uri = await _upload_to_gcs_async(str(output_file), gcs_bucket_path)

            result = (
                f"✅ Vídeo gerado com sucesso!\n"
                f"📁 Salvo localmente em: {output_path}\n"
                f"🎬 Modelo: {model_variant}\n"
                f"📐 Resolução: {resolution} | Proporção: {aspect_ratio} | Duração: {duration_seconds}s"
            )
            if gcs_uri:
                result += f"\n☁️ URI no GCS: {gcs_uri}"

            return [TextContent(type="text", text=result)]

        # ── extend_video ──────────────────────────────────────────────────────
        elif name == "extend_video":
            prompt          = arguments["prompt"]
            video_path      = arguments["video_path"]
            output_path     = arguments.get("output_path", "./extended_video.mp4")
            model_variant   = arguments.get("model_variant", "veo-3.1-generate-preview")
            gcs_bucket_path = arguments.get("gcs_bucket_path")

            video_file = Path(video_path)
            if not video_file.exists():
                return [TextContent(type="text", text=f"❌ Vídeo não encontrado: {video_path}")]

            # Leitura do vídeo em executor
            def sync_read_video():
                with open(video_file, "rb") as f:
                    return f.read()

            video_bytes = await asyncio.get_event_loop().run_in_executor(None, sync_read_video)
            video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")

            # Inicia extensão em executor
            print(f"DEBUG: Iniciando extensão com modelo '{model_variant}'...")
            operation = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_videos(
                    model=model_variant,
                    prompt=prompt,
                    video=video_part,
                ),
            )

            # Polling assíncrono
            operation = await _poll_operation_async(client, operation)

            # Download e salvamento em executor
            generated_video = operation.response.generated_videos[0]
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            def sync_download_and_save():
                client.files.download(file=generated_video.video)
                generated_video.video.save(str(output_file))

            await asyncio.get_event_loop().run_in_executor(None, sync_download_and_save)

            # Upload para GCS (se solicitado)
            gcs_uri = None
            if gcs_bucket_path:
                gcs_uri = await _upload_to_gcs_async(str(output_file), gcs_bucket_path)

            result = (
                f"✅ Vídeo estendido com sucesso!\n"
                f"📁 Salvo localmente em: {output_path}\n"
                f"🎬 Modelo: {model_variant}"
            )
            if gcs_uri:
                result += f"\n☁️ URI no GCS: {gcs_uri}"

            return [TextContent(type="text", text=result)]

        else:
            return [TextContent(type="text", text=f"❌ Ferramenta desconhecida: {name}")]

    except TimeoutError as e:
        print(f"TIMEOUT: {e}")
        return [TextContent(type="text", text=f"⏱️ Timeout: {str(e)}")]
    except Exception as e:
        print(f"ERRO: {e}")
        return [TextContent(type="text", text=f"❌ Erro: {str(e)}")]


# ──────────────────────────────────────────────
# Lógica do Servidor ASGI DIRETA
# ──────────────────────────────────────────────

sse = SseServerTransport("/messages")


async def app(scope, receive, send):
    """Aplicação ASGI Pura compatível com Uvicorn e GKE."""

    if scope["type"] == "http":
        # Trata requisições OPTIONS (CORS preflight)
        if scope["method"] == "OPTIONS":
            await send({
                "type": "http.response.start", "status": 204,
                "headers": [
                    (b"access-control-allow-origin", b"*"),
                    (b"access-control-allow-methods", b"*"),
                    (b"access-control-allow-headers", b"*"),
                ]
            })
            await send({"type": "http.response.body", "body": b""})
            return

        path = scope["path"]

        # Endpoint SSE — cliente conecta e mantém a stream aberta
        if path == "/sse":
            print("DEBUG: Iniciando conexão SSE...")
            async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await app_mcp.run(read_stream, write_stream, app_mcp.create_initialization_options())
            return

        # Endpoint de mensagens — cliente envia tools calls via POST
        elif path.startswith("/messages"):
            await sse.handle_post_message(scope, receive, send)
            return

        # Qualquer outro caminho — resposta 200 simples (health check)
        await send({
            "type": "http.response.start", "status": 200,
            "headers": [(b"content-type", b"text/plain")]
        })
        await send({"type": "http.response.body", "body": b"Veo MCP SSE Server Running."})


if __name__ == "__main__":
    import uvicorn
    print(f"🎬 Veo MCP SSE Server iniciando na porta {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=3600)
