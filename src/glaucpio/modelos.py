"""Wrappers unificados para chamadas a diferentes LVMs.

Interface comum:
    modelo = ModeloFactory.criar("gemini")
    resposta = modelo.chamar(prompt, imagens)
"""
from abc import ABC, abstractmethod
import base64
from io import BytesIO
import time
from typing import Any, Optional, cast
import  sys
from pathlib import Path
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

#Permite executar este arquivo diretamente, além de `python -m glaucpio.modelos`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "glaucpio"

from .config import (
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    TEMPERATURE,
    MAX_RETRIES,
)
from .logger import get_logger

log = get_logger(__name__)


def _is_google_rate_limit(exc: BaseException) -> bool:
    """Retenta apenas erros 429 do google-genai."""
    try:
        from google.genai import errors
    except ImportError:
        return False

    if isinstance(exc, errors.ClientError):
        message = str(exc).upper()
        return "429" in message or "RESOURCE_EXHAUSTED" in message
    return False


def _is_google_transient(exc: BaseException) -> bool:
    """Retenta erros 429 e erros 5xx do google-genai."""
    try:
        from google.genai import errors
    except ImportError:
        return False

    if isinstance(exc, errors.ClientError):
        message = str(exc).upper()
        return "429" in message or "RESOURCE_EXHAUSTED" in message
    return isinstance(exc, errors.ServerError)


class ModeloLVM(ABC):
    """Interface comum para todos os modelos de linguagem/visão."""

    nome: str

    @abstractmethod
    def chamar(self, prompt: str, imagens: Optional[list[Any]] = None) -> dict[str, Any]:
        """Envia prompt e imagens opcionais e retorna resposta e métricas."""
        raise NotImplementedError


class ModeloGemini(ModeloLVM):
    def __init__(self, nome: str = "gemini-3.6-flash"):
        from google import genai

        self.nome = nome
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        log.info("ModeloGemini inicializado: %s", nome)

    def chamar(self, prompt: str, imagens: Optional[list[Any]] = None) -> dict[str, Any]:
        from google.genai import types

        contents: list[Any] = [prompt]
        if imagens:
            contents.extend(imagens)

        @retry(
            retry=retry_if_exception(_is_google_transient),
            wait=wait_exponential(multiplier=5, min=10, max=120),
            stop=stop_after_attempt(MAX_RETRIES),
            reraise=True,
        )
        def _chamar() -> dict[str, Any]:
            t0 = time.time()
            response = self.client.models.generate_content(
                model=self.nome,
                contents=contents,
                config=types.GenerateContentConfig(temperature=TEMPERATURE),
            )
            usage = getattr(response, "usage_metadata", None)
            return {
                "text": response.text or "",
                "latency_s": round(time.time() - t0, 3),
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
            }

        return _chamar()


class ModeloOpenAI(ModeloLVM):
    def __init__(self, nome: str = "gpt-5"):
        from openai import OpenAI

        self.nome = nome
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        log.info("ModeloOpenAI inicializado: %s", nome)

    def chamar(self, prompt: str, imagens: Optional[list[Any]] = None) -> dict[str, Any]:
        # O cast limita a coerção ao limite do SDK: os blocos abaixo seguem o
        # formato Chat Completions para texto e imagens.
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if imagens:
            for img in imagens:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )

        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.nome,
            messages=cast(Any, [{"role": "user", "content": content}]),
            temperature=TEMPERATURE,
        )
        usage = response.usage
        return {
            "text": response.choices[0].message.content or "",
            "latency_s": round(time.time() - t0, 3),
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
        }


class ModeloAnthropic(ModeloLVM):
    def __init__(self, nome: str = "claude-sonnet-4-5"):
        from anthropic import Anthropic

        self.nome = nome
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        log.info("ModeloAnthropic inicializado: %s", nome)

    def chamar(self, prompt: str, imagens: Optional[list[Any]] = None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if imagens:
            for img in imagens:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    }
                )

        t0 = time.time()
        response = self.client.messages.create(
            model=self.nome,
            max_tokens=1024,
            temperature=TEMPERATURE,
            messages=cast(Any, [{"role": "user", "content": content}]),
        )
        # A resposta Anthropic pode conter outros blocos além de texto.
        text_blocks = [getattr(block, "text", "") for block in response.content]
        return {
            "text": "".join(text_blocks),
            "latency_s": round(time.time() - t0, 3),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }


class ModeloFactory:
    @staticmethod
    def criar(provider: str, nome: Optional[str] = None) -> ModeloLVM:
        providers = {
            "gemini": ModeloGemini,
            "gpt": ModeloOpenAI,
            "claude": ModeloAnthropic,
        }
        if provider not in providers:
            raise ValueError(f"Provider desconhecido: {provider}")
        model_class = providers[provider]
        return model_class(nome=nome) if nome else model_class()




# """
# Wrappers unificados para chamadas a diferentes LVMs.
#
# Interface comum:
#     modelo = ModeloFactory.criar("gemini")
#     resposta = modelo.chamar(prompt, imagens)
# """
# from abc import ABC, abstractmethod
# import base64
# from io import BytesIO
# from pathlib import Path
# from typing import Optional
# import time
# import sys
#
#
# # Permite executar este arquivo diretamente, além de `python -m glaucpio.modelos`.
# if __package__ in (None, ""):
#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#     __package__ = "glaucpio"
#
# from PIL import Image
# from tenacity import (
#     retry, stop_after_attempt, wait_exponential,
#     retry_base, RetryCallState  # <-- nome correto aqui
# )
#
# from .config import (
#     GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
#     TEMPERATURE, MAX_RETRIES,
# )
# from .logger import get_logger
#
# log = get_logger(__name__)
#
#
# # ============================================================
# # POLÍTICA DE RETRY
# # ============================================================
# class RetryIfRateLimit(retry_base):
#     """
#     Política de retry do tenacity: retenta apenas erros HTTP 429
#     (rate limit / RESOURCE_EXHAUSTED) do google-genai.
#
#     Erros 400, 401, 403, 404 são permanentes e não são retentados.
#     """
#
#     def __call__(self, retry_state: RetryCallState) -> bool:
#         if retry_state.outcome is None or not retry_state.outcome.failed:
#             return False
#
#         exc = retry_state.outcome.exception()
#         if exc is None:
#             return False
#
#         # Import local para evitar dependência circular
#         from google.genai import errors
#
#         if isinstance(exc, errors.ClientError):
#             msg = str(exc).upper()
#             return "429" in msg or "RESOURCE_EXHAUSTED" in msg
#
#         return False
#
#
# class RetryIfTransientError(retry_base):
#     """
#     Política mais permissiva: retenta 429 (rate limit) e 5xx (erros
#     transitórios do servidor). Útil para produção.
#     """
#
#     def __call__(self, retry_state: RetryCallState) -> bool:
#         if retry_state.outcome is None or not retry_state.outcome.failed:
#             return False
#
#         exc = retry_state.outcome.exception()
#         if exc is None:
#             return False
#
#         from google.genai import errors
#
#         if isinstance(exc, errors.ClientError):
#             msg = str(exc).upper()
#             return "429" in msg or "RESOURCE_EXHAUSTED" in msg
#
#         if isinstance(exc, errors.ServerError):
#             return True  # 5xx são transitórios
#
#         return False
#
#
# # ============================================================
# # INTERFACE ABSTRATA
# # ============================================================
# class ModeloLVM(ABC):
#     """Interface comum para todos os LVMs."""
#
#     nome: str  # identificador do modelo na API; usado também no nome dos arquivos de saída
#
#     @abstractmethod
#     def chamar(self, prompt: str, imagens: Optional[list] = None) -> dict:
#         """
#         Envia prompt (e imagens opcionais) e retorna resposta.
#
#         Returns:
#             dict com:
#             - text: str
#             - latency_s: float
#             - input_tokens: int | None
#             - output_tokens: int | None
#         """
#         pass
#
#
# # ============================================================
# # GEMINI
# # ============================================================
# class ModeloGemini(ModeloLVM):
#     def __init__(self, nome: str = "gemini-3.6-flash"):
#         from google import genai
#         self.nome = nome
#         self.client = genai.Client(api_key=GOOGLE_API_KEY)
#         log.info(f"ModeloGemini inicializado: {nome}")
#
#     def chamar(self, prompt: str, imagens: Optional[list] = None) -> dict:
#         from google.genai import types
#
#         contents = [prompt]
#         if imagens:
#             contents.extend(imagens)
#
#         @retry(
#             retry=RetryIfTransientError(),   # ou RetryIfRateLimit() para ser estrito
#             wait=wait_exponential(multiplier=5, min=10, max=120),
#             stop=stop_after_attempt(MAX_RETRIES),
#             reraise=True,
#         )
#         def _chamar():
#             t0 = time.time()
#             response = self.client.models.generate_content(
#                 model=self.nome,
#                 contents=contents,
#                 config=types.GenerateContentConfig(temperature=TEMPERATURE),
#             )
#             elapsed = time.time() - t0
#             usage = getattr(response, "usage_metadata", None)
#             return {
#                 "text": response.text if hasattr(response, "text") else "",
#                 "latency_s": round(elapsed, 3),
#                 "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
#                 "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
#             }
#
#         return _chamar()
#
#
# # ============================================================
# # OPENAI
# # ============================================================
# class ModeloOpenAI(ModeloLVM):
#     def __init__(self, nome: str = "gpt-5"):
#         from openai import OpenAI
#         self.nome = nome
#         self.client = OpenAI(api_key=OPENAI_API_KEY)
#         log.info(f"ModeloOpenAI inicializado: {nome}")
#
#     def chamar(self, prompt: str, imagens: Optional[list] = None) -> dict:
#         contents = [{"type": "text", "text": prompt}]
#         if imagens:
#             for img in imagens:
#                 if img.mode not in ("RGB", "L"):
#                     img = img.convert("RGB")
#                 buffered = BytesIO()
#                 img.save(buffered, format="PNG")
#                 b64 = base64.b64encode(buffered.getvalue()).decode()
#                 contents.append({
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{b64}"},
#                 })
#
#         t0 = time.time()
#         response = self.client.chat.completions.create(
#             model=self.nome,
#             messages=[{"role": "user", "content": contents}],
#             temperature=TEMPERATURE,
#         )
#         elapsed = time.time() - t0
#         usage = response.usage
#         return {
#             "text": response.choices[0].message.content,
#             "latency_s": round(elapsed, 3),
#             "input_tokens": usage.prompt_tokens if usage else None,
#             "output_tokens": usage.completion_tokens if usage else None,
#         }
#
#
# # ============================================================
# # ANTHROPIC
# # ============================================================
# class ModeloAnthropic(ModeloLVM):
#     def __init__(self, nome: str = "claude-sonnet-4-5"):
#         from anthropic import Anthropic
#         self.nome = nome
#         self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
#         log.info(f"ModeloAnthropic inicializado: {nome}")
#
#     def chamar(self, prompt: str, imagens: Optional[list] = None) -> dict:
#         contents = [{"type": "text", "text": prompt}]
#         if imagens:
#             for img in imagens:
#                 if img.mode not in ("RGB", "L"):
#                     img = img.convert("RGB")
#                 buffered = BytesIO()
#                 img.save(buffered, format="PNG")
#                 b64 = base64.b64encode(buffered.getvalue()).decode()
#                 contents.append({
#                     "type": "image",
#                     "source": {
#                         "type": "base64",
#                         "media_type": "image/png",
#                         "data": b64,
#                     },
#                 })
#
#         t0 = time.time()
#         response = self.client.messages.create(
#             model=self.nome,
#             max_tokens=1024,
#             temperature=TEMPERATURE,
#             messages=[{"role": "user", "content": contents}],
#         )
#         elapsed = time.time() - t0
#         return {
#             "text": response.content[0].text,
#             "latency_s": round(elapsed, 3),
#             "input_tokens": response.usage.input_tokens,
#             "output_tokens": response.usage.output_tokens,
#         }
#
#
# # ============================================================
# # FACTORY
# # ============================================================
# class ModeloFactory:
#     @staticmethod
#     def criar(provider: str, nome: Optional[str] = None) -> ModeloLVM:
#         providers = {
#             "gemini": ModeloGemini,
#             "gpt": ModeloOpenAI,
#             "claude": ModeloAnthropic,
#         }
#         if provider not in providers:
#             raise ValueError(f"Provider desconhecido: {provider}")
#         if nome:
#             return providers[provider](nome=nome)
#         return providers[provider]()



# """
# Wrappers unificados para chamadas a diferentes LVMs.
#
# Interface comum:
#     modelo = ModeloFactory.criar("gemini")
#     resposta = modelo.chamar(prompt, imagens)
# """
# from abc import ABC, abstractmethod
# import base64
# from io import BytesIO
# from pathlib import Path
# from typing import Optional
# import time
# import sys
#
# # Permite executar este arquivo diretamente, além de `python -m glaucpio.avaliacao`.
# if __package__ in (None, ""):
#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#     __package__ = "glaucpio"
# from PIL import Image
# from tenacity import (
#     retry, stop_after_attempt, wait_exponential,
#     RetryCallState, RetryError,
# )
#
# from .config import (
#     GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
#     TEMPERATURE, MAX_RETRIES,
# )
# from .logger import get_logger
#
# log = get_logger(__name__)
#
#
# # ============================================================
# # INTERFACE ABSTRATA
# # ============================================================
# class ModeloLVM(ABC):
#     """Interface comum para todos os LVMs."""
#
#     nome: str  # identificador do modelo na API; usado também no nome dos arquivos de saída
#
#     @abstractmethod
#     def chamar(self, prompt: str, imagens: Optional[list] = None) -> dict:
#         """
#         Envia prompt (e imagens opcionais) e retorna resposta.
#
#         Returns:
#             dict com:
#             - text: str
#             - latency_s: float
#             - input_tokens: int | None
#             - output_tokens: int | None
#         """
#         pass
#
#
# # ============================================================
# # GEMINI
# # ============================================================
# class ModeloGemini(ModeloLVM):
#     def __init__(self, nome: str = "gemini-3.6-flash"):
#         from google import genai
#         self.nome = nome
#         self.client = genai.Client(api_key=GOOGLE_API_KEY)
#         log.info(f"ModeloGemini inicializado: {nome}")
#
#     def chamar(self, prompt: str, imagens: Optional[list] = None) -> dict:
#         from google import genai
#         from google.genai import types
#         from google.genai import errors
#
#         contents = [prompt]
#         if imagens:
#             contents.extend(imagens)
#
#         def is_rate_limit(exc: BaseException) -> bool:
#             if isinstance(exc, errors.ClientError):
#                 msg = str(exc)
#                 return "429" in msg or "RESOURCE_EXHAUSTED" in msg
#             return False
#
#         def should_retry(retry_state: RetryCallState) -> bool:
#             if retry_state.outcome is None or not retry_state.outcome.failed:
#                 return False
#             exc = retry_state.outcome.exception()
#             return exc is not None and is_rate_limit(exc)
#
#         @retry(
#             retry=should_retry,
#             wait=wait_exponential(multiplier=5, min=10, max=120),
#             stop=stop_after_attempt(MAX_RETRIES),
#             reraise=True,
#         )
#         def _chamar():
#             t0 = time.time()
#             response = self.client.models.generate_content(
#                 model=self.nome,
#                 contents=contents,
#                 config=types.GenerateContentConfig(temperature=TEMPERATURE),
#             )
#             elapsed = time.time() - t0
#             usage = getattr(response, "usage_metadata", None)
#             return {
#                 "text": response.text if hasattr(response, "text") else "",
#                 "latency_s": round(elapsed, 3),
#                 "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
#                 "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
#             }
#
#         return _chamar()
#
#
# # ============================================================
# # OPENAI
# # ============================================================
# class ModeloOpenAI(ModeloLVM):
#     def __init__(self, nome: str = "gpt-5"):
#         from openai import OpenAI
#         self.nome = nome
#         self.client = OpenAI(api_key=OPENAI_API_KEY)
#         log.info(f"ModeloOpenAI inicializado: {nome}")
#
#     def chamar(self, prompt: str, imagens: Optional[list] = None) -> dict:
#         contents = [{"type": "text", "text": prompt}]
#         if imagens:
#             for img in imagens:
#                 if img.mode not in ("RGB", "L"):
#                     img = img.convert("RGB")
#                 buffered = BytesIO()
#                 img.save(buffered, format="PNG")
#                 b64 = base64.b64encode(buffered.getvalue()).decode()
#                 contents.append({
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{b64}"},
#                 })
#
#         t0 = time.time()
#         response = self.client.chat.completions.create(
#             model=self.nome,
#             messages=[{"role": "user", "content": contents}],
#             temperature=TEMPERATURE,
#         )
#         elapsed = time.time() - t0
#         usage = response.usage
#         return {
#             "text": response.choices[0].message.content,
#             "latency_s": round(elapsed, 3),
#             "input_tokens": usage.prompt_tokens if usage else None,
#             "output_tokens": usage.completion_tokens if usage else None,
#         }
#
#
# # ============================================================
# # ANTHROPIC
# # ============================================================
# class ModeloAnthropic(ModeloLVM):
#     def __init__(self, nome: str = "claude-sonnet-4-5"):
#         from anthropic import Anthropic
#         self.nome = nome
#         self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
#         log.info(f"ModeloAnthropic inicializado: {nome}")
#
#     def chamar(self, prompt: str, imagens: Optional[list] = None) -> dict:
#         contents = [{"type": "text", "text": prompt}]
#         if imagens:
#             for img in imagens:
#                 if img.mode not in ("RGB", "L"):
#                     img = img.convert("RGB")
#                 buffered = BytesIO()
#                 img.save(buffered, format="PNG")
#                 b64 = base64.b64encode(buffered.getvalue()).decode()
#                 contents.append({
#                     "type": "image",
#                     "source": {
#                         "type": "base64",
#                         "media_type": "image/png",
#                         "data": b64,
#                     },
#                 })
#
#         t0 = time.time()
#         response = self.client.messages.create(
#             model=self.nome,
#             max_tokens=1024,
#             temperature=TEMPERATURE,
#             messages=[{"role": "user", "content": contents}],
#         )
#         elapsed = time.time() - t0
#         return {
#             "text": response.content[0].text,
#             "latency_s": round(elapsed, 3),
#             "input_tokens": response.usage.input_tokens,
#             "output_tokens": response.usage.output_tokens,
#         }
#
#
# # ============================================================
# # FACTORY
# # ============================================================
# class ModeloFactory:
#     @staticmethod
#     def criar(provider: str, nome: Optional[str] = None) -> ModeloLVM:
#         providers = {
#             "gemini": ModeloGemini,
#             "gpt": ModeloOpenAI,
#             "claude": ModeloAnthropic,
#         }
#         if provider not in providers:
#             raise ValueError(f"Provider desconhecido: {provider}")
#         if nome:
#             return providers[provider](nome=nome)
#         return providers[provider]()