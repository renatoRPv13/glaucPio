"""
Logging estruturado para o Projeto GlaucPIO.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "glaucpio"

import logging
from .config import LOGS_DIR


def get_logger(nome: str = "glaucpio") -> logging.Logger:
    logger = logging.getLogger(nome)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    fh = logging.FileHandler(LOGS_DIR / "glaucpio.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    for lib in ["google.genai", "httpx", "google.api_core", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logger


if __name__ == "__main__":
    log = get_logger("test")
    log.info("Logger inicializado com sucesso.")
    log.warning("Exemplo de warning.")
    log.error("Exemplo de erro.")


# """
# Logging estruturado do Projeto GlaucPIO.
#
# Uso:
#     from .logger import get_logger
#     log = get_logger(__name__)
# """
# import logging
# import sys
# from  pathlib import Path
# # Permite executar este arquivo diretamente, além de `python -m glaucpio.avaliacao`.
# if __package__ in (None, ""):
#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#     __package__ = "glaucpio"
# from .config import LOGS_DIR
#
# LOGGER_RAIZ = "glaucpio"
# LOG_FILE = LOGS_DIR / "glaucpio.log"
# FORMATO = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
#
# # Bibliotecas de terceiros muito verbosas
# BIBLIOTECAS_SILENCIADAS = [
#     "google.genai", "google.api_core", "httpx", "httpcore",
#     "openai", "anthropic", "urllib3", "PIL",
# ]
#
#
# def _configurar_raiz() -> logging.Logger:
#     """Configura (uma única vez) o logger raiz do pacote."""
#     raiz = logging.getLogger(LOGGER_RAIZ)
#     if raiz.handlers:
#         return raiz
#
#     raiz.setLevel(logging.INFO)
#     formatter = logging.Formatter(FORMATO)
#
#     file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
#     file_handler.setFormatter(formatter)
#     raiz.addHandler(file_handler)
#
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(formatter)
#     raiz.addHandler(stream_handler)
#
#     # Evita duplicar mensagens caso o root logger também esteja configurado
#     raiz.propagate = False
#
#     for lib in BIBLIOTECAS_SILENCIADAS:
#         logging.getLogger(lib).setLevel(logging.WARNING)
#
#     return raiz
#
#
# def get_logger(name: str) -> logging.Logger:
#     """
#     Retorna um logger filho de 'glaucpio'.
#
#     Args:
#         name: normalmente __name__ do módulo (ex.: "glaucpio.avaliacao")
#     """
#     _configurar_raiz()
#     if not name.startswith(LOGGER_RAIZ):
#         name = f"{LOGGER_RAIZ}.{name}"
#     return logging.getLogger(name)
