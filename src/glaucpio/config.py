"""
Configurações centrais do Projeto GlaucPIO..
"""
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "glaucpio"

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CAMINHOS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"
CSV_DIR = OUTPUT_DIR / "csv"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, CSV_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# API KEYS
# ============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ============================================================
# MODELOS
# ============================================================
MODELOS = {
    "gemini": {"nome": "gemini-3.6-flash", "provider": "google"},
    "gpt":    {"nome": "gpt-5",             "provider": "openai"},
    "claude": {"nome": "claude-sonnet-4-5", "provider": "anthropic"},
}

# ============================================================
# PARÂMETROS DE AVALIAÇÃO
# ============================================================
TEMPERATURE = 0.0
SLEEP_BETWEEN_CALLS = 4.0
MAX_RETRIES = 6
CHECKPOINT_EVERY = 10
MAX_CASES = None

# ============================================================
# PARÂMETROS ESTATÍSTICOS
# ============================================================
ALPHA = 0.05
MARGEM_NAO_INFERIORIDADE = 0.05
PODER = 0.80

# ============================================================
# DATASETS
# ============================================================
DATASET_MULTIMODAL = RAW_DIR / "glaucoma_multimodal_500_dataset.csv"
DATASET_GOBIRA = RAW_DIR / "gobira_2026_dados.csv"

OUTPUT_AVALIACAO_CSV = CSV_DIR / "gemini_3_6_flash_eval_results.csv"
OUTPUT_AVALIACAO_JSONL = CSV_DIR / "gemini_3_6_flash_eval_raw.jsonl"


def caminhos_saida(nome_modelo: str) -> tuple[Path, Path]:
    """Retorna caminhos de saída específicos para o identificador do modelo."""
    slug = re.sub(r"[^a-z0-9]+", "_", nome_modelo.lower()).strip("_")
    return (
        CSV_DIR / f"{slug}_eval_results.csv",
        CSV_DIR / f"{slug}_eval_raw.jsonl",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG — GlaucPIO")
    print("=" * 60)
    print(f"BASE_DIR:      {BASE_DIR}")
    print(f"CSV_DIR:       {CSV_DIR}")
    print(f"FIGURES_DIR:   {FIGURES_DIR}")
    print(f"LOGS_DIR:      {LOGS_DIR}")
    print(f"GOOGLE_API_KEY presente: {GOOGLE_API_KEY is not None}")
    print(f"OPENAI_API_KEY presente: {OPENAI_API_KEY is not None}")
    print(f"ANTHROPIC_API_KEY presente: {ANTHROPIC_API_KEY is not None}")

# """
# Configurações centrais do Projeto GlaucPIO.
# """
# import os
# import re
# from pathlib import Path
# from dotenv import load_dotenv
#
# load_dotenv()
#
# # ============================================================
# # CAMINHOS
# # ============================================================
# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# DATA_DIR = BASE_DIR / "data"
# RAW_DIR = DATA_DIR / "raw"
# PROCESSED_DIR = DATA_DIR / "processed"
# DATASET_DIR = BASE_DIR / "dataset"
# IMAGES_DIR = DATASET_DIR / "images"
# OUTPUT_DIR = BASE_DIR / "outputs"
# CSV_DIR = OUTPUT_DIR / "csv"
# FIGURES_DIR = OUTPUT_DIR / "figures"
# LOGS_DIR = OUTPUT_DIR / "logs"
#
# for d in [RAW_DIR, PROCESSED_DIR, CSV_DIR, FIGURES_DIR, LOGS_DIR]:
#     d.mkdir(parents=True, exist_ok=True)
#
# # ============================================================
# # API KEYS
# # ============================================================
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
#
# # ============================================================
# # MODELOS
# # ============================================================
# MODELOS = {
#     "gemini": {
#         "nome": "gemini-3.6-flash",
#         "provider": "google",
#     },
#     "gpt": {
#         "nome": "gpt-5",
#         "provider": "openai",
#     },
#     "claude": {
#         "nome": "claude-sonnet-4-5",
#         "provider": "anthropic",
#     },
#     "llama": {
#         "nome": "llama-3.3-70b",
#         "provider": "meta",
#     },
# }
#
# # ============================================================
# # PARÂMETROS DE AVALIAÇÃO
# # ============================================================
# TEMPERATURE = 0.0
# SLEEP_BETWEEN_CALLS = 4.0        # segundos (evita 429)
# MAX_RETRIES = 6
# CHECKPOINT_EVERY = 10
# MAX_CASES = None                 # None = todos os 500
#
# # ============================================================
# # PARÂMETROS ESTATÍSTICOS
# # ============================================================
# ALPHA = 0.05
# MARGEM_NAO_INFERIORIDADE = 0.05
# PODER = 0.80
#
# # ============================================================
# # DATASETS
# # ============================================================
# DATASET_MULTIMODAL = RAW_DIR / "glaucoma_multimodal_500_dataset.csv"
# DATASET_GOBIRA = RAW_DIR / "gobira_2026_dados.csv"
#
#
#
# def caminhos_saida(nome_modelo: str) -> tuple:
#     """
#     Caminhos de saída da avaliação, um par por modelo.
#
#     Ex.: "gemini-3.6-flash" -> gemini_3_6_flash_eval_results.csv / _eval_raw.jsonl
#     """
#     slug = re.sub(r"[^a-z0-9]+", "_", nome_modelo.lower()).strip("_")
#     return (
#         CSV_DIR / f"{slug}_eval_results.csv",
#         CSV_DIR / f"{slug}_eval_raw.jsonl",
#     )