#!/usr/bin/env python3
"""
Diagnostica a estrutura do PAPILA para entender
como parear metadados e imagens.

Uso:
    python scripts/diagnosticar_papila.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from glaucpio.config import BASE_DIR
from glaucpio.logger import get_logger

log = get_logger("glaucpio.diagnostico")


def main():
    papila_dir = BASE_DIR / "data" / "external" / "PAPILA"

    log.info("=" * 60)
    log.info("DIAGNÓSTICO PAPILA")
    log.info("=" * 60)

    # 1. Procurar Excels
    od_xlsx = list(papila_dir.rglob("patient_data_od.xlsx"))
    if not od_xlsx:
        log.error("patient_data_od.xlsx não encontrado.")
        sys.exit(1)
    od_xlsx = od_xlsx[0]
    log.info(f"Excel: {od_xlsx}")

    # 2. Ler COM header=0 e SEM header
    log.info("-" * 60)
    log.info("Lendo COM header=0:")
    df0 = pd.read_excel(od_xlsx, header=0)
    log.info(f"  Shape: {df0.shape}")
    log.info(f"  Colunas: {df0.columns.tolist()}")
    log.info(f"  Primeiras 3 linhas:")
    for i, row in df0.head(3).iterrows():
        log.info(f"    Linha {i}: {row.tolist()}")

    log.info("-" * 60)
    log.info("Lendo SEM header (header=None):")
    df_none = pd.read_excel(od_xlsx, header=None)
    log.info(f"  Shape: {df_none.shape}")
    log.info(f"  Primeiras 4 linhas:")
    for i, row in df_none.head(4).iterrows():
        log.info(f"    Linha {i}: {row.tolist()}")

    # 3. Procurar por IDs de paciente (RET*)
    log.info("-" * 60)
    log.info("Procurando por IDs no formato RET*:")
    for col in df_none.columns:
        valores = df_none[col].astype(str).tolist()
        rets = [v for v in valores if v.upper().startswith("RET")]
        if rets:
            log.info(f"  Coluna {col}: {len(rets)} IDs RET* encontrados")
            log.info(f"    Exemplos: {rets[:5]}")

    # 4. Listar imagens
    log.info("-" * 60)
    log.info("Imagens em FundusImages:")
    fundus_dirs = list(papila_dir.rglob("FundusImages"))
    if fundus_dirs:
        fundus_dir = fundus_dirs[0]
        imagens = sorted([p.name for p in fundus_dir.glob("*.jpg")])
        log.info(f"  Total: {len(imagens)}")
        log.info(f"  Primeiras 10: {imagens[:10]}")
        log.info(f"  Últimas 5: {imagens[-5:]}")

    log.info("=" * 60)
    log.info("✅ Diagnóstico concluído.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()