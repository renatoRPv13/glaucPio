#!/usr/bin/env python3
"""
Pré-processamento do dataset multimodal do Projeto GlaucPIO.

Lê data/raw/, limpa, enriquece e salva em data/processed/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from glaucpio.config import RAW_DIR, PROCESSED_DIR
from glaucpio.logger import get_logger

log = get_logger("glaucpio.preprocessamento")


def limpar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e normaliza colunas."""
    # Normalizar nomes de colunas
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Remover linhas sem Case_ID ou Ground_Truth
    df = df.dropna(subset=["case_id", "ground_truth_diagnosis"]).copy()

    # Normalizar texto
    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["ground_truth_diagnosis"] = (
        df["ground_truth_diagnosis"].astype(str).str.lower().str.strip()
    )

    return df.reset_index(drop=True)


def enriquecer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas."""
    # Presença de exames
    for exame, col in [
        ("oct", "oct_image_path"),
        ("fundus", "fundus_image_path"),
        ("vf", "vf_image_path"),
    ]:
        if col in df.columns:
            df[f"tem_{exame}"] = df[col].notna()

    # Contagem de exames
    cols_exame = [c for c in ["tem_oct", "tem_fundus", "tem_vf"] if c in df.columns]
    if cols_exame:
        df["n_exames"] = df[cols_exame].sum(axis=1)

    # Gravidade codificada
    if "gravidade" in df.columns:
        mapa = {"leve": 1, "moderado": 2, "avancado": 3, "avançado": 3}
        df["gravidade_num"] = df["gravidade"].str.lower().map(mapa)

    return df


def estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """Gera resumo estatístico do dataset."""
    stats = {
        "n_casos": len(df),
        "n_colunas": len(df.columns),
        "n_classes": df["ground_truth_diagnosis"].nunique(),
        "media_exames_por_caso": df.get("n_exames", pd.Series([0])).mean(),
    }

    # Distribuição de classes
    dist = df["ground_truth_diagnosis"].value_counts()
    for classe, count in dist.items():
        stats[f"classe_{classe}"] = count

    return pd.DataFrame([stats])


def main():
    log.info("=" * 60)
    log.info("PRÉ-PROCESSAMENTO — Projeto GlaucPIO")
    log.info("=" * 60)

    # Localizar dataset bruto
    raw_csv = RAW_DIR / "glaucoma_multimodal_500_dataset.csv"
    if not raw_csv.exists():
        # Tentar na raiz do projeto
        raw_csv = Path(__file__).resolve().parent.parent / "glaucoma_multimodal_500_dataset.csv"

    if not raw_csv.exists():
        log.error(f"Dataset bruto não encontrado: {raw_csv}")
        sys.exit(1)

    log.info(f"Lendo: {raw_csv}")
    df = pd.read_csv(raw_csv)
    log.info(f"Shape original: {df.shape}")

    # Limpeza
    df = limpar_dataset(df)
    log.info(f"Após limpeza: {df.shape}")

    # Enriquecimento
    df = enriquecer_dataset(df)
    log.info(f"Após enriquecimento: {df.shape}")

    # Salvar dataset limpo
    out_limpo = PROCESSED_DIR / "dataset_limpo.csv"
    df.to_csv(out_limpo, index=False)
    log.info(f"→ {out_limpo}")

    # Salvar com features
    out_features = PROCESSED_DIR / "dataset_com_features.csv"
    df.to_csv(out_features, index=False)
    log.info(f"→ {out_features}")

    # Estatísticas
    stats = estatisticas_descritivas(df)
    out_stats = PROCESSED_DIR / "estatisticas_descritivas.csv"
    stats.to_csv(out_stats, index=False)
    log.info(f"→ {out_stats}")

    log.info("=" * 60)
    log.info("✅ Pré-processamento concluído.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()