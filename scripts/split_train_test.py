#!/usr/bin/env python3
"""
Split estratificado treino/teste do dataset do Projeto GlaucPIO.

Uso:
    python scripts/split_train_test.py
    python scripts/split_train_test.py --test-size 0.3 --seed 42
    python scripts/split_train_test.py --stratify ground_truth_diagnosis
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from glaucpio.config import PROCESSED_DIR
from glaucpio.logger import get_logger

log = get_logger("glaucpio.split")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split treino/teste estratificado."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DIR / "dataset_com_features.csv"),
        help="CSV de entrada (default: data/processed/dataset_com_features.csv).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporção do conjunto de teste (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--stratify",
        type=str,
        default="ground_truth_diagnosis",
        help="Coluna para estratificação (default: ground_truth_diagnosis).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_DIR),
        help="Diretório de saída.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("SPLIT TREINO/TESTE — Projeto GlaucIA")
    log.info("=" * 60)

    input_csv = Path(args.input)
    if not input_csv.exists():
        log.error(f"CSV não encontrado: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    log.info(f"Shape original: {df.shape}")

    # Coluna de estratificação
    stratify_col = None
    if args.stratify and args.stratify in df.columns:
        # Verificar se todas as classes têm >= 2 amostras
        contagem = df[args.stratify].value_counts()
        if contagem.min() >= 2:
            stratify_col = df[args.stratify]
            log.info(f"Estratificando por: {args.stratify}")
        else:
            log.warning(
                f"Coluna '{args.stratify}' tem classes com <2 amostras. "
                "Split sem estratificação."
            )
    else:
        log.warning(f"Coluna de estratificação '{args.stratify}' não encontrada.")

    # Split
    train, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_col,
    )

    log.info(f"Treino: {len(train)} ({100*len(train)/len(df):.1f}%)")
    log.info(f"Teste:  {len(test)} ({100*len(test)/len(df):.1f}%)")

    # Salvar
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_train = output_dir / "train.csv"
    out_test = output_dir / "test.csv"
    train.to_csv(out_train, index=False)
    test.to_csv(out_test, index=False)

    log.info(f"→ {out_train}")
    log.info(f"→ {out_test}")

    # Distribuição de classes
    if stratify_col is not None:
        log.info("-" * 60)
        log.info("Distribuição de classes:")
        dist_train = train[args.stratify].value_counts(normalize=True).round(3)
        dist_test = test[args.stratify].value_counts(normalize=True).round(3)
        for classe in sorted(dist_train.index):
            log.info(
                f"  {classe}: treino={dist_train.get(classe, 0):.3f}, "
                f"teste={dist_test.get(classe, 0):.3f}"
            )

    log.info("=" * 60)
    log.info("✅ Split concluído.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()