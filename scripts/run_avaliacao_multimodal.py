#!/usr/bin/env python3
"""
Script principal do Projeto GlaucPIO.

Executa a avaliação multimodal de um LVM (Gemini, GPT, Claude)
sobre o dataset de 500 casos clínicos de glaucoma.

Uso:
    python scripts/run_avaliacao_multimodal.py --modelo gemini
    python scripts/run_avaliacao_multimodal.py --modelo gpt --max-cases 10
    python scripts/run_avaliacao_multimodal.py --modelo claude --sleep 6
"""
import argparse
import sys
from pathlib import Path

# Adicionar src/ ao PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from glaucpio.config import (
    DATASET_MULTIMODAL,
    MODELOS,
    SLEEP_BETWEEN_CALLS,
    MAX_CASES,
    caminhos_saida,
)
from glaucpio.logger import get_logger
from glaucpio.modelos import ModeloFactory
from glaucpio.avaliacao import avaliar_dataset

log = get_logger("glaucPIO.main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliação multimodal de LVM em glaucoma (Projeto GlaucPIO)."
    )
    parser.add_argument(
        "--modelo",
        choices=sorted(MODELOS),
        default="gemini",
        help="Qual LVM usar (default: gemini).",
    )
    parser.add_argument(
        "--nome-modelo",
        type=str,
        default=None,
        help="Nome específico do modelo (ex.: gemini-3.6-flash).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=MAX_CASES,
        help="Número máximo de casos (default: todos).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=SLEEP_BETWEEN_CALLS,
        help="Segundos entre chamadas (evita rate limit).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho do CSV de saída (default: outputs/csv/...).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_MULTIMODAL),
        help="Caminho do dataset CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("PROJETO GlaucPIO — Avaliação Multimodal")
    log.info("=" * 60)
    log.info(f"Modelo: {args.modelo}")
    log.info(f"Dataset: {args.dataset}")
    log.info(f"Max cases: {args.max_cases}")
    log.info(f"Sleep: {args.sleep}s")

    # 1. Instanciar modelo
    nome_modelo = args.nome_modelo or MODELOS[args.modelo]["nome"]
    log.info(f"Inicializando modelo: {nome_modelo}")

    try:
        modelo = ModeloFactory.criar(args.modelo, nome=nome_modelo)
    except Exception as e:
        log.error(f"Falha ao inicializar modelo: {e}")
        sys.exit(1)

    # 2. Definir saída
    if args.output:
        output_csv = Path(args.output)
        output_jsonl = output_csv.with_suffix(".jsonl")
    else:
        output_csv, output_jsonl = caminhos_saida(nome_modelo)

    log.info(f"Saída CSV: {output_csv}")
    log.info(f"Saída JSONL: {output_jsonl}")

    # 3. Rodar avaliação
    try:
        df = avaliar_dataset(
            modelo=modelo,
            dataset_path=Path(args.dataset),
            output_csv=output_csv,
            output_jsonl=output_jsonl,
            max_cases=args.max_cases,
            sleep=args.sleep,
        )
    except KeyboardInterrupt:
        log.warning("Interrompido pelo usuário. Resultados parciais salvos.")
        sys.exit(0)
    except Exception as e:
        log.error(f"Erro durante a avaliação: {e}")
        sys.exit(1)

    # 4. Estatísticas rápidas
    log.info("=" * 60)
    log.info("RESUMO")
    log.info("=" * 60)

    ok = df[df["Status"] == "OK"]
    log.info(f"Casos processados: {len(df)}")
    log.info(f"Casos OK: {len(ok)}")
    log.info(f"Casos com erro: {len(df) - len(ok)}")

    if len(ok) > 0:
        log.info(f"Acurácia: {ok['Correto'].mean():.3f}")
        log.info(f"Latência média: {ok['Latency_s'].mean():.2f}s")
        if ok["Input_Tokens"].notna().any():
            log.info(f"Tokens in (média): {ok['Input_Tokens'].mean():.0f}")
        if ok["Output_Tokens"].notna().any():
            log.info(f"Tokens out (média): {ok['Output_Tokens'].mean():.0f}")
        if ok["N_Images"].notna().any():
            log.info(f"Imagens por caso (média): {ok['N_Images'].mean():.1f}")

    log.info("✅ Avaliação concluída.")


if __name__ == "__main__":
    main()