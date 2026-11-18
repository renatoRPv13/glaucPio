#!/usr/bin/env python3
"""
Script de geração de todas as figuras do Projeto GlaucPIO..

Gera:
    Baseline (Gobira et al., 2026):
        1. Acurácia por participante com IC 95% Wilson
        2. Histograma da diferença por item

    Multimodal (Projeto GlaucIA):
        3. Matriz de confusão
        4. Reliability diagram (calibração)
        5. Curvas ROC por subgrupo
        6. Análise de equidade (etnia, sexo, tipo, gravidade)
        7. Distribuição de latência
        8. Acurácia por tipo de exame

Uso:
    # Todas as figuras (baseline + multimodal)
    python scripts/run_gerar_figuras.py --baseline --csv outputs/csv/gemini_3_6_flash_eval_results.csv --baseline-pairs data/raw/paired_item_results.csv

    # Apenas baseline
    python scripts/run_gerar_figuras.py --baseline

    # Apenas figuras específicas
    python scripts/run_gerar_figuras.py --csv outputs/csv/gemini_eval_results.csv --figuras 3,4,7

    # Todas as figuras multimodais (sem baseline)
    python scripts/run_gerar_figuras.py --csv outputs/csv/gemini_eval_results.csv

    # Usar CSV de outro modelo
    python scripts/run_gerar_figuras.py --csv outputs/csv/gpt_eval_results.csv
"""
import argparse
import sys
from pathlib import Path

# Adicionar src/ ao PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from glaucpio.config import FIGURES_DIR, OUTPUT_AVALIACAO_CSV
from glaucpio.logger import get_logger
from glaucpio.metricas import wilson_ci
from glaucpio.visualizacao import (
    figura1_acuracia_wilson,
    figura2_diferenca_item,
    figura3_matriz_confusao,
    figura4_reliability_diagram,
    figura5_curvas_roc,
    figura6_analise_equidade,
    figura7_distribuicao_latencia,
    figura8_acuracia_por_exame,
)

log = get_logger("glaucPIO.figuras")


# ============================================================
# DADOS DO BASELINE (Gobira et al., 2026)
# ============================================================
BASELINE_PARTICIPANTES = {
    "ChatGPT-5":   {"corretas": 164, "total": 189},
    "Resident 01": {"corretas": 132, "total": 189},
    "Resident 02": {"corretas": 134, "total": 189},
    "Resident 03": {"corretas": 123, "total": 189},
    "Resident 04": {"corretas": 145, "total": 189},
    "Resident 05": {"corretas":  94, "total": 189},
    "Resident 06": {"corretas":  85, "total": 189},
}
BASELINE_MEDIA = {"corretas": 713, "total": 1134}
# ============================================================
# MAPEAMENTO DE FIGURAS
# ============================================================
FIGURAS_DISPONIVEIS = {
    1: "Acurácia por participante com IC 95% Wilson (baseline)",
    2: "Histograma da diferença por item (baseline; requer dados pareados)",
    3: "Matriz de confusão (multimodal)",
    4: "Reliability diagram / calibração (multimodal)",
    5: "Curvas ROC por subgrupo (multimodal)",
    6: "Análise de equidade (multimodal)",
    7: "Distribuição de latência (multimodal)",
    8: "Acurácia por tipo de exame (multimodal)",
}


# ============================================================
# HELPERS
# ============================================================
def _parse_figuras(figuras_str: str) -> set:
    """
    Converte '1,2,3' ou 'all' em um set de inteiros.
    """
    if figuras_str.lower() == "all":
        return set(FIGURAS_DISPONIVEIS.keys())
    try:
        return {int(x.strip()) for x in figuras_str.split(",") if x.strip()}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Formato inválido para --figuras: {figuras_str}. "
            "Use 'all' ou lista separada por vírgula (ex.: '1,2,3')."
        )


def _carregar_resultados(csv_path: Path) -> pd.DataFrame:
    """Carrega CSV de resultados da avaliação multimodal."""
    if not csv_path.exists():
        log.warning(f"CSV não encontrado: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    total = len(df)
    ok = df[df["Status"] == "OK"].copy()
    log.info(f"Carregados {len(ok)} casos OK de {total} totais.")

    if len(df) > len(ok):
        erros = df[df["Status"] != "OK"]
        log.info(f"Casos com erro: {len(erros)}")
        if "Status" in erros.columns:
            for status, count in erros["Status"].value_counts().items():
                log.info(f"  {status}: {count}")

    return ok


def _df_acuracia_baseline() -> pd.DataFrame:
    """Constrói DataFrame de acurácia do baseline."""
    rows = []
    for nome, d in BASELINE_PARTICIPANTES.items():
        acc = 100 * d["corretas"] / d["total"]
        lo, hi = wilson_ci(d["corretas"], d["total"])
        rows.append({
            "Participante": nome,
            "Corretas": d["corretas"],
            "Total": d["total"],
            "Acurácia (%)": round(acc, 1),
            "IC 95% (Wilson)": f"{lo:.2f}–{hi:.2f}",
        })

    acc_media = 100 * BASELINE_MEDIA["corretas"] / BASELINE_MEDIA["total"]
    lo, hi = wilson_ci(BASELINE_MEDIA["corretas"], BASELINE_MEDIA["total"])
    rows.append({
        "Participante": "Média dos Residentes",
        "Corretas": BASELINE_MEDIA["corretas"],
        "Total": BASELINE_MEDIA["total"],
        "Acurácia (%)": round(acc_media, 1),
        "IC 95% (Wilson)": f"{lo:.2f}–{hi:.2f}",
    })

    return pd.DataFrame(rows)


# ============================================================
# GERAÇÃO — BASELINE
# ============================================================
def gerar_figura1() -> Path:
    """Figura 1 — Acurácia com IC Wilson (baseline)."""
    log.info("Gerando Figura 1 (acurácia com IC Wilson)...")
    df = _df_acuracia_baseline()
    path = figura1_acuracia_wilson(df, salvar=True)
    log.info(f"  → {path}")
    return path


def gerar_figura2(dados_pareados: pd.DataFrame) -> Path:
    """Figura 2 — Histograma da diferença por item (baseline)."""
    colunas = ("llm_correct", "resident_mean_accuracy")
    ausentes = set(colunas) - set(dados_pareados.columns)
    if ausentes:
        raise ValueError(
            "CSV pareado precisa das colunas: llm_correct e resident_mean_accuracy; "
            f"ausentes: {', '.join(sorted(ausentes))}."
        )

    dados = dados_pareados[list(colunas)].apply(pd.to_numeric, errors="raise").dropna()
    if dados.empty:
        raise ValueError("CSV pareado não contém observações completas.")
    if not dados["llm_correct"].isin([0, 1]).all():
        raise ValueError("A coluna llm_correct deve conter apenas 0 ou 1.")
    if not dados["resident_mean_accuracy"].between(0, 1).all():
        raise ValueError("A coluna resident_mean_accuracy deve estar entre 0 e 1.")

    log.info("Gerando Figura 2 (histograma da diferença por item)...")
    path = figura2_diferenca_item(
        dados["llm_correct"].to_numpy(),
        dados["resident_mean_accuracy"].to_numpy(),
        salvar=True,
    )
    log.info(f"  → {path}")
    return path


# ============================================================
# GERAÇÃO — MULTIMODAL
# ============================================================
def gerar_figura3(df: pd.DataFrame) -> Path | None:
    """Figura 3 — Matriz de confusão."""
    if df.empty:
        log.warning("DataFrame vazio — pulando Figura 3.")
        return None
    if "Ground_Truth_Diagnosis" not in df.columns:
        log.warning("Coluna 'Ground_Truth_Diagnosis' ausente — pulando Figura 3.")
        return None
    if "Classificacao_Extraida" not in df.columns:
        log.warning("Coluna 'Classificacao_Extraida' ausente — pulando Figura 3.")
        return None

    log.info("Gerando Figura 3 (matriz de confusão)...")
    try:
        path = figura3_matriz_confusao(df, salvar=True)
        log.info(f"  → {path}")
        return path
    except Exception as e:
        log.error(f"Falha na Figura 3: {e}")
        return None


def gerar_figura4(df: pd.DataFrame) -> Path | None:
    """Figura 4 — Reliability diagram."""
    if df.empty:
        log.warning("DataFrame vazio — pulando Figura 4.")
        return None
    if "Confianca_Declarada" not in df.columns:
        log.warning("Coluna 'Confianca_Declarada' ausente — pulando Figura 4.")
        return None
    if not df["Confianca_Declarada"].notna().any():
        log.warning("Nenhuma confiança declarada válida — pulando Figura 4.")
        return None

    log.info("Gerando Figura 4 (reliability diagram)...")
    try:
        path = figura4_reliability_diagram(df, salvar=True)
        log.info(f"  → {path}")
        return path
    except Exception as e:
        log.error(f"Falha na Figura 4: {e}")
        return None


def gerar_figura5(df: pd.DataFrame) -> Path | None:
    """Figura 5 — Curvas ROC."""
    if df.empty:
        log.warning("DataFrame vazio — pulando Figura 5.")
        return None
    if "Probabilidade" not in df.columns:
        log.info("Coluna 'Probabilidade' ausente — pulando Figura 5.")
        return None

    log.info("Gerando Figura 5 (curvas ROC)...")
    try:
        path = figura5_curvas_roc(df, salvar=True)
        log.info(f"  → {path}")
        return path
    except Exception as e:
        log.error(f"Falha na Figura 5: {e}")
        return None


def gerar_figura6(df: pd.DataFrame) -> Path | None:
    """Figura 6 — Análise de equidade."""
    if df.empty:
        log.warning("DataFrame vazio — pulando Figura 6.")
        return None

    colunas_equidade = ["Etnia", "Sexo", "Tipo_Glaucoma", "Gravidade"]
    disponiveis = [c for c in colunas_equidade if c in df.columns]
    if not disponiveis:
        log.info("Nenhuma coluna de equidade encontrada — pulando Figura 6.")
        return None

    log.info(f"Gerando Figura 6 (equidade: {', '.join(disponiveis)})...")
    try:
        path = figura6_analise_equidade(df, colunas=disponiveis, salvar=True)
        log.info(f"  → {path}")
        return path
    except Exception as e:
        log.error(f"Falha na Figura 6: {e}")
        return None


def gerar_figura7(df: pd.DataFrame) -> Path | None:
    """Figura 7 — Distribuição de latência."""
    if df.empty:
        log.warning("DataFrame vazio — pulando Figura 7.")
        return None
    if "Latency_s" not in df.columns:
        log.warning("Coluna 'Latency_s' ausente — pulando Figura 7.")
        return None
    if not df["Latency_s"].notna().any():
        log.warning("Nenhuma latência válida — pulando Figura 7.")
        return None

    log.info("Gerando Figura 7 (distribuição de latência)...")
    try:
        path = figura7_distribuicao_latencia(df, salvar=True)
        log.info(f"  → {path}")
        return path
    except Exception as e:
        log.error(f"Falha na Figura 7: {e}")
        return None


def gerar_figura8(df: pd.DataFrame) -> Path | None:
    """Figura 8 — Acurácia por tipo de exame."""
    if df.empty:
        log.warning("DataFrame vazio — pulando Figura 8.")
        return None
    if "Tipo_Exame" not in df.columns:
        log.info("Coluna 'Tipo_Exame' ausente — pulando Figura 8.")
        return None

    log.info("Gerando Figura 8 (acurácia por tipo de exame)...")
    try:
        path = figura8_acuracia_por_exame(df, salvar=True)
        log.info(f"  → {path}")
        return path
    except Exception as e:
        log.error(f"Falha na Figura 8: {e}")
        return None


# ============================================================
# ORQUESTRADOR
# ============================================================
def gerar_figuras(
        df: pd.DataFrame,
        figuras: set,
        incluir_baseline: bool,
        dados_pareados: pd.DataFrame | None = None,
) -> dict:
    """
    Gera as figuras solicitadas.

    Returns:
        dict com mapeamento {número: path_ou_None}
    """
    resultados = {}

    # Baseline (1 e 2)
    if incluir_baseline or 1 in figuras or 2 in figuras:
        if 1 in figuras or incluir_baseline:
            resultados[1] = gerar_figura1()
        if 2 in figuras or incluir_baseline:
            if dados_pareados is None:
                log.warning(
                    "Figura 2 ignorada: informe --baseline-pairs com resultados pareados por item."
                )
                resultados[2] = None
            else:
                resultados[2] = gerar_figura2(dados_pareados)

    # Multimodais (3 a 8)
    geradores = {
        3: gerar_figura3,
        4: gerar_figura4,
        5: gerar_figura5,
        6: gerar_figura6,
        7: gerar_figura7,
        8: gerar_figura8,
    }
    for num, gerador in geradores.items():
        if num in figuras:
            resultados[num] = gerador(df)

    return resultados


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera todas as figuras do Projeto GlaucPIO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Baseline + multimodal completo
  python scripts/run_gerar_figuras.py --baseline --csv outputs/csv/gemini_3_6_flash_eval_results.csv --baseline-pairs data/raw/paired_item_results.csv

  # Apenas baseline
  python scripts/run_gerar_figuras.py --baseline

  # Apenas figuras específicas
  python scripts/run_gerar_figuras.py --csv outputs/csv/gemini_3_6_flash_eval_results.csv --figuras 3,4,7

  # Listar figuras disponíveis
  python scripts/run_gerar_figuras.py --listar
        """,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(OUTPUT_AVALIACAO_CSV),
        help="CSV de resultados da avaliação multimodal.",
    )
    parser.add_argument(
        "--figuras",
        type=str,
        default="all",
        help="Quais figuras gerar (ex.: '1,2,3' ou 'all'). Default: all.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Incluir as figuras do baseline (1 e 2).",
    )
    parser.add_argument(
        "--baseline-pairs",
        type=Path,
        default=None,
        help=(
            "CSV de resultados pareados por item com colunas llm_correct (0/1) "
            "e resident_mean_accuracy (0-1), necessário para a Figura 2."
        ),
    )
    parser.add_argument(
        "--listar",
        action="store_true",
        help="Listar figuras disponíveis e sair.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Modo listar
    if args.listar:
        print("=" * 60)
        print("FIGURAS DISPONÍVEIS — Projeto GlaucPIO")
        print("=" * 60)
        for num, desc in FIGURAS_DISPONIVEIS.items():
            print(f"  [{num}] {desc}")
        print()
        print("Uso: --figuras 1,2,3  ou  --figuras all")
        return

    # Parse das figuras
    try:
        figuras = _parse_figuras(args.figuras)
    except argparse.ArgumentTypeError as e:
        log.error(str(e))
        sys.exit(1)

    log.info("=" * 60)
    log.info("GERAÇÃO DE FIGURAS — PROJETO GlaucPIO")
    log.info("=" * 60)
    log.info(f"CSV:              {args.csv}")
    log.info(f"Figuras:          {sorted(figuras)}")
    log.info(f"Baseline:         {args.baseline}")
    log.info(f"Saída:            {FIGURES_DIR}")
    log.info("=" * 60)

    # Carregar dados multimodais (se aplicável)
    df = pd.DataFrame()
    if not args.baseline or any(n >= 3 for n in figuras):
        df = _carregar_resultados(Path(args.csv))

    # Gerar figuras
    dados_pareados = None
    if args.baseline_pairs is not None:
        dados_pareados = pd.read_csv(args.baseline_pairs)
    resultados = gerar_figuras(
        df,
        figuras,
        incluir_baseline=args.baseline,
        dados_pareados=dados_pareados,
    )

    # Resumo
    log.info("=" * 60)
    log.info("RESUMO")
    log.info("=" * 60)
    ok = sum(1 for p in resultados.values() if p is not None)
    falhas = sum(1 for p in resultados.values() if p is None)
    log.info(f"Figuras geradas com sucesso: {ok}")
    log.info(f"Figuras puladas/falhas:      {falhas}")

    for num, path in sorted(resultados.items()):
        status = "✅" if path is not None else "⚠️"
        log.info(f"  {status} Figura {num}: {path if path else 'não gerada'}")

    log.info("=" * 60)
    log.info(f"✅ Concluído. Verifique {FIGURES_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()




#!/usr/bin/env python3
"""
Script de geração de todas as figuras do Projeto GlaucPIO.

Gera:
- Figura 1: Acurácia por participante com IC 95% Wilson (baseline)
- Figura 2: Histograma da diferença por item (baseline)
- Figura 3: Matriz de confusão multimodal
- Figura 4: Reliability diagram (calibração)
- Figura 5: Curvas ROC por nível de treinamento
- Figura 6: Análise de equidade (desempenho por subgrupo)
- Figura 7: Distribuição de latência por modelo
- Figura 8: Taxa de acerto por tipo de exame

Uso:
    python scripts/run_gerar_figuras.py --csv outputs/csv/gemini_eval_results.csv
    python scripts/run_gerar_figuras.py --csv outputs/csv/gemini_eval_results.csv --figuras 1,2,3
"""
# import argparse
# import sys
# from pathlib import Path
#
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
#
# import numpy as np
# import pandas as pd
#
# from glaucpio.config import CSV_DIR, FIGURES_DIR
# from glaucpio.logger import get_logger
# from glaucpio.visualizacao import (
#     figura1_acuracia_wilson,
#     figura2_diferenca_item,
#     figura3_matriz_confusao,
#     figura4_reliability_diagram,
#     figura5_curvas_roc,
#     figura6_analise_equidade,
#     figura7_distribuicao_latencia,
#     figura8_acuracia_por_exame,
# )
#
# log = get_logger("glaucia.figuras")
#
#
# # ============================================================
# # DADOS DO BASELINE (Gobira et al., 2026)
# # ============================================================
# BASELINE_PARTICIPANTES = {
#     "ChatGPT-5":   {"corretas": 164, "total": 189},
#     "Resident 01": {"corretas": 132, "total": 189},
#     "Resident 02": {"corretas": 134, "total": 189},
#     "Resident 03": {"corretas": 123, "total": 189},
#     "Resident 04": {"corretas": 145, "total": 189},
#     "Resident 05": {"corretas":  94, "total": 189},
#     "Resident 06": {"corretas":  85, "total": 189},
# }
# BASELINE_MEDIA = {"corretas": 713, "total": 1134}
#
#
# def _carregar_resultados(csv_path: Path) -> pd.DataFrame:
#     """Carrega CSV de resultados da avaliação multimodal."""
#     if not csv_path.exists():
#         log.warning(f"CSV não encontrado: {csv_path}")
#         return pd.DataFrame()
#     df = pd.read_csv(csv_path)
#     ok = df[df["Status"] == "OK"].copy()
#     log.info(f"Carregados {len(ok)} casos OK de {len(df)} totais.")
#     return ok
#
#
# def gerar_figuras_baseline():
#     """Gera figuras 1 e 2 (replicação do baseline)."""
#     log.info("=" * 60)
#     log.info("FIGURAS DO BASELINE (Gobira et al., 2026)")
#     log.info("=" * 60)
#
#     # Tabela de acurácia para a Figura 1
#     rows = []
#     for nome, d in BASELINE_PARTICIPANTES.items():
#         from glaucpio.metricas import wilson_ci
#         acc = 100 * d["corretas"] / d["total"]
#         lo, hi = wilson_ci(d["corretas"], d["total"])
#         rows.append({
#             "Participante": nome,
#             "Acurácia (%)": round(acc, 1),
#             "IC 95% (Wilson)": f"{lo:.2f}–{hi:.2f}",
#         })
#
#     from glaucpio.metricas import wilson_ci
#     acc_media = 100 * BASELINE_MEDIA["corretas"] / BASELINE_MEDIA["total"]
#     lo, hi = wilson_ci(BASELINE_MEDIA["corretas"], BASELINE_MEDIA["total"])
#     rows.append({
#         "Participante": "Média dos Residentes",
#         "Acurácia (%)": round(acc_media, 1),
#         "IC 95% (Wilson)": f"{lo:.2f}–{hi:.2f}",
#     })
#
#     df_acuracia = pd.DataFrame(rows)
#
#     # Figura 1
#     log.info("Gerando Figura 1 (acurácia com IC Wilson)...")
#     p1 = figura1_acuracia_wilson(df_acuracia, salvar=True)
#     log.info(f"  → {p1}")
#
#     # Figura 2
#     log.info("Gerando Figura 2 (histograma da diferença por item)...")
#     np.random.seed(42)
#     n = 189
#     llm_correto = np.array([1] * 164 + [0] * 25)
#     np.random.shuffle(llm_correto)
#     media_res = np.where(
#         llm_correto == 1,
#         np.random.uniform(0.3, 0.95, n),
#         np.random.uniform(0.4, 0.9, n),
#     )
#     p2 = figura2_diferenca_item(llm_correto, media_res, salvar=True)
#     log.info(f"  → {p2}")
#
#
# def gerar_figuras_multimodal(df: pd.DataFrame):
#     """Gera figuras 3–8 a partir dos resultados multimodais."""
#     if df.empty:
#         log.warning("DataFrame vazio — pulando figuras multimodais.")
#         return
#
#     log.info("=" * 60)
#     log.info("FIGURAS MULTIMODAIS (Projeto GlaucIA)")
#     log.info("=" * 60)
#
#     # Figura 3: Matriz de confusão
#     if "Ground_Truth_Diagnosis" in df.columns and "Classificacao_Extraida" in df.columns:
#         log.info("Gerando Figura 3 (matriz de confusão)...")
#         try:
#             p3 = figura3_matriz_confusao(df, salvar=True)
#             log.info(f"  → {p3}")
#         except Exception as e:
#             log.warning(f"Falha na Figura 3: {e}")
#
#     # Figura 4: Reliability diagram
#     if "Confianca_Declarada" in df.columns and df["Confianca_Declarada"].notna().any():
#         log.info("Gerando Figura 4 (reliability diagram)...")
#         try:
#             p4 = figura4_reliability_diagram(df, salvar=True)
#             log.info(f"  → {p4}")
#         except Exception as e:
#             log.warning(f"Falha na Figura 4: {e}")
#     else:
#         log.warning("Confiança declarada ausente — pulando Figura 4.")
#
#     # Figura 5: Curvas ROC (se houver probabilidades)
#     if "Probabilidade" in df.columns:
#         log.info("Gerando Figura 5 (curvas ROC)...")
#         try:
#             p5 = figura5_curvas_roc(df, salvar=True)
#             log.info(f"  → {p5}")
#         except Exception as e:
#             log.warning(f"Falha na Figura 5: {e}")
#     else:
#         log.info("Coluna 'Probabilidade' ausente — pulando Figura 5.")
#
#     # Figura 6: Análise de equidade
#     colunas_equidade = ["Etnia", "Sexo", "Tipo_Glaucoma", "Gravidade"]
#     disponiveis = [c for c in colunas_equidade if c in df.columns]
#     if disponiveis:
#         log.info("Gerando Figura 6 (análise de equidade)...")
#         try:
#             p6 = figura6_analise_equidade(df, colunas=disponiveis, salvar=True)
#             log.info(f"  → {p6}")
#         except Exception as e:
#             log.warning(f"Falha na Figura 6: {e}")
#     else:
#         log.info("Colunas de equidade ausentes — pulando Figura 6.")
#
#     # Figura 7: Distribuição de latência
#     if "Latency_s" in df.columns and df["Latency_s"].notna().any():
#         log.info("Gerando Figura 7 (distribuição de latência)...")
#         try:
#             p7 = figura7_distribuicao_latencia(df, salvar=True)
#             log.info(f"  → {p7}")
#         except Exception as e:
#             log.warning(f"Falha na Figura 7: {e}")
#
#     # Figura 8: Acurácia por tipo de exame
#     if "Tipo_Exame" in df.columns:
#         log.info("Gerando Figura 8 (acurácia por tipo de exame)...")
#         try:
#             p8 = figura8_acuracia_por_exame(df, salvar=True)
#             log.info(f"  → {p8}")
#         except Exception as e:
#             log.warning(f"Falha na Figura 8: {e}")
#     else:
#         log.info("Coluna 'Tipo_Exame' ausente — pulando Figura 8.")
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Gera todas as figuras do Projeto GlaucIA."
#     )
#     parser.add_argument(
#         "--csv",
#         type=str,
#         default=str(CSV_DIR / "gemini_eval_results.csv"),
#         help="CSV de resultados da avaliação multimodal.",
#     )
#     parser.add_argument(
#         "--figuras",
#         type=str,
#         default="all",
#         help="Quais figuras gerar (ex.: '1,2,3' ou 'all').",
#     )
#     parser.add_argument(
#         "--baseline",
#         action="store_true",
#         help="Gerar também as figuras do baseline (1 e 2).",
#     )
#     return parser.parse_args()
#
#
# def main():
#     args = parse_args()
#
#     log.info("=" * 60)
#     log.info("GERAÇÃO DE FIGURAS — PROJETO GlaucIA")
#     log.info("=" * 60)
#     log.info(f"CSV: {args.csv}")
#     log.info(f"Figuras: {args.figuras}")
#     log.info(f"Baseline: {args.baseline}")
#
#     # Figuras do baseline (se solicitado)
#     if args.baseline or args.figuras == "all":
#         gerar_figuras_baseline()
#
#     # Figuras multimodais
#     df = _carregar_resultados(Path(args.csv))
#     gerar_figuras_multimodal(df)
#
#     log.info("=" * 60)
#     log.info(f"✅ Figuras geradas em: {FIGURES_DIR}")
#     log.info("=" * 60)
#
#
# if __name__ == "__main__":
#     main()