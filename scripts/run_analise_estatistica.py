#!/usr/bin/env python3
"""
Replicação do baseline: Gobira et al. (2026) — 189 MCQs de glaucoma.

Gera:
- Tabela 1: acurácia por participante com IC 95% Wilson
- Tabela 2: McNemar exato + Haldane-Anscombe
- Figura 1: barras com IC

A Figura 2 exige resultados pareados por item e não é criada a partir de dados
agregados do artigo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from glaucpio.config import CSV_DIR, FIGURES_DIR
from glaucpio.logger import get_logger
from glaucpio.metricas import (
    wilson_ci, mcnemar_exact, haldane_anscombe_or,
)
from glaucpio.visualizacao import (
    figura1_acuracia_wilson,
)

log = get_logger("glaucpio.replicacao")


# Dados do artigo original (Tabela 1)
PARTICIPANTES = {
    "ChatGPT-5":   {"corretas": 164, "total": 189},
    "Resident 01": {"corretas": 132, "total": 189},
    "Resident 02": {"corretas": 134, "total": 189},
    "Resident 03": {"corretas": 123, "total": 189},
    "Resident 04": {"corretas": 145, "total": 189},
    "Resident 05": {"corretas":  94, "total": 189},
    "Resident 06": {"corretas":  85, "total": 189},
}
MEDIA_RESIDENTES = {"corretas": 713, "total": 1134}

# Discordâncias pareadas (Tabela 2)
DISCORDANCIAS = {
    "Resident 01": {"b": 45, "c": 13},
    "Resident 02": {"b": 48, "c": 18},
    "Resident 03": {"b": 58, "c": 17},
    "Resident 04": {"b": 41, "c": 21},
    "Resident 05": {"b": 87, "c": 17},
    "Resident 06": {"b": 85, "c":  6},
}


def tabela1():
    log.info("Gerando Tabela 1: acurácia por participante...")
    rows = []
    for nome, d in PARTICIPANTES.items():
        acc = 100 * d["corretas"] / d["total"]
        lo, hi = wilson_ci(d["corretas"], d["total"])
        rows.append({
            "Participante": nome,
            "Corretas": d["corretas"],
            "Total": d["total"],
            "Acurácia (%)": round(acc, 1),
            "IC 95% (Wilson)": f"{lo:.2f}–{hi:.2f}",
        })

    acc_media = 100 * MEDIA_RESIDENTES["corretas"] / MEDIA_RESIDENTES["total"]
    lo, hi = wilson_ci(MEDIA_RESIDENTES["corretas"], MEDIA_RESIDENTES["total"])
    rows.append({
        "Participante": "Média dos Residentes",
        "Corretas": MEDIA_RESIDENTES["corretas"],
        "Total": MEDIA_RESIDENTES["total"],
        "Acurácia (%)": round(acc_media, 1),
        "IC 95% (Wilson)": f"{lo:.2f}–{hi:.2f}",
    })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / "tabela1_acuracia.csv", index=False)
    log.info(f"Tabela 1 salva. {len(df)} linhas.")
    return df


def tabela2():
    log.info("Gerando Tabela 2: McNemar + Haldane-Anscombe...")
    rows = []
    for nome, d in DISCORDANCIAS.items():
        b, c = d["b"], d["c"]
        p, n = mcnemar_exact(b, c)
        or_val, or_lo, or_hi = haldane_anscombe_or(b, c)
        rows.append({
            "Comparação": f"ChatGPT-5 vs. {nome}",
            "b (LLM+/R−)": b,
            "c (LLM−/R+)": c,
            "Pares discordantes": n,
            "Odds Ratio": round(or_val, 2),
            "IC 95% OR": f"{or_lo:.2f}–{or_hi:.2f}",
            "p-valor (McNemar)": f"{p:.4f}" if p >= 0.0001 else "<0.0001",
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / "tabela2_mcnemar.csv", index=False)
    log.info(f"Tabela 2 salva. {len(df)} linhas.")
    return df


def figuras(df_tabela1):
    log.info("Gerando Figura 1 (acurácia com IC)...")
    figura1_acuracia_wilson(df_tabela1, salvar=True)
    log.warning(
        "Figura 2 não gerada: os dados agregados não contêm resultados pareados por item."
    )


def main():
    log.info("=" * 60)
    log.info("REPLICAÇÃO DO BASELINE — Gobira et al. (2026)")
    log.info("=" * 60)

    df1 = tabela1()
    print("\n" + df1.to_string(index=False) + "\n")

    df2 = tabela2()
    print("\n" + df2.to_string(index=False) + "\n")

    figuras(df1)

    log.info("✅ Replicação concluída.")
    log.info(f"Saídas em: {CSV_DIR} e {FIGURES_DIR}")


if __name__ == "__main__":
    main()
