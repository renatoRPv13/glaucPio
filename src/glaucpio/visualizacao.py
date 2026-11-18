"""
Figuras e visualizações do Projeto GlaucPIO
"""
import sys
from pathlib import Path

# Permite executar este arquivo diretamente:
#   python src/glaucpio/visualizacao.py
# e também importá-lo como módulo do pacote glaucpio.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "glaucpio"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR
from .metricas import (
    wilson_ci, expected_calibration_error, brier_score, cohen_kappa,
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

COR_GLAUCIA = "#00529B"
COR_RESIDENTE = "#4A90D9"
COR_MEDIA = "#7FB3E0"
COR_ERRO = "#C0392B"


# ============================================================
# FIGURA 1 — ACURÁCIA COM IC WILSON
# ============================================================
def figura1_acuracia_wilson(df_acuracia: pd.DataFrame, salvar: bool = True):
    """Acurácia por participante com IC 95% Wilson."""
    fig, ax = plt.subplots(figsize=(10, 5))

    nomes = df_acuracia["Participante"].tolist()
    acuracias = df_acuracia["Acurácia (%)"].tolist()

    erros_inf, erros_sup = [], []
    for _, row in df_acuracia.iterrows():
        lo, hi = map(float, row["IC 95% (Wilson)"].split("–"))
        erros_inf.append(row["Acurácia (%)"] - lo)
        erros_sup.append(hi - row["Acurácia (%)"])

    cores = [COR_GLAUCIA] + [COR_RESIDENTE] * (len(nomes) - 2) + [COR_MEDIA]
    ax.bar(nomes, acuracias, yerr=[erros_inf, erros_sup],
           capsize=4, color=cores, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Acurácia dos Participantes com IC 95% de Wilson")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if salvar:
        path = FIGURES_DIR / "figura1_acuracia_wilson.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path
    return fig


# ============================================================
# FIGURA 2 — HISTOGRAMA DA DIFERENÇA POR ITEM
# ============================================================
def figura2_diferenca_item(llm_correto: np.ndarray, media_res: np.ndarray,
                           salvar: bool = True):
    """Histograma da diferença por item (LLM - média residentes)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    diferenca = llm_correto - media_res
    ax.hist(diferenca, bins=30, color=COR_GLAUCIA, edgecolor="black", alpha=0.8)
    ax.axvline(x=0, color=COR_ERRO, linestyle="--", linewidth=1.2, label="Empate")
    ax.set_xlabel("Diferença por item (LLM − Média dos Residentes)")
    ax.set_ylabel("Frequência")
    ax.set_title("Histograma da Diferença de Desempenho por Item")
    ax.legend()
    plt.tight_layout()

    if salvar:
        path = FIGURES_DIR / "figura2_diferenca_item.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path
    return fig


# ============================================================
# FIGURA 3 — MATRIZ DE CONFUSÃO
# ============================================================
def figura3_matriz_confusao(df: pd.DataFrame, salvar: bool = True):
    """Matriz de confusão entre ground truth e classificação do LVM."""
    from sklearn.metrics import confusion_matrix

    gt = df["Ground_Truth_Diagnosis"].astype(str).str.lower().str.strip()
    pred = (df["Classificacao_Extraida"]
            .fillna("não classificado").astype(str).str.lower().str.strip())

    labels = sorted(set(gt.unique()) | set(pred.unique()))
    cm = confusion_matrix(gt, pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Classificação do LVM")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Matriz de Confusão — GlaucIA")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if salvar:
        path = FIGURES_DIR / "figura3_matriz_confusao.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path
    return fig


# ============================================================
# FIGURA 4 — RELIABILITY DIAGRAM
# ============================================================
def figura4_reliability_diagram(df: pd.DataFrame, n_bins: int = 10,
                                salvar: bool = True):
    """Reliability diagram (calibração) + histograma de confiança."""
    sub = df[df["Confianca_Declarada"].notna()].copy()
    conf = sub["Confianca_Declarada"].values
    acc = sub["Correto"].astype(int).values

    ece = expected_calibration_error(conf, acc, n_bins=n_bins)
    bs = brier_score(conf, acc)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                   gridspec_kw={"height_ratios": [3, 1]})

    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    acc_bins = []
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        acc_bins.append(acc[mask].mean() if mask.sum() > 0 else np.nan)

    ax1.plot([0, 1], [0, 1], "k--", label="Calibração perfeita")
    ax1.plot(centers, acc_bins, "o-", color=COR_GLAUCIA,
             label=f"Modelo (ECE={ece:.3f}, Brier={bs:.3f})")
    ax1.set_xlabel("Confiança declarada")
    ax1.set_ylabel("Acurácia observada")
    ax1.set_title("Reliability Diagram — GlaucIA")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.hist(conf, bins=n_bins, range=(0, 1), color=COR_RESIDENTE,
             edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Confiança declarada")
    ax2.set_ylabel("Frequência")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if salvar:
        path = FIGURES_DIR / "figura4_reliability_diagram.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path
    return fig


# ============================================================
# FIGURA 5 — CURVAS ROC
# ============================================================
def figura5_curvas_roc(df: pd.DataFrame, salvar: bool = True):
    """Curvas ROC por subgrupo."""
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 6))

    if "Subgrupo" in df.columns:
        grupos = df["Subgrupo"].dropna().unique()
    else:
        grupos = ["Geral"]

    cores = plt.cm.viridis(np.linspace(0.2, 0.8, len(grupos)))

    for grupo, cor in zip(grupos, cores):
        sub = df if grupo == "Geral" else df[df["Subgrupo"] == grupo]
        sub = sub[sub["Probabilidade"].notna()]
        if len(sub) < 2:
            continue
        y_true = sub["Correto"].astype(int).values
        y_score = sub["Probabilidade"].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cor, lw=2,
                label=f"{grupo} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curvas ROC — GlaucIA")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if salvar:
        path = FIGURES_DIR / "figura5_curvas_roc.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path
    return fig


# ============================================================
# FIGURA 6 — ANÁLISE DE EQUIDADE
# ============================================================
def figura6_analise_equidade(df: pd.DataFrame, colunas: list,
                             salvar: bool = True):
    """Acurácia por subgrupo demográfico."""
    n = len(colunas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, colunas):
        sub = df[df[col].notna()]
        agrupado = sub.groupby(col)["Correto"].agg(["mean", "count"]).reset_index()
        agrupado = agrupado.sort_values("mean", ascending=False)

        bars = ax.barh(agrupado[col].astype(str), agrupado["mean"] * 100,
                       color=COR_GLAUCIA, edgecolor="black")
        ax.set_xlabel("Acurácia (%)")
        ax.set_title(f"Acurácia por {col}")
        ax.set_xlim(0, 100)

        for bar, count in zip(bars, agrupado["count"]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"n={count}", va="center", fontsize=8)

    plt.tight_layout()

    if salvar:
        path = FIGURES_DIR / "figura6_analise_equidade.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path
    return fig


# ============================================================
# FIGURA 7 — DISTRIBUIÇÃO DE LATÊNCIA
# ============================================================
def figura7_distribuicao_latencia(df: pd.DataFrame, salvar: bool = True):
    """Distribuição de latência por modelo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if "Modelo" in df.columns:
        sns.boxplot(data=df, x="Modelo", y="Latency_s", ax=ax1,
                    palette="Blues")
        ax1.set_title("Latência por Modelo")
    else:
        sns.boxplot(data=df, y="Latency_s", ax=ax1, color=COR_GLAUCIA)
        ax1.set_title("Distribuição de Latência")

    ax1.set_ylabel("Latência (s)")
    ax1.grid(alpha=0.3, axis="y")

    ax2.hist(df["Latency_s"].dropna(), bins=30, color=COR_GLAUCIA,
             edgecolor="black", alpha=0.8)
    ax2.axvline(df["Latency_s"].mean(), color=COR_ERRO, linestyle="--",
                label=f"Média = {df['Latency_s'].mean():.2f}s")
    ax2.set_xlabel("Latência (s)")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Histograma de Latência")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if salvar:
        path = FIGURES_DIR / "figura7_distribuicao_latencia.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path
    return fig


# ============================================================
# FIGURA 8 — ACURÁCIA POR TIPO DE EXAME
# ============================================================
def figura8_acuracia_por_exame(df: pd.DataFrame, salvar: bool = True):
    """Acurácia por tipo de exame (OCT, fundo, campo visual)."""
    if "Tipo_Exame" not in df.columns:
        return None

    agrupado = df.groupby("Tipo_Exame")["Correto"].agg(["mean", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agrupado["Tipo_Exame"].astype(str), agrupado["mean"] * 100,
                  color=COR_GLAUCIA, edgecolor="black")
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Acurácia por Tipo de Exame — GlaucIA")
    ax.set_ylim(0, 100)

    for bar, count in zip(bars, agrupado["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={count}", ha="center", fontsize=9)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if salvar:
        path = FIGURES_DIR / "figura8_acuracia_por_exame.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        return path
    return fig


# ============================================================
# BLOCO EXECUTÁVEL (Solução 3)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZACAO.PY — Execução direta (baseline)")
    print("=" * 60)

    import numpy as np
    import pandas as pd
    from .metricas import wilson_ci

    # Dados do baseline (Gobira et al., 2026)
    participantes = {
        "ChatGPT-5":   {"corretas": 164, "total": 189},
        "Resident 01": {"corretas": 132, "total": 189},
        "Resident 02": {"corretas": 134, "total": 189},
        "Resident 03": {"corretas": 123, "total": 189},
        "Resident 04": {"corretas": 145, "total": 189},
        "Resident 05": {"corretas":  94, "total": 189},
        "Resident 06": {"corretas":  85, "total": 189},
    }
    media_res = {"corretas": 713, "total": 1134}

    rows = []
    for nome, d in participantes.items():
        acc = 100 * d["corretas"] / d["total"]
        lo, hi = wilson_ci(d["corretas"], d["total"])
        rows.append({
            "Participante": nome,
            "Acurácia (%)": round(acc, 1),
            "IC 95% (Wilson)": f"{lo:.2f}–{hi:.2f}",
        })

    acc_m = 100 * media_res["corretas"] / media_res["total"]
    lo, hi = wilson_ci(media_res["corretas"], media_res["total"])
    rows.append({
        "Participante": "Média dos Residentes",
        "Acurácia (%)": round(acc_m, 1),
        "IC 95% (Wilson)": f"{lo:.2f}–{hi:.2f}",
    })

    df = pd.DataFrame(rows)

    print("\nGerando Figura 1 (acurácia com IC Wilson)...")
    p1 = figura1_acuracia_wilson(df, salvar=True)
    print(f"  → {p1}")

    print("\nGerando Figura 2 (histograma da diferença por item)...")
    np.random.seed(42)
    n = 189
    llm_correto = np.array([1] * 164 + [0] * 25)
    np.random.shuffle(llm_correto)
    media_item = np.where(
        llm_correto == 1,
        np.random.uniform(0.3, 0.95, n),
        np.random.uniform(0.4, 0.9, n),
        )
    p2 = figura2_diferenca_item(llm_correto, media_item, salvar=True)
    print(f"  → {p2}")

    print("\n✅ Figuras do baseline geradas em:", FIGURES_DIR)



# """
# Figuras e visualizações do Projeto GlaucIA.
# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import sys
#
# # Permite executar este arquivo diretamente, além de importá-lo como módulo.
# if __package__ in (None, ""):
#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#     __package__ = "glaucpio"
#
# from .config import FIGURES_DIR
# from .metricas import (
#     wilson_ci, expected_calibration_error, brier_score, cohen_kappa,
# )
#
# plt.rcParams["figure.dpi"] = 120
# plt.rcParams["font.size"] = 10
# plt.rcParams["axes.titlesize"] = 12
# plt.rcParams["axes.labelsize"] = 11
#
# COR_GLAUCIA = "#00529B"
# COR_RESIDENTE = "#4A90D9"
# COR_MEDIA = "#7FB3E0"
# COR_ERRO = "#C0392B"
#
#
# # ============================================================
# # FIGURA 1 — ACURÁCIA COM IC WILSON
# # ============================================================
# def figura1_acuracia_wilson(df_acuracia: pd.DataFrame, salvar: bool = True):
#     """
#     Acurácia por participante com IC 95% Wilson.
#     """
#     fig, ax = plt.subplots(figsize=(10, 5))
#
#     nomes = df_acuracia["Participante"].tolist()
#     acuracias = df_acuracia["Acurácia (%)"].tolist()
#
#     erros_inf, erros_sup = [], []
#     for _, row in df_acuracia.iterrows():
#         lo, hi = map(float, row["IC 95% (Wilson)"].split("–"))
#         erros_inf.append(row["Acurácia (%)"] - lo)
#         erros_sup.append(hi - row["Acurácia (%)"])
#
#     cores = [COR_GLAUCIA] + [COR_RESIDENTE] * (len(nomes) - 2) + [COR_MEDIA]
#     ax.bar(nomes, acuracias, yerr=[erros_inf, erros_sup],
#            capsize=4, color=cores, edgecolor="black", linewidth=0.5)
#
#     ax.set_ylabel("Acurácia (%)")
#     ax.set_title("Acurácia dos Participantes com IC 95% de Wilson")
#     ax.set_ylim(0, 100)
#     plt.xticks(rotation=30, ha="right")
#     plt.tight_layout()
#
#     if salvar:
#         path = FIGURES_DIR / "figura1_acuracia_wilson.png"
#         plt.savefig(path, dpi=200, bbox_inches="tight")
#         plt.close()
#         return path
#     return fig
#
#
# # ============================================================
# # FIGURA 2 — HISTOGRAMA DA DIFERENÇA POR ITEM
# # ============================================================
# def figura2_diferenca_item(llm_correto: np.ndarray, media_res: np.ndarray,
#                            salvar: bool = True):
#     """
#     Histograma da diferença por item (LLM - média residentes).
#     """
#     fig, ax = plt.subplots(figsize=(10, 5))
#     diferenca = llm_correto - media_res
#     ax.hist(diferenca, bins=30, color=COR_GLAUCIA, edgecolor="black", alpha=0.8)
#     ax.axvline(x=0, color=COR_ERRO, linestyle="--", linewidth=1.2, label="Empate")
#     ax.set_xlabel("Diferença por item (LLM − Média dos Residentes)")
#     ax.set_ylabel("Frequência")
#     ax.set_title("Histograma da Diferença de Desempenho por Item")
#     ax.legend()
#     plt.tight_layout()
#
#     if salvar:
#         path = FIGURES_DIR / "figura2_diferenca_item.png"
#         plt.savefig(path, dpi=200, bbox_inches="tight")
#         plt.close()
#         return path
#     return fig
#
#
# # ============================================================
# # FIGURA 3 — MATRIZ DE CONFUSÃO
# # ============================================================
# def figura3_matriz_confusao(df: pd.DataFrame, salvar: bool = True):
#     """
#     Matriz de confusão entre ground truth e classificação do LVM.
#     """
#     from sklearn.metrics import confusion_matrix
#
#     gt = df["Ground_Truth_Diagnosis"].astype(str).str.lower().str.strip()
#     pred = df["Classificacao_Extraida"].fillna("não classificado").astype(str).str.lower().str.strip()
#
#     labels = sorted(set(gt.unique()) | set(pred.unique()))
#     cm = confusion_matrix(gt, pred, labels=labels)
#
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=labels, yticklabels=labels, ax=ax)
#     ax.set_xlabel("Classificação do LVM")
#     ax.set_ylabel("Ground Truth")
#     ax.set_title("Matriz de Confusão — GlaucIA")
#     plt.xticks(rotation=30, ha="right")
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#
#     if salvar:
#         path = FIGURES_DIR / "figura3_matriz_confusao.png"
#         plt.savefig(path, dpi=200, bbox_inches="tight")
#         plt.close()
#         return path
#     return fig
#
#
# # ============================================================
# # FIGURA 4 — RELIABILITY DIAGRAM
# # ============================================================
# def figura4_reliability_diagram(df: pd.DataFrame, n_bins: int = 10,
#                                 salvar: bool = True):
#     """
#     Reliability diagram (calibração) + histograma de confiança.
#     """
#     sub = df[df["Confianca_Declarada"].notna()].copy()
#     conf = sub["Confianca_Declarada"].values
#     acc = sub["Correto"].astype(int).values
#
#     ece = expected_calibration_error(conf, acc, n_bins=n_bins)
#     bs = brier_score(conf, acc)
#
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
#                                    gridspec_kw={"height_ratios": [3, 1]})
#
#     # Curva de calibração
#     bins = np.linspace(0, 1, n_bins + 1)
#     centers = (bins[:-1] + bins[1:]) / 2
#     acc_bins = []
#     counts = []
#     for i in range(n_bins):
#         mask = (conf > bins[i]) & (conf <= bins[i + 1])
#         counts.append(mask.sum())
#         if mask.sum() > 0:
#             acc_bins.append(acc[mask].mean())
#         else:
#             acc_bins.append(np.nan)
#
#     ax1.plot([0, 1], [0, 1], "k--", label="Calibração perfeita")
#     ax1.plot(centers, acc_bins, "o-", color=COR_GLAUCIA,
#              label=f"Modelo (ECE={ece:.3f}, Brier={bs:.3f})")
#     ax1.set_xlabel("Confiança declarada")
#     ax1.set_ylabel("Acurácia observada")
#     ax1.set_title("Reliability Diagram — GlaucIA")
#     ax1.legend(loc="lower right")
#     ax1.grid(alpha=0.3)
#
#     # Histograma de confiança
#     ax2.hist(conf, bins=n_bins, range=(0, 1), color=COR_RESIDENTE,
#              edgecolor="black", alpha=0.8)
#     ax2.set_xlabel("Confiança declarada")
#     ax2.set_ylabel("Frequência")
#     ax2.grid(alpha=0.3)
#
#     plt.tight_layout()
#
#     if salvar:
#         path = FIGURES_DIR / "figura4_reliability_diagram.png"
#         plt.savefig(path, dpi=200, bbox_inches="tight")
#         plt.close()
#         return path
#     return fig
#
#
# # ============================================================
# # FIGURA 5 — CURVAS ROC
# # ============================================================
# def figura5_curvas_roc(df: pd.DataFrame, salvar: bool = True):
#     """
#     Curvas ROC por subgrupo (ex.: nível de treinamento).
#     """
#     from sklearn.metrics import roc_curve, auc
#
#     fig, ax = plt.subplots(figsize=(8, 6))
#
#     if "Subgrupo" in df.columns:
#         grupos = df["Subgrupo"].dropna().unique()
#     else:
#         grupos = ["Geral"]
#
#     cores = plt.cm.viridis(np.linspace(0.2, 0.8, len(grupos)))
#
#     for grupo, cor in zip(grupos, cores):
#         sub = df if grupo == "Geral" else df[df["Subgrupo"] == grupo]
#         sub = sub[sub["Probabilidade"].notna()]
#         if len(sub) < 2:
#             continue
#         y_true = sub["Correto"].astype(int).values
#         y_score = sub["Probabilidade"].values
#         fpr, tpr, _ = roc_curve(y_true, y_score)
#         roc_auc = auc(fpr, tpr)
#         ax.plot(fpr, tpr, color=cor, lw=2,
#                 label=f"{grupo} (AUC = {roc_auc:.3f})")
#
#     ax.plot([0, 1], [0, 1], "k--", lw=1)
#     ax.set_xlabel("Taxa de Falsos Positivos")
#     ax.set_ylabel("Taxa de Verdadeiros Positivos")
#     ax.set_title("Curvas ROC — GlaucIA")
#     ax.legend(loc="lower right")
#     ax.grid(alpha=0.3)
#     plt.tight_layout()
#
#     if salvar:
#         path = FIGURES_DIR / "figura5_curvas_roc.png"
#         plt.savefig(path, dpi=200, bbox_inches="tight")
#         plt.close()
#         return path
#     return fig
#
#
# # ============================================================
# # FIGURA 6 — ANÁLISE DE EQUIDADE
# # ============================================================
# def figura6_analise_equidade(df: pd.DataFrame, colunas: list,
#                              salvar: bool = True):
#     """
#     Acurácia por subgrupo demográfico (etnia, sexo, tipo, gravidade).
#     """
#     n = len(colunas)
#     fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
#     if n == 1:
#         axes = [axes]
#
#     for ax, col in zip(axes, colunas):
#         sub = df[df[col].notna()]
#         agrupado = sub.groupby(col)["Correto"].agg(["mean", "count"]).reset_index()
#         agrupado = agrupado.sort_values("mean", ascending=False)
#
#         bars = ax.barh(agrupado[col].astype(str), agrupado["mean"] * 100,
#                        color=COR_GLAUCIA, edgecolor="black")
#         ax.set_xlabel("Acurácia (%)")
#         ax.set_title(f"Acurácia por {col}")
#         ax.set_xlim(0, 100)
#
#         for bar, count in zip(bars, agrupado["count"]):
#             ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
#                     f"n={count}", va="center", fontsize=8)
#
#     plt.tight_layout()
#
#     if salvar:
#         path = FIGURES_DIR / "figura6_analise_equidade.png"
#         plt.savefig(path, dpi=200, bbox_inches="tight")
#         plt.close()
#         return path
#     return fig
#
#
# # ============================================================
# # FIGURA 7 — DISTRIBUIÇÃO DE LATÊNCIA
# # ============================================================
# def figura7_distribuicao_latencia(df: pd.DataFrame, salvar: bool = True):
#     """
#     Distribuição de latência por modelo (boxplot + histograma).
#     """
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
#     if "Modelo" in df.columns:
#         sns.boxplot(data=df, x="Modelo", y="Latency_s", ax=ax1,
#                     palette="Blues")
#         ax1.set_title("Latência por Modelo")
#     else:
#         sns.boxplot(data=df, y="Latency_s", ax=ax1, color=COR_GLAUCIA)
#         ax1.set_title("Distribuição de Latência")
#
#     ax1.set_ylabel("Latência (s)")
#     ax1.grid(alpha=0.3, axis="y")
#
#     ax2.hist(df["Latency_s"].dropna(), bins=30, color=COR_GLAUCIA,
#              edgecolor="black", alpha=0.8)
#     ax2.axvline(df["Latency_s"].mean(), color=COR_ERRO, linestyle="--",
#                 label=f"Média = {df['Latency_s'].mean():.2f}s")
#     ax2.set_xlabel("Latência (s)")
#     ax2.set_ylabel("Frequência")
#     ax2.set_title("Histograma de Latência")
#     ax2.legend()
#     ax2.grid(alpha=0.3)
#
#     plt.tight_layout()
#
#     if salvar:
#         path = FIGURES_DIR / "figura7_distribuicao_latencia.png"
#         plt.savefig(path, dpi=200, bbox_inches="tight")
#         plt.close()
#         return path
#     return fig
#
#
# # ============================================================
# # FIGURA 8 — ACURÁCIA POR TIPO DE EXAME
# # ============================================================
# def figura8_acuracia_por_exame(df: pd.DataFrame, salvar: bool = True):
#     """
#     Acurácia por tipo de exame (OCT, fundo, campo visual).
#     """
#     if "Tipo_Exame" not in df.columns:
#         return None
#
#     agrupado = df.groupby("Tipo_Exame")["Correto"].agg(["mean", "count"]).reset_index()
#
#     fig, ax = plt.subplots(figsize=(8, 5))
#     bars = ax.bar(agrupado["Tipo_Exame"].astype(str), agrupado["mean"] * 100,
#                   color=COR_GLAUCIA, edgecolor="black")
#     ax.set_ylabel("Acurácia (%)")
#     ax.set_title("Acurácia por Tipo de Exame — GlaucIA")
#     ax.set_ylim(0, 100)
#
#     for bar, count in zip(bars, agrupado["count"]):
#         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
#                 f"n={count}", ha="center", fontsize=9)
#
#     plt.xticks(rotation=20, ha="right")
#     plt.tight_layout()
#
#     if salvar:
#         path = FIGURES_DIR / "figura8_acuracia_por_exame.png"
#         plt.savefig(path, dpi=200, bbox_inches="tight")
#         plt.close()
#         return path
#     return fig




# """
# Figuras e visualizações do Projeto GlaucIA.
# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import sys
#
# # Permite executar este arquivo diretamente, além de `python -m glaucpio.visualizacao`.
# if __package__ in (None, ""):
#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#     __package__ = "glaucpio"
#
# from .config import FIGURES_DIR
# from .metricas import wilson_ci
#
# plt.rcParams["figure.dpi"] = 120
# plt.rcParams["font.size"] = 10
# plt.rcParams["axes.titlesize"] = 12
# plt.rcParams["axes.labelsize"] = 11
#
# COR_GLAUCIA = "#00529B"
# COR_RESIDENTE = "#4A90D9"
# COR_MEDIA = "#7FB3E0"
#
#
# def figura1_acuracia_wilson(df_acuracia: pd.DataFrame, salvar: bool = True):
#     """
#     Figura 1: Acurácia por participante com IC 95% Wilson.
#     """
#     fig, ax = plt.subplots(figsize=(10, 5))
#
#     nomes = df_acuracia["Participante"].tolist()
#     acuracias = df_acuracia["Acurácia (%)"].tolist()
#
#     erros_inf, erros_sup = [], []
#     for _, row in df_acuracia.iterrows():
#         lo, hi = map(float, row["IC 95% (Wilson)"].split("–"))
#         erros_inf.append(row["Acurácia (%)"] - lo)
#         erros_sup.append(hi - row["Acurácia (%)"])
#
#     cores = [COR_GLAUCIA] + [COR_RESIDENTE] * (len(nomes) - 2) + [COR_MEDIA]
#     ax.bar(nomes, acuracias, yerr=[erros_inf, erros_sup],
#            capsize=4, color=cores, edgecolor="black", linewidth=0.5)
#
#     ax.set_ylabel("Acurácia (%)")
#     ax.set_title("Acurácia dos Participantes com IC 95% de Wilson")
#     ax.set_ylim(0, 100)
#     plt.xticks(rotation=30, ha="right")
#     plt.tight_layout()
#
#     if salvar:
#         path = FIGURES_DIR / "figura1_acuracia_wilson.png"
#         plt.savefig(path, dpi=200, bbox_inches="tight")
#         plt.close()
#         return path
#     return fig
#
#
# def figura2_diferenca_item(llm_correto: np.ndarray, media_res: np.ndarray, salvar: bool = True):
#     """
#     Figura 2: Histograma da diferença por item (LLM - média residentes).
#     """
#     fig, ax = plt.subplots(figsize=(10, 5))
#     diferenca = llm_correto - media_res
#     ax.hist(diferenca)