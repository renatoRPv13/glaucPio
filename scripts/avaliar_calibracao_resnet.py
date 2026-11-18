import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

from glaucpio.config import CSV_DIR, FIGURES_DIR
from glaucpio.logger import get_logger
from glaucpio.metricas import expected_calibration_error, brier_score

log = get_logger("glaucpio.calibracao_resnet")

CLASSES = ["normal", "suspect", "glaucoma"]
CORES = {"normal": "#2ECC71", "suspect": "#F39C12", "glaucoma": "#E74C3C"}


def carregar_predicoes_resnet():
    """Carrega predições salvas pela ResNet."""
    path = CSV_DIR / "resnet_predicoes.csv"
    if not path.exists():
        log.error(f"Predições não encontradas: {path}")
        log.error("Rode primeiro: python scripts/treinar_resnet.py")
        sys.exit(1)

    df = pd.read_csv(path)
    y_test = df["y_test"].values
    y_pred = df["y_pred"].values
    y_proba = df[["y_proba_normal", "y_proba_suspect", "y_proba_glaucoma"]].values

    log.info(f"Predições carregadas: {len(y_test)} casos")
    return y_test, y_pred, y_proba


def plot_reliability(y_test, y_pred, y_proba, n_bins=10):
    """Reliability diagram."""
    confianca = y_proba.max(axis=1)
    correto = (y_pred == y_test).astype(int)

    ece = expected_calibration_error(confianca, correto, n_bins=n_bins)
    bs = brier_score(confianca, correto)

    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    acc_bins = []
    counts = []

    for i in range(n_bins):
        mask = (confianca > bins[i]) & (confianca <= bins[i + 1])
        counts.append(mask.sum())
        if mask.sum() > 0:
            acc_bins.append(correto[mask].mean())
        else:
            acc_bins.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot([0, 1], [0, 1], "k--", label="Calibração perfeita")
    ax1.plot(centers, acc_bins, "o-", color="#00529B",
             label=f"ResNet-18\nECE={ece:.3f}\nBrier={bs:.3f}")
    ax1.set_xlabel("Confiança declarada")
    ax1.set_ylabel("Acurácia observada")
    ax1.set_title("Reliability Diagram — ResNet-18")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(centers, counts, width=0.1, color="#3498DB",
            edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Confiança declarada")
    ax2.set_ylabel("Frequência")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_resnet_reliability.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"→ {path}")

    return ece, bs


def plot_roc(y_test, y_proba):
    """Curvas ROC multiclasse."""
    y_bin = label_binarize(y_test, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 6))
    auc_resultados = {}

    for i, classe in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_resultados[classe] = roc_auc
        ax.plot(fpr, tpr, color=CORES[classe], lw=2,
                label=f"{classe} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curvas ROC — ResNet-18 (PAPILA)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_resnet_roc.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"→ {path}")

    return auc_resultados


def plot_matriz_confusao(y_test, y_pred):
    """Matriz de confusão normalizada."""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_xlabel("Predito")
    axes[0].set_ylabel("Real")
    axes[0].set_title("Matriz de Confusão (absoluta)")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_xlabel("Predito")
    axes[1].set_ylabel("Real")
    axes[1].set_title("Matriz de Confusão (normalizada)")

    plt.tight_layout()
    path = FIGURES_DIR / "fig_resnet_matriz_confusao.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"→ {path}")


def main():
    log.info("=" * 60)
    log.info("AVALIAÇÃO DE CALIBRAÇÃO — RESNET-18")
    log.info("=" * 60)

    y_test, y_pred, y_proba = carregar_predicoes_resnet()

    log.info("Gerando reliability diagram...")
    ece, bs = plot_reliability(y_test, y_pred, y_proba)

    log.info("Gerando curvas ROC...")
    aucs = plot_roc(y_test, y_proba)

    log.info("Gerando matriz de confusão...")
    plot_matriz_confusao(y_test, y_pred)

    # Tabela final
    from sklearn.metrics import f1_score, accuracy_score
    tabela = pd.DataFrame([{
        "Modelo": "ResNet-18",
        "F1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "F1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "ECE": round(ece, 4),
        "Brier": round(bs, 4),
        "AUC_normal": round(aucs["normal"], 4),
        "AUC_suspect": round(aucs["suspect"], 4),
        "AUC_glaucoma": round(aucs["glaucoma"], 4),
        "AUC_macro": round(np.mean(list(aucs.values())), 4),
    }])
    tabela.to_csv(CSV_DIR / "resnet_calibracao.csv", index=False)
    log.info(f"→ {CSV_DIR / 'resnet_calibracao.csv'}")

    log.info("\n" + tabela.to_string(index=False))
    log.info("=" * 60)
    log.info("✅ Avaliação de calibração concluída.")


if __name__ == "__main__":
    main()