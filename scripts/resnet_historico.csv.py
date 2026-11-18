import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import matplotlib.pyplot as plt

from glaucpio.config import CSV_DIR, FIGURES_DIR
from glaucpio.logger import get_logger

log = get_logger("glaucpio.curvas")

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
})


def main():
    log.info("Gerando curvas de aprendizado...")

    df = pd.read_csv(CSV_DIR / "resnet_historico.csv")
    log.info(f"Histórico: {len(df)} épocas")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Loss
    axes[0].plot(df["epoch"], df["train_loss"], "o-", label="Train", color="#3498DB")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss por época")
    axes[0].legend()

    # 2. Acurácia
    axes[1].plot(df["epoch"], df["train_acc"] * 100, "o-", label="Train", color="#2ECC71")
    axes[1].plot(df["epoch"], df["val_acc"] * 100, "s-", label="Validação", color="#E74C3C")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia (%)")
    axes[1].set_title("Acurácia: treino vs validação")
    axes[1].legend()

    # 3. F1 macro
    axes[2].plot(df["epoch"], df["val_f1_macro"], "o-", color="#9B59B6")
    best_epoch = df["val_f1_macro"].idxmax() + 1
    best_f1 = df["val_f1_macro"].max()
    axes[2].axhline(y=best_f1, color="red", linestyle="--", alpha=0.5,
                    label=f"Melhor: {best_f1:.3f} (época {best_epoch})")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("F1 macro")
    axes[2].set_title("F1 macro (validação)")
    axes[2].legend()

    plt.tight_layout()
    path = FIGURES_DIR / "fig_resnet_curvas.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info(f"→ {path}")

    # Figura combinada para o artigo
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["epoch"], df["train_acc"] * 100, "o-", label="Train Acc",
             color="#3498DB", alpha=0.7)
    ax1.plot(df["epoch"], df["val_acc"] * 100, "s-", label="Val Acc",
             color="#E74C3C")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Acurácia (%)", color="#3498DB")
    ax1.tick_params(axis="y", labelcolor="#3498DB")

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_f1_macro"], "^-", label="F1 macro",
             color="#9B59B6", linewidth=2)
    ax2.set_ylabel("F1 macro", color="#9B59B6")
    ax2.tick_params(axis="y", labelcolor="#9B59B6")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("ResNet-18 — Curvas de Aprendizado (PAPILA)")
    plt.tight_layout()
    path = FIGURES_DIR / "fig_resnet_curvas_artigo.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info(f"→ {path}")

    log.info("✅ Curvas geradas.")


if __name__ == "__main__":
    main()