import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from glaucpio.config import FIGURES_DIR
from glaucpio.logger import get_logger

log = get_logger("glaucpio.figura_combinada")

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
})


def main():
    log.info("Gerando figura combinada 2×2...")

    # Caminhos das figuras
    figuras = {
        "curvas": FIGURES_DIR / "fig_resnet_curvas.png",
        "confusao": FIGURES_DIR / "fig_resnet_matriz_confusao.png",
        "reliability": FIGURES_DIR / "fig_resnet_reliability.png",
        "roc": FIGURES_DIR / "fig_resnet_roc.png",
    }

    # Verificar se todas existem
    for nome, path in figuras.items():
        if not path.exists():
            log.error(f"Figura não encontrada: {path}")
            sys.exit(1)
        log.info(f"✓ {nome}: {path}")

    # Criar figura 2×2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) Curvas de aprendizado
    img = mpimg.imread(figuras["curvas"])
    axes[0, 0].imshow(img)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("(a) Curvas de aprendizado", fontsize=12, fontweight="bold", pad=10)

    # (0,1) Matriz de confusão
    img = mpimg.imread(figuras["confusao"])
    axes[0, 1].imshow(img)
    axes[0, 1].axis("off")
    axes[0, 1].set_title("(b) Matriz de confusão", fontsize=12, fontweight="bold", pad=10)

    # (1,0) Reliability diagram
    img = mpimg.imread(figuras["reliability"])
    axes[1, 0].imshow(img)
    axes[1, 0].axis("off")
    axes[1, 0].set_title("(c) Reliability diagram", fontsize=12, fontweight="bold", pad=10)

    # (1,1) Curvas ROC
    img = mpimg.imread(figuras["roc"])
    axes[1, 1].imshow(img)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("(d) Curvas ROC", fontsize=12, fontweight="bold", pad=10)

    plt.suptitle(
        "ResNet-18 no PAPILA — Avaliação Completa",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout()

    # Salvar
    output = FIGURES_DIR / "fig_resnet_combinada.png"
    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"→ {output}")

    # Também gerar versão em PDF (vetorial) para o artigo
    output_pdf = FIGURES_DIR / "fig_resnet_combinada.pdf"
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (nome, path) in zip(axes.ravel(), figuras.items()):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
    plt.suptitle("ResNet-18 no PAPILA — Avaliação Completa",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"→ {output_pdf}")

    log.info("✅ Figura combinada gerada.")


if __name__ == "__main__":
    main()