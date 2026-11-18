"""
Métricas estatísticas para avaliação de LVMs em glaucoma.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "glaucpio"

import numpy as np
from scipy.stats import binomtest, norm
import warnings

warnings.filterwarnings("ignore")


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple:
    p = successes / total
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (center - margin) * 100, (center + margin) * 100


def mcnemar_exact(b: int, c: int) -> tuple:
    n = b + c
    if n == 0:
        return 1.0, 0
    result = binomtest(min(b, c), n, 0.5, alternative="two-sided")
    return result.pvalue, n


def haldane_anscombe_or(b: int, c: int, alpha: float = 0.05) -> tuple:
    b_adj = b + 0.5
    c_adj = c + 0.5
    or_val = b_adj / c_adj
    se_log = np.sqrt(1 / b_adj + 1 / c_adj)
    z = norm.ppf(1 - alpha / 2)
    or_lo = np.exp(np.log(or_val) - z * se_log)
    or_hi = np.exp(np.log(or_val) + z * se_log)
    return or_val, or_lo, or_hi


def expected_calibration_error(confidences, accuracies, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc_bin = accuracies[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return ece


def brier_score(probabilities, outcomes) -> float:
    return np.mean((probabilities - outcomes) ** 2)


def cohen_kappa(confusion_matrix) -> float:
    n = confusion_matrix.sum()
    if n == 0:
        return 0.0
    po = np.trace(confusion_matrix) / n
    pe = (confusion_matrix.sum(axis=0) * confusion_matrix.sum(axis=1)).sum() / n**2
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def acuracia(y_true, y_pred) -> float:
    return (y_true == y_pred).mean()


def sensibilidade_especificidade(y_true, y_pred, pos_label=1):
    tp = ((y_pred == pos_label) & (y_true == pos_label)).sum()
    fn = ((y_pred != pos_label) & (y_true == pos_label)).sum()
    tn = ((y_pred != pos_label) & (y_true != pos_label)).sum()
    fp = ((y_pred == pos_label) & (y_true != pos_label)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    espec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, espec


def f1_score(y_true, y_pred, pos_label=1) -> float:
    tp = ((y_pred == pos_label) & (y_true == pos_label)).sum()
    fp = ((y_pred == pos_label) & (y_true != pos_label)).sum()
    fn = ((y_pred != pos_label) & (y_true == pos_label)).sum()
    if (2 * tp + fp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    print("=" * 60)
    print("TESTES RÁPIDOS — metricas.py")
    print("=" * 60)

    # Wilson
    lo, hi = wilson_ci(164, 189)
    print(f"Wilson IC 95% (164/189): {lo:.2f}% – {hi:.2f}%")

    # McNemar
    p, n = mcnemar_exact(45, 13)
    print(f"McNemar (b=45, c=13): p={p:.4f}, n={n}")

    # Haldane-Anscombe OR
    orv, orlo, orhi = haldane_anscombe_or(45, 13)
    print(f"OR (Haldane-Anscombe): {orv:.2f} (IC 95%: {orlo:.2f}–{orhi:.2f})")

    # ECE
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    acc = np.array([1, 1, 0, 1, 0])
    print(f"ECE: {expected_calibration_error(conf, acc):.4f}")
    print(f"Brier: {brier_score(conf, acc):.4f}")



# """
# Métricas estatísticas para avaliação de LVMs em glaucoma.
#
# Inclui:
# - Intervalo de Confiança de Wilson
# - Teste Exato de McNemar
# - Correção de Haldane-Anscombe
# - Expected Calibration Error (ECE)
# - Brier Score
# - Coeficiente de Correlação Intraclasse (ICC)
# - Kappa de Cohen e Fleiss
# """
# import numpy as np
# from scipy.stats import binomtest, norm
# from statsmodels.stats.proportion import proportion_confint
# import warnings
#
# warnings.filterwarnings("ignore")
#
#
# # ============================================================
# # INTERVALOS DE CONFIANÇA
# # ============================================================
# def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple:
#     """
#     Intervalo de Confiança de Wilson para proporção.
#
#     Args:
#         successes: número de sucessos
#         total: número total de tentativas
#         alpha: nível de significância (default 0.05)
#
#     Returns:
#         (limite_inferior, limite_superior) em percentual
#     """
#     p = successes / total
#     z = norm.ppf(1 - alpha / 2)
#     denom = 1 + z**2 / total
#     center = (p + z**2 / (2 * total)) / denom
#     margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
#     return (center - margin) * 100, (center + margin) * 100
#
#
# # ============================================================
# # MCNEMAR
# # ============================================================
# def mcnemar_exact(b: int, c: int) -> tuple:
#     """
#     Teste Exato de McNemar (binomial bilateral).
#
#     Args:
#         b: LLM correto / comparador errado
#         c: LLM errado / comparador correto
#
#     Returns:
#         (p_valor, n_pares_discordantes)
#     """
#     n = b + c
#     if n == 0:
#         return 1.0, 0
#     result = binomtest(min(b, c), n, 0.5, alternative="two-sided")
#     return result.pvalue, n
#
#
# def haldane_anscombe_or(b: int, c: int, alpha: float = 0.05) -> tuple:
#     """
#     Odds Ratio pareado com correção de Haldane-Anscombe.
#
#     Adiciona 0,5 a cada célula discordante para estabilizar
#     estimativas quando b ou c se aproximam de zero.
#     IC na escala logarítmica (método de Woolf).
#
#     Returns:
#         (or, ic_inferior, ic_superior)
#     """
#     b_adj = b + 0.5
#     c_adj = c + 0.5
#     or_val = b_adj / c_adj
#     se_log = np.sqrt(1 / b_adj + 1 / c_adj)
#     z = norm.ppf(1 - alpha / 2)
#     or_lo = np.exp(np.log(or_val) - z * se_log)
#     or_hi = np.exp(np.log(or_val) + z * se_log)
#     return or_val, or_lo, or_hi
#
#
# # ============================================================
# # CALIBRAÇÃO
# # ============================================================
# def expected_calibration_error(
#     confidences: np.ndarray,
#     accuracies: np.ndarray,
#     n_bins: int = 10,
# ) -> float:
#     """
#     Expected Calibration Error (ECE).
#
#     Args:
#         confidences: confiança prevista pelo modelo (0-1)
#         accuracies: acurácia observada (0-1)
#         n_bins: número de bins
#
#     Returns:
#         ECE (0 = perfeito)
#     """
#     bins = np.linspace(0, 1, n_bins + 1)
#     ece = 0.0
#     n = len(confidences)
#
#     for i in range(n_bins):
#         mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
#         if mask.sum() == 0:
#             continue
#         acc_bin = accuracies[mask].mean()
#         conf_bin = confidences[mask].mean()
#         ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
#
#     return ece
#
#
# def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
#     """
#     Brier Score para classificação binária.
#
#     BS = (1/N) * Σ (p_i - o_i)²
#     """
#     return np.mean((probabilities - outcomes) ** 2)
#
#
# # ============================================================
# # CONCORDÂNCIA
# # ============================================================
# def cohen_kappa(confusion_matrix: np.ndarray) -> float:
#     """
#     Kappa de Cohen para matriz de confusão NxN.
#     """
#     n = confusion_matrix.sum()
#     if n == 0:
#         return 0.0
#     po = np.trace(confusion_matrix) / n
#     pe = (confusion_matrix.sum(axis=0) * confusion_matrix.sum(axis=1)).sum() / n**2
#     if pe == 1:
#         return 1.0
#     return (po - pe) / (1 - pe)
#
#
# def icc_2_1(data: np.ndarray) -> float:
#     """
#     Coeficiente de Correlação Intraclasse ICC(2,1) — two-way random, absolute agreement.
#
#     Args:
#         data: matriz N x K (N sujeitos, K medições)
#
#     Returns:
#         ICC(2,1)
#     """
#     n, k = data.shape
#     grand_mean = data.mean()
#     ms_rows = k * ((data.mean(axis=1) - grand_mean) ** 2).sum() / (n - 1)
#     ms_cols = n * ((data.mean(axis=0) - grand_mean) ** 2).sum() / (k - 1)
#     ms_error = (
#         ((data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + grand_mean) ** 2).sum()
#         / ((n - 1) * (k - 1))
#     )
#     icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
#     return icc
#
#
# # ============================================================
# # ACURÁCIA E MÉTRICAS DE CLASSIFICAÇÃO
# # ============================================================
# def acuracia(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """Acurácia simples."""
#     return (y_true == y_pred).mean()
#
#
# def sensibilidade_especificidade(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1):
#     """
#     Sensibilidade e especificidade para classificação binária.
#     """
#     tp = ((y_pred == pos_label) & (y_true == pos_label)).sum()
#     fn = ((y_pred != pos_label) & (y_true == pos_label)).sum()
#     tn = ((y_pred != pos_label) & (y_true != pos_label)).sum()
#     fp = ((y_pred == pos_label) & (y_true != pos_label)).sum()
#
#     sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     espec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#     return sens, espec
#
#
# def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1) -> float:
#     """F1-score para classificação binária."""
#     tp = ((y_pred == pos_label) & (y_true == pos_label)).sum()
#     fp = ((y_pred == pos_label) & (y_true != pos_label)).sum()
#     fn = ((y_pred != pos_label) & (y_true == pos_label)).sum()
#     if (2 * tp + fp + fn) == 0:
#         return 0.0
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     if (precision + recall) == 0:
#         return 0.0
#     return 2 * precision * recall / (precision + recall)