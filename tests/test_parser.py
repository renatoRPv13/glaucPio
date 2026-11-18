"""
Testes unitários para o parser do GlaucPIO — adaptados ao PAPILA.

Testa reconhecimento das 3 classes:
    - Normal (NON)
    - Glaucoma Suspect (SUSP)
    - Glaucomatous Optic Neuropathy (GON)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from glaucpio.parser import (
    CLASSES_PAPILA,
    SINONIMOS_PAPILA,
    extrair_classificacao,
    extrair_confianca,
    comparar_com_ground_truth,
)


# ============================================================
# CONSTANTES
# ============================================================
GON = "Glaucomatous Optic Neuropathy (GON)"
SUSP = "Glaucoma Suspect (SUSP)"
NON = "Normal (NON)"


# ============================================================
# TESTES — CLASSES
# ============================================================
class TestClasses:
    def test_tres_classes_definidas(self):
        assert len(CLASSES_PAPILA) == 3

    def test_classes_contem_gon(self):
        assert GON in CLASSES_PAPILA

    def test_classes_contem_susp(self):
        assert SUSP in CLASSES_PAPILA

    def test_classes_contem_non(self):
        assert NON in CLASSES_PAPILA


# ============================================================
# TESTES — EXTRAÇÃO DE CLASSIFICAÇÃO
# ============================================================
class TestExtrairClassificacao:
    def test_diagnostico_gon_ingles(self):
        texto = "Diagnosis: Glaucomatous Optic Neuropathy (GON). Confidence: 90%."
        assert extrair_classificacao(texto) == GON

    def test_diagnostico_susp_ingles(self):
        texto = "This is a glaucoma suspect case. Confidence: 0.75"
        assert extrair_classificacao(texto) == SUSP

    def test_diagnostico_non_ingles(self):
        texto = "Normal eye, no signs of glaucoma. Confidence: 95%"
        assert extrair_classificacao(texto) == NON

    def test_sinonimo_poag(self):
        texto = "This is a case of POAG."
        assert extrair_classificacao(texto) == GON

    def test_sinonimo_suspeita(self):
        texto = "Suspeita de glaucoma. Confiança: 70%."
        assert extrair_classificacao(texto) == SUSP

    def test_sinonimo_saudavel(self):
        texto = "Olho saudável. Confiança: 92%."
        assert extrair_classificacao(texto) == NON

    def test_sinonimo_ocular_hypertension(self):
        texto = "Ocular hypertension detected."
        assert extrair_classificacao(texto) == SUSP

    def test_resposta_vazia(self):
        assert extrair_classificacao("") is None

    def test_resposta_none(self):
        assert extrair_classificacao(None) is None

    def test_resposta_sem_classe(self):
        assert extrair_classificacao("Paciente apresenta catarata.") is None


# ============================================================
# TESTES — EXTRAÇÃO DE CONFIANÇA
# ============================================================
class TestExtrairConfianca:
    def test_confianca_porcentagem(self):
        assert extrair_confianca("Confiança: 85%") == 0.85

    def test_confianca_porcentagem_espaco(self):
        assert extrair_confianca("Confidence: 90 %") == 0.90

    def test_confianca_decimal(self):
        assert extrair_confianca("Confidence: 0.75") == 0.75

    def test_confianca_decimal_com_palavra_de(self):
        assert extrair_confianca("Confiança de 0.85") == 0.85

    def test_confianca_ausente(self):
        assert extrair_confianca("Diagnóstico: GON.") is None

    def test_confianca_texto_vazio(self):
        assert extrair_confianca("") is None


# ============================================================
# TESTES — COMPARAÇÃO COM GROUND TRUTH
# ============================================================
class TestComparacaoComGroundTruth:
    def test_acerto_direto(self):
        resposta = "Diagnosis: Glaucomatous Optic Neuropathy (GON). Confidence: 90%."
        resultado = comparar_com_ground_truth(resposta, GON)
        assert resultado["correto"] is True
        assert resultado["classificacao_extraida"] == GON
        assert resultado["confianca"] == 0.90

    def test_erro_classificacao(self):
        resposta = "Diagnosis: Normal (NON)."
        resultado = comparar_com_ground_truth(resposta, GON)
        assert resultado["correto"] is False
        assert resultado["classificacao_extraida"] == NON

    def test_resposta_pt_com_gabarito_en(self):
        resposta = "Este caso é de glaucoma. Confiança: 88%."
        resultado = comparar_com_ground_truth(resposta, GON)
        assert resultado["correto"] is True
        assert resultado["classificacao_extraida"] == GON

    def test_resposta_en_com_gabarito_en(self):
        resposta = "Glaucoma suspect. Confidence: 75%."
        resultado = comparar_com_ground_truth(resposta, SUSP)
        assert resultado["correto"] is True

    def test_resposta_sem_classificacao(self):
        resposta = "Não consigo determinar."
        resultado = comparar_com_ground_truth(resposta, GON)
        assert resultado["correto"] is False
        assert resultado["classificacao_extraida"] is None

    def test_ground_truth_vazio(self):
        resposta = "Diagnosis: GON."
        resultado = comparar_com_ground_truth(resposta, "")
        assert resultado["correto"] is False


# ============================================================
# TESTES — SINÔNIMOS
# ============================================================
class TestSinonimos:
    def test_sinonimos_para_gon(self):
        for termo in ["gon", "glaucoma", "poag", "glaucomatous"]:
            texto = f"Diagnosis: {termo}."
            assert extrair_classificacao(texto) == GON, f"Falha com '{termo}'"

    def test_sinonimos_para_susp(self):
        for termo in ["suspect", "suspeita", "ocular hypertension"]:
            texto = f"Result: {termo}."
            assert extrair_classificacao(texto) == SUSP, f"Falha com '{termo}'"

    def test_sinonimos_para_non(self):
        for termo in ["normal", "healthy", "saudável"]:
            texto = f"Result: {termo}."
            assert extrair_classificacao(texto) == NON, f"Falha com '{termo}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# """
# Parser de respostas dos LVMs — adaptado para PAPILA.
#
# Reconhece as 3 classes do PAPILA:
#     - Normal (NON)
#     - Glaucoma Suspect (SUSP)
#     - Glaucomatous Optic Neuropathy (GON)
# """
# import sys
# from pathlib import Path
#
# if __package__ in (None, ""):
#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#     __package__ = "glaucpio"
#
# import re
# from typing import Optional
#
#
# # ============================================================
# # CLASSES DO PAPILA
# # ============================================================
# CLASSES_PAPILA = [
#     "Glaucomatous Optic Neuropathy (GON)",
#     "Glaucoma Suspect (SUSP)",
#     "Normal (NON)",
# ]
#
# # Sinônimos mapeados para as classes do PAPILA.
# # IMPORTANTE: a ordem importa! Sinônimos mais específicos vêm primeiro.
# # O parser verifica cada chave na ordem de inserção.
# SINONIMOS_PAPILA = {
#     # ============================================================
#     # SUSP — VEM PRIMEIRO (mais específico que GON genérico)
#     # ============================================================
#     "glaucoma suspect": "Glaucoma Suspect (SUSP)",
#     "ocular hypertension": "Glaucoma Suspect (SUSP)",
#     "hipertensão ocular": "Glaucoma Suspect (SUSP)",
#     "suspeita de glaucoma": "Glaucoma Suspect (SUSP)",
#     "suspeita": "Glaucoma Suspect (SUSP)",
#     "suspeito": "Glaucoma Suspect (SUSP)",
#     "susp": "Glaucoma Suspect (SUSP)",
#     "suspect": "Glaucoma Suspect (SUSP)",
#     "suspicious": "Glaucoma Suspect (SUSP)",
#
#     # ============================================================
#     # NON — VEM ANTES DE GON (evitar match de "glaucoma" no NON)
#     # ============================================================
#     "no signs of glaucoma": "Normal (NON)",
#     "no glaucoma": "Normal (NON)",
#     "sem glaucoma": "Normal (NON)",
#     "não glaucomatoso": "Normal (NON)",
#     "normal": "Normal (NON)",
#     "healthy": "Normal (NON)",
#     "saudável": "Normal (NON)",
#     "non": "Normal (NON)",
#
#     # ============================================================
#     # GON — POR ÚLTIMO (mais genérico)
#     # ============================================================
#     "glaucomatous optic neuropathy": "Glaucomatous Optic Neuropathy (GON)",
#     "primary open-angle glaucoma": "Glaucomatous Optic Neuropathy (GON)",
#     "glaucoma de ângulo aberto": "Glaucomatous Optic Neuropathy (GON)",
#     "poag": "Glaucomatous Optic Neuropathy (GON)",
#     "gon": "Glaucomatous Optic Neuropathy (GON)",
#     "glaucoma": "Glaucomatous Optic Neuropathy (GON)",
#     "glaucomatoso": "Glaucomatous Optic Neuropathy (GON)",
# }
#
#
# # ============================================================
# # PARSERS
# # ============================================================
# def extrair_classificacao(resposta: str) -> Optional[str]:
#     """
#     Extrai a classificação diagnóstica da resposta do modelo,
#     mapeando para as 3 classes do PAPILA.
#     """
#     if not resposta:
#         return None
#
#     texto = resposta.lower().strip()
#
#     # 1. Busca direta por classes do PAPILA (case-insensitive)
#     for classe in CLASSES_PAPILA:
#         if classe.lower() in texto:
#             return classe
#
#     # 2. Busca por sinônimos na ORDEM DE INSERÇÃO (mais específicos primeiro)
#     for sinonimo, classe in SINONIMOS_PAPILA.items():
#         if sinonimo in texto:
#             return classe
#
#     # 3. Busca por padrões "diagnóstico: X"
#     match = re.search(
#         r"(?:diagn[óo]stico|diagnosis|classifica[çc][ãa]o|impression)\s*[:\-]\s*([^\n\.]+)",
#         texto,
#     )
#     if match:
#         fragmento = match.group(1).strip()
#         for sinonimo, classe in SINONIMOS_PAPILA.items():
#             if sinonimo in fragmento:
#                 return classe
#
#     return None
#
#
# def extrair_confianca(resposta: str) -> Optional[float]:
#     """
#     Extrai a confiança declarada (0-1 ou 0-100%).
#
#     Aceita:
#         "Confiança: 85%" / "Confidence: 85 %" / "confiança de 0.85"
#         "Confidence: 0.75"
#     """
#     if not resposta:
#         return None
#
#     texto = resposta.lower()
#
#     # 1. Porcentagem (com ou sem espaço antes de %)
#     match = re.search(
#         r"(?:confian[çc]a|confidence)\s*(?:de|:)?\s*(\d{1,3})\s*%",
#         texto,
#     )
#     if match:
#         return float(match.group(1)) / 100.0
#
#     # 2. Decimal (0.75 ou .75)
#     match = re.search(
#         r"(?:confian[çc]a|confidence)\s*(?:de|:)?\s*(\d?\.\d+)",
#         texto,
#     )
#     if match:
#         val = float(match.group(1))
#         return val if val <= 1.0 else val / 100.0
#
#     return None
#
#
# def comparar_com_ground_truth(
#         resposta: str,
#         ground_truth: str,
# ) -> dict:
#     """
#     Compara a classificação extraída da resposta com o ground truth do PAPILA.
#     """
#     extraida = extrair_classificacao(resposta)
#     confianca = extrair_confianca(resposta)
#
#     gt_norm = ground_truth.strip() if ground_truth else ""
#
#     correto = False
#     if extraida and gt_norm:
#         correto = extraida == gt_norm
#
#     return {
#         "classificacao_extraida": extraida,
#         "ground_truth": gt_norm,
#         "correto": correto,
#         "confianca": confianca,
#     }
#
#
# # ============================================================
# # TESTES
# # ============================================================
# if __name__ == "__main__":
#     print("=" * 60)
#     print("TESTES — parser.py (PAPILA)")
#     print("=" * 60)
#
#     exemplos = [
#         ("Diagnosis: Glaucomatous Optic Neuropathy (GON). Confidence: 90%.",
#          "Glaucomatous Optic Neuropathy (GON)"),
#         ("The patient has glaucoma suspect. Confidence: 0.75",
#          "Glaucoma Suspect (SUSP)"),
#         ("Normal eye, no signs of glaucoma. Confidence: 95%",
#          "Normal (NON)"),
#         ("This is a case of POAG. Confidence: 88%.",
#          "Glaucomatous Optic Neuropathy (GON)"),
#         ("Suspeita de glaucoma. Confiança: 70%.",
#          "Glaucoma Suspect (SUSP)"),
#         ("Olho saudável. Confiança: 92%.",
#          "Normal (NON)"),
#     ]
#
#     for texto, gt in exemplos:
#         res = comparar_com_ground_truth(texto, gt)
#         status = "✅" if res["correto"] else "❌"
#         print(f"{status} Texto: {texto[:60]}...")
#         print(f"   → classe: {res['classificacao_extraida']}")
#         print(f"   → confiança: {res['confianca']}")
#         print(f"   → correto: {res['correto']}")
#         print()
#



# """Testes do parser de respostas dos LVMs."""
# import pytest
#
# from glaucpio.parser import (
#     GAA, GAF, SECUNDARIO, CONGENITO, PRESSAO_NORMAL, SUSPEITA, NORMAL, OUTRO,
#     comparar_com_ground_truth,
#     extrair_classificacao,
#     extrair_confianca,
#     normalizar_ground_truth,
# )
#
#
# # ============================================================
# # CLASSIFICAÇÃO
# # ============================================================
# @pytest.mark.parametrize("resposta, esperado", [
#     # Português, com e sem acento
#     ("Diagnóstico: glaucoma primário de ângulo aberto, estágio moderado.", GAA),
#     ("diagnostico: glaucoma de angulo aberto", GAA),
#     ("Impressão: GPAA em OD.", GAA),
#     ("Diagnóstico: glaucoma de ângulo fechado agudo", GAF),
#     ("Diagnóstico: glaucoma pseudoexfoliativo", SECUNDARIO),
#     ("Diagnóstico: glaucoma congênito primário", CONGENITO),
#     ("Diagnóstico: glaucoma de pressão normal", PRESSAO_NORMAL),
#     ("Diagnóstico: suspeita de glaucoma", SUSPEITA),
#     ("Diagnóstico: normal", NORMAL),
#     # Inglês
#     ("1) Diagnosis: Primary Open-Angle Glaucoma (POAG)", GAA),
#     ("Diagnosis: acute primary angle-closure glaucoma", GAF),
#     ("Diagnosis: normal-tension glaucoma", PRESSAO_NORMAL),
#     ("Diagnosis: primary open-angle glaucoma suspect", SUSPEITA),
#     ("Diagnosis: pseudoexfoliation glaucoma", SECUNDARIO),
#     ("Diagnosis: Normal", NORMAL),
#     # Markdown
#     ("**Diagnóstico:** glaucoma de ângulo aberto", GAA),
#     ("**Diagnosis**: POAG, moderate stage", GAA),
# ])
# def test_linha_rotulada(resposta, esperado):
#     assert extrair_classificacao(resposta) == esperado
#
#
# def test_json_tem_prioridade():
#     resposta = (
#         'Análise: o ângulo fechado foi descartado.\n'
#         '```json\n{"diagnostico": "Glaucoma de ângulo aberto", "confianca": 0.8}\n```'
#     )
#     assert extrair_classificacao(resposta) == GAA
#
#
# def test_json_em_ingles():
#     assert extrair_classificacao('{"diagnosis": "angle-closure glaucoma"}') == GAF
#
#
# def test_diagnostico_diferencial_e_ignorado():
#     resposta = (
#         "Diagnóstico diferencial: glaucoma de ângulo fechado\n"
#         "Diagnóstico: glaucoma de ângulo aberto"
#     )
#     assert extrair_classificacao(resposta) == GAA
#
#
# def test_differential_diagnosis_e_ignorado():
#     resposta = (
#         "Differential diagnosis: angle-closure glaucoma\n"
#         "Diagnosis: POAG"
#     )
#     assert extrair_classificacao(resposta) == GAA
#
#
# def test_primeira_mencao_no_texto_vence_ordem_da_lista():
#     # Antes o parser devolvia a primeira classe da lista (ângulo aberto)
#     resposta = "Quadro compatível com glaucoma de ângulo fechado; ângulo aberto é menos provável."
#     assert extrair_classificacao(resposta) == GAF
#
#
# @pytest.mark.parametrize("resposta, esperado", [
#     ("Não é glaucoma de ângulo fechado. Trata-se de glaucoma de ângulo aberto.", GAA),
#     ("Sem sinais de glaucoma de ângulo fechado; achado compatível com GPAA.", GAA),
#     ("This is not angle-closure glaucoma; findings suggest POAG.", GAA),
#     ("There is no angle-closure. The picture is consistent with open-angle glaucoma.", GAA),
# ])
# def test_negacao(resposta, esperado):
#     assert extrair_classificacao(resposta) == esperado
#
#
# def test_no_como_contracao_nao_e_negacao():
#     assert extrair_classificacao("Achado no exame: glaucoma de ângulo aberto.") == GAA
#
#
# @pytest.mark.parametrize("resposta", [
#     "Pressão intraocular normal, córnea normal. Glaucoma de ângulo aberto em OD.",
#     "IOP within normal range; normal cornea. Findings consistent with POAG.",
# ])
# def test_palavra_normal_solta_nao_vira_classe_normal(resposta):
#     # Antes, qualquer "normal" no texto virava a classe "normal"
#     assert extrair_classificacao(resposta) == GAA
#
#
# @pytest.mark.parametrize("resposta", [
#     "Exame sem sinais de glaucoma.",
#     "No evidence of glaucoma in either eye.",
#     "Sem glaucoma.",
# ])
# def test_exame_normal(resposta):
#     assert extrair_classificacao(resposta) == NORMAL
#
#
# def test_diagnostico_fora_das_classes_vira_outro():
#     assert extrair_classificacao("Diagnóstico: neuropatia óptica isquêmica") == OUTRO
#
#
# @pytest.mark.parametrize("resposta", ["", None, "Não foi possível avaliar as imagens."])
# def test_sem_classificacao(resposta):
#     assert extrair_classificacao(resposta) is None
#
#
# # ============================================================
# # CONFIANÇA
# # ============================================================
# @pytest.mark.parametrize("resposta, esperado", [
#     ("Confiança: 85%", 0.85),
#     ("confiança de 70 %", 0.70),
#     ("85% de confiança", 0.85),
#     ("I am 90% confident", 0.90),
#     ("Confidence: 0.75", 0.75),
#     ("Confiança: 0,6", 0.60),
#     ("Confidence level: 80", 0.80),
#     ('{"diagnosis": "POAG", "confidence": 0.9}', 0.90),
#     ('{"diagnostico": "GPAA", "confianca": 85}', 0.85),
#     ('{"diagnostico": "GPAA", "confianca": "70%"}', 0.70),
# ])
# def test_confianca(resposta, esperado):
#     assert extrair_confianca(resposta) == pytest.approx(esperado)
#
#
# @pytest.mark.parametrize("resposta", ["", None, "Diagnóstico: GPAA", "Confiança: 250%"])
# def test_confianca_ausente_ou_invalida(resposta):
#     assert extrair_confianca(resposta) is None
#
#
# # ============================================================
# # GROUND TRUTH E COMPARAÇÃO
# # ============================================================
# @pytest.mark.parametrize("gt, esperado", [
#     ("Primary Open-Angle Glaucoma (POAG) - Moderate Stage", GAA),
#     ("glaucoma de ângulo aberto", GAA),
#     ("Glaucoma de Angulo Aberto", GAA),
#     ("Normal", NORMAL),
#     ("Angle-closure glaucoma", GAF),
#     ("Normal-tension glaucoma", PRESSAO_NORMAL),
# ])
# def test_normalizar_ground_truth(gt, esperado):
#     assert normalizar_ground_truth(gt) == esperado
#
#
# @pytest.mark.parametrize("gt", [None, "", float("nan"), "Diagnóstico indefinido"])
# def test_ground_truth_nao_mapeavel(gt):
#     assert normalizar_ground_truth(gt) is None
#
#
# def test_comparacao_resposta_pt_com_gabarito_en():
#     # Antes: "glaucoma de ângulo aberto" != gabarito cru em inglês -> sempre errado
#     r = comparar_com_ground_truth(
#         "Diagnóstico: glaucoma de ângulo aberto. Confiança: 80%",
#         "Primary Open-Angle Glaucoma (POAG) - Moderate Stage",
#     )
#     assert r == {
#         "classificacao_extraida": GAA,
#         "ground_truth": GAA,
#         "correto": True,
#         "confianca": pytest.approx(0.8),
#     }
#
#
# def test_comparacao_errada():
#     r = comparar_com_ground_truth("Diagnosis: angle-closure glaucoma", "POAG")
#     assert r["correto"] is False
#
#
# def test_resposta_sem_classificacao_conta_como_erro():
#     r = comparar_com_ground_truth("Não sei.", "POAG")
#     assert r["classificacao_extraida"] is None
#     assert r["correto"] is False
#
#
# def test_gabarito_nao_mapeavel_fica_fora_da_avaliacao():
#     r = comparar_com_ground_truth("Diagnóstico: GPAA", "")
#     assert r["correto"] is None
