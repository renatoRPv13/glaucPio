"""
Parser de respostas dos LVMs — adaptado para PAPILA.

Reconhece as 3 classes do PAPILA:
    - Normal (NON)
    - Glaucoma Suspect (SUSP)
    - Glaucomatous Optic Neuropathy (GON)
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "glaucpio"

import re
from typing import Optional


# ============================================================
# CLASSES DO PAPILA
# ============================================================
CLASSES_PAPILA = [
    "Glaucomatous Optic Neuropathy (GON)",
    "Glaucoma Suspect (SUSP)",
    "Normal (NON)",
]

# Sinônimos mapeados para as classes do PAPILA
SINONIMOS_PAPILA = {
    # GON
    "gon": "Glaucomatous Optic Neuropathy (GON)",
    "glaucomatous optic neuropathy": "Glaucomatous Optic Neuropathy (GON)",
    "glaucoma": "Glaucomatous Optic Neuropathy (GON)",
    "glaucomatoso": "Glaucomatous Optic Neuropathy (GON)",
    "poag": "Glaucomatous Optic Neuropathy (GON)",
    "primary open-angle glaucoma": "Glaucomatous Optic Neuropathy (GON)",
    "glaucoma de ângulo aberto": "Glaucomatous Optic Neuropathy (GON)",

    # SUSP
    "susp": "Glaucoma Suspect (SUSP)",
    "suspect": "Glaucoma Suspect (SUSP)",
    "suspicious": "Glaucoma Suspect (SUSP)",
    "suspeita": "Glaucoma Suspect (SUSP)",
    "suspeito": "Glaucoma Suspect (SUSP)",
    "glaucoma suspect": "Glaucoma Suspect (SUSP)",
    "ocular hypertension": "Glaucoma Suspect (SUSP)",
    "hipertensão ocular": "Glaucoma Suspect (SUSP)",

    # NON
    "non": "Normal (NON)",
    "normal": "Normal (NON)",
    "healthy": "Normal (NON)",
    "saudável": "Normal (NON)",
    "no glaucoma": "Normal (NON)",
    "sem glaucoma": "Normal (NON)",
    "não glaucomatoso": "Normal (NON)",
}


# ============================================================
# PARSERS
# ============================================================
def extrair_classificacao(resposta: str) -> Optional[str]:
    """
    Extrai a classificação diagnóstica da resposta do modelo,
    mapeando para as 3 classes do PAPILA.
    """
    if not resposta:
        return None

    texto = resposta.lower().strip()

    # 1. Busca direta por classes do PAPILA (case-insensitive)
    for classe in CLASSES_PAPILA:
        if classe.lower() in texto:
            return classe

    # 2. Busca por sinônimos (ordenar por tamanho decrescente para
    #    evitar matches curtos antes de longos)
    for sinonimo in sorted(SINONIMOS_PAPILA.keys(), key=len, reverse=True):
        if sinonimo in texto:
            return SINONIMOS_PAPILA[sinonimo]

    # 3. Busca por padrões "diagnóstico: X"
    match = re.search(
        r"(?:diagn[óo]stico|diagnosis|classifica[çc][ãa]o|impression)\s*[:\-]\s*([^\n\.]+)",
        texto,
    )
    if match:
        fragmento = match.group(1).strip()
        for sinonimo, classe in SINONIMOS_PAPILA.items():
            if sinonimo in fragmento:
                return classe

    return None


def extrair_confianca(resposta: str) -> Optional[float]:
    """
    Extrai a confiança declarada (0-1 ou 0-100%).
    """
    if not resposta:
        return None

    texto = resposta.lower()

    match = re.search(r"confian[çc]a\s*(?:de|:)?\s*(\d{1,3})\s*%", texto)
    if match:
        return float(match.group(1)) / 100.0

    match = re.search(r"confian[çc]a\s*(?:de|:)?\s*(0?\.\d+|\d+\.\d+)", texto)
    if match:
        val = float(match.group(1))
        return val if val <= 1.0 else val / 100.0

    return None


def comparar_com_ground_truth(
        resposta: str,
        ground_truth: str,
) -> dict:
    """
    Compara a classificação extraída da resposta com o ground truth do PAPILA.
    """
    extraida = extrair_classificacao(resposta)
    confianca = extrair_confianca(resposta)

    gt_norm = ground_truth.strip() if ground_truth else ""

    correto = False
    if extraida and gt_norm:
        correto = extraida == gt_norm

    return {
        "classificacao_extraida": extraida,
        "ground_truth": gt_norm,
        "correto": correto,
        "confianca": confianca,
    }


# ============================================================
# TESTES
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTES — parser.py (PAPILA)")
    print("=" * 60)

    exemplos = [
        ("Diagnosis: Glaucomatous Optic Neuropathy (GON). Confidence: 90%.",
         "Glaucomatous Optic Neuropathy (GON)"),
        ("The patient has glaucoma suspect. Confidence: 0.75",
         "Glaucoma Suspect (SUSP)"),
        ("Normal eye, no signs of glaucoma. Confidence: 95%",
         "Normal (NON)"),
        ("This is a case of POAG. Confidence: 88%.",
         "Glaucomatous Optic Neuropathy (GON)"),
        ("Suspeita de glaucoma. Confiança: 70%.",
         "Glaucoma Suspect (SUSP)"),
        ("Olho saudável. Confiança: 92%.",
         "Normal (NON)"),
    ]

    for texto, gt in exemplos:
        res = comparar_com_ground_truth(texto, gt)
        status = "✅" if res["correto"] else "❌"
        print(f"{status} Texto: {texto[:60]}...")
        print(f"   → classe: {res['classificacao_extraida']}")
        print(f"   → confiança: {res['confianca']}")
        print(f"   → correto: {res['correto']}")
        print()



# """
# Parser de respostas dos LVMs.
#
# Extrai a classificação diagnóstica e a confiança declarada
# das respostas em texto livre (português ou inglês).
#
# Ordem de extração da classificação:
#     1. Bloco JSON com chave "diagnostico"/"diagnosis"
#     2. Linha rotulada ("Diagnóstico: ...", "Diagnosis: ...", "Impressão: ...")
#     3. Primeira menção não negada no texto completo
#
# O ground truth passa pela mesma normalização, para que a comparação
# seja feita sempre entre classes canônicas.
# """
# import json
# import re
# import unicodedata
# from typing import Optional
#
# # Classes de glaucoma esperadas (forma canônica)
# GAA = "glaucoma de ângulo aberto"
# GAF = "glaucoma de ângulo fechado"
# SECUNDARIO = "glaucoma secundário"
# CONGENITO = "glaucoma congênito"
# PRESSAO_NORMAL = "glaucoma de pressão normal"
# SUSPEITA = "suspeita de glaucoma"
# NORMAL = "normal"
# OUTRO = "outro"
#
# CLASSES_VALIDAS = [GAA, GAF, SECUNDARIO, CONGENITO, PRESSAO_NORMAL, SUSPEITA, NORMAL, OUTRO]
#
# _SEM_SUBTIPO = (
#     r"(?!\s*(?:de (?:angulo|pressao)|primario|secundario|congenito|"
#     r"pseudo|pigmentar|neovascular|uveitico|cortisonico))"
# )
#
# # Padrões aplicados ao texto em minúsculas e sem acentos.
# # Os prefixos opcionais em SUSPEITA fazem "primary open-angle glaucoma suspect"
# # começar antes de "open-angle" e, portanto, vencer como menção mais à esquerda.
# PADROES = {
#     SUSPEITA: [
#         r"suspeit[ao] de glaucoma",
#         r"(?:primary )?(?:open[- ]angle |angle[- ]closure )?glaucoma suspect",
#         r"(?:primary )?angle[- ]closure suspect",
#         r"suspicious (?:for|of) glaucoma",
#     ],
#     PRESSAO_NORMAL: [
#         r"pressao normal",
#         r"normal[- ]tension",
#         r"normal[- ]pressure glaucoma",
#         r"low[- ]tension glaucoma",
#         r"\bntg\b",
#         r"\bgpn\b",
#     ],
#     GAA: [
#         r"angulo aberto",
#         r"open[- ]angle",
#         r"\bpoag\b",
#         r"\bgpaa\b",
#         r"\bgaa\b",
#     ],
#     GAF: [
#         r"angulo fechado",
#         r"fechamento angular",
#         r"angle[- ]closure",
#         r"closed[- ]angle",
#         r"\bpacg\b",
#         r"\bgpaf\b",
#         r"\bgaf\b",
#     ],
#     SECUNDARIO: [
#         r"glaucoma secundario",
#         r"secondary glaucoma",
#         r"glaucoma (?:pseudo-?e[xs]foliativo|pigmentar|neovascular|uveitico|cortisonico)",
#         r"(?:pseudo-?exfoliat\w*|pigmentary|neovascular|uveitic|steroid[- ]induced) glaucoma",
#     ],
#     CONGENITO: [
#         r"congenit",
#         r"glaucoma infantil",
#         r"childhood glaucoma",
#         r"\bpcg\b",
#     ],
#     # Em português o subtipo vem depois ("sem glaucoma de ângulo fechado"),
#     # e aí a frase nega só o subtipo, não indica exame normal.
#     NORMAL: [
#         rf"sem (?:sinais|evidencias?) de glaucoma{_SEM_SUBTIPO}",
#         rf"sem glaucoma{_SEM_SUBTIPO}",
#         rf"ausencia de glaucoma{_SEM_SUBTIPO}",
#         r"no (?:signs|evidence) of glaucoma",
#         r"\bno glaucoma\b",
#         r"olho normal",
#         r"exame normal",
#         r"(?:healthy|normal) (?:eye|exam|study|optic (?:disc|nerve))",
#         r"\bsaudavel\b",
#     ],
# }
# _PADROES_COMPILADOS = {
#     classe: [re.compile(p) for p in padroes] for classe, padroes in PADROES.items()
# }
#
# # Negação: palavras que anulam a menção quando aparecem logo antes dela.
# # "no" só conta como negação imediatamente antes, pois em português
# # também é contração ("no olho direito").
# _NEGACAO_JANELA = re.compile(
#     r"\b(?:nao|sem|descart\w*|exclu\w*|not|without|unlikely|improvavel|ausencia de)\b"
# )
# _NEGACAO_IMEDIATA = re.compile(r"\bno\s*$")
# # 6 palavras cobrem "sem sinais de glaucoma de | ângulo fechado"; a janela
# # nunca atravessa o fim da frase (ver _negado).
# _JANELA_PALAVRAS = 6
#
# # Linha rotulada. "Diagnóstico diferencial" / "differential diagnosis" são ignorados.
# _ROTULO = re.compile(
#     r"(?<!differential )"
#     r"\b(?:diagnostico|diagnosis|classificacao|classification|impressao|impression)"
#     r"(?:\s+(?:final|principal|primary|provavel|most likely|mais provavel))?"
#     r"[\s*_]*(?::|\s[-–]\s)\s*[*_]*\s*(?P<valor>[^\n]+)"
# )
#
# _CHAVES_DIAGNOSTICO = ["diagnostico", "diagnosis", "classificacao", "classification"]
# _CHAVES_CONFIANCA = ["confianca", "confidence"]
#
#
# def normalizar_texto(texto: str) -> str:
#     """Minúsculas, sem acentos e com espaços colapsados."""
#     sem_acento = unicodedata.normalize("NFKD", texto)
#     sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
#     return re.sub(r"[ \t]+", " ", sem_acento.lower())
#
#
# def _negado(texto: str, inicio: int) -> bool:
#     """Verifica se a menção que começa em `inicio` está negada."""
#     anterior = re.split(r"[.;\n]", texto[:inicio])[-1]
#     if _NEGACAO_IMEDIATA.search(anterior):
#         return True
#     ultimas = " ".join(anterior.split()[-_JANELA_PALAVRAS:])
#     return bool(_NEGACAO_JANELA.search(ultimas))
#
#
# def _primeira_mencao(texto: str) -> Optional[str]:
#     """
#     Retorna a classe mencionada mais à esquerda (não negada).
#     Empates na mesma posição são resolvidos pela menção mais longa.
#     """
#     melhor = None  # (inicio, -comprimento, classe)
#     for classe, padroes in _PADROES_COMPILADOS.items():
#         for padrao in padroes:
#             for m in padrao.finditer(texto):
#                 # A negação faz parte dos próprios padrões de NORMAL ("sem glaucoma")
#                 if classe != NORMAL and _negado(texto, m.start()):
#                     continue
#                 candidato = (m.start(), -(m.end() - m.start()), classe)
#                 if melhor is None or candidato < melhor:
#                     melhor = candidato
#                 break  # só a primeira ocorrência não negada de cada padrão
#     return melhor[2] if melhor else None
#
#
# def _classificar_valor(valor: str) -> str:
#     """Classifica o conteúdo de um campo de diagnóstico explícito."""
#     valor = normalizar_texto(valor).strip(" *_\"'.")
#     classe = _primeira_mencao(valor)
#     if classe:
#         return classe
#     if re.match(r"(?:normal|healthy|saudavel)\b", valor):
#         return NORMAL
#     # O modelo deu um diagnóstico, mas fora das classes previstas
#     return OUTRO
#
#
# def _extrair_json(resposta: str) -> Optional[dict]:
#     """Tenta extrair o primeiro objeto JSON da resposta."""
#     candidatos = []
#
#     # Extrai blocos JSON dentro de markdown fences (```json ... ``` ou ``` ... ```)
#     for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", resposta, re.DOTALL):
#         candidatos.append(m.group(1).strip())
#
#     ini, fim = resposta.find("{"), resposta.rfind("}")
#     if ini != -1 and fim > ini:
#         candidatos.append(resposta[ini:fim + 1])
#     candidatos.extend(re.findall(r"\{[^{}]*\}", resposta))
#
#     for bloco in candidatos:
#         try:
#             obj = json.loads(bloco)
#         except json.JSONDecodeError:
#             continue
#         if isinstance(obj, dict):
#             return {normalizar_texto(str(k)).strip(): v for k, v in obj.items()}
#     return None
#
#
# def extrair_classificacao(resposta: str) -> Optional[str]:
#     """
#     Extrai a classificação diagnóstica da resposta do modelo.
#
#     Args:
#         resposta: texto bruto do modelo
#
#     Returns:
#         Uma das CLASSES_VALIDAS, ou None se nenhuma classificação foi encontrada
#     """
#     if not resposta:
#         return None
#
#     # 1. JSON estruturado
#     obj = _extrair_json(resposta)
#     if obj:
#         for chave in _CHAVES_DIAGNOSTICO:
#             if isinstance(obj.get(chave), str) and obj[chave].strip():
#                 return _classificar_valor(obj[chave])
#
#     texto = normalizar_texto(resposta)
#
#     # 2. Linha rotulada ("Diagnóstico: ...")
#     m = _ROTULO.search(texto)
#     if m:
#         return _classificar_valor(m.group("valor"))
#
#     # 3. Primeira menção não negada no texto completo
#     return _primeira_mencao(texto)
#
#
# def _para_probabilidade(valor: str, percentual: bool) -> Optional[float]:
#     val = float(valor.replace(",", "."))
#     if percentual or val > 1.0:
#         val /= 100.0
#     return val if 0.0 <= val <= 1.0 else None
#
#
# def extrair_confianca(resposta: str) -> Optional[float]:
#     """
#     Extrai a confiança declarada, normalizada para 0-1.
#
#     Reconhece "confiança: 85%", "85% de confiança", "confidence: 0.85",
#     "confiança de 0,85", "I am 85% confident" e o campo "confianca"/"confidence" em JSON.
#     """
#     if not resposta:
#         return None
#
#     obj = _extrair_json(resposta)
#     if obj:
#         for chave in _CHAVES_CONFIANCA:
#             valor = obj.get(chave)
#             if isinstance(valor, (int, float)) and not isinstance(valor, bool):
#                 return _para_probabilidade(str(valor), percentual=False)
#             if isinstance(valor, str):
#                 m = re.search(r"(\d+(?:[.,]\d+)?)\s*(%)?", valor)
#                 if m:
#                     return _para_probabilidade(m.group(1), percentual=bool(m.group(2)))
#
#     texto = normalizar_texto(resposta)
#     numero = r"(\d{1,3}(?:[.,]\d+)?)"
#
#     # "confiança: 85%" / "confidence of 85 %"
#     m = re.search(rf"(?:confianca|confidence)[^\d\n]{{0,25}}?{numero}\s*%", texto)
#     if m:
#         return _para_probabilidade(m.group(1), percentual=True)
#
#     # "85% de confiança" / "85% confident"
#     m = re.search(rf"{numero}\s*%\s*(?:de\s+)?(?:confianca|confidence|confident)", texto)
#     if m:
#         return _para_probabilidade(m.group(1), percentual=True)
#
#     # "confiança: 0.85" / "confidence: 0,85"
#     m = re.search(rf"(?:confianca|confidence)[^\d\n]{{0,25}}?{numero}(?!\s*[%\d/])", texto)
#     if m:
#         return _para_probabilidade(m.group(1), percentual=False)
#
#     return None
#
#
# def normalizar_ground_truth(ground_truth: Optional[str]) -> Optional[str]:
#     """
#     Converte o diagnóstico de referência do dataset para uma classe canônica.
#
#     Ex.: "Primary Open-Angle Glaucoma (POAG) - Moderate Stage"
#          -> "glaucoma de ângulo aberto"
#     """
#     if not isinstance(ground_truth, str) or not ground_truth.strip():
#         return None
#
#     gt = normalizar_texto(ground_truth).strip()
#     for classe in CLASSES_VALIDAS:
#         if gt == normalizar_texto(classe):
#             return classe
#
#     classe = _primeira_mencao(gt)
#     if classe:
#         return classe
#     if re.match(r"(?:normal|healthy|saudavel)\b", gt):
#         return NORMAL
#     return None
#
#
# def comparar_com_ground_truth(
#     resposta: str,
#     ground_truth: str,
# ) -> dict:
#     """
#     Compara a classificação extraída da resposta com o ground truth.
#
#     Returns:
#         dict com:
#         - classificacao_extraida: classe da resposta (None se não encontrada)
#         - ground_truth: classe canônica do gabarito (None se não mapeável)
#         - correto: True/False; None se o gabarito não pôde ser mapeado
#         - confianca: 0-1 ou None
#     """
#     extraida = extrair_classificacao(resposta)
#     gt_classe = normalizar_ground_truth(ground_truth)
#
#     # Resposta sem classificação conta como erro; gabarito não mapeável
#     # não pode ser avaliado e fica de fora da acurácia.
#     correto = None if gt_classe is None else extraida == gt_classe
#
#     return {
#         "classificacao_extraida": extraida,
#         "ground_truth": gt_classe,
#         "correto": correto,
#         "confianca": extrair_confianca(resposta),
#     }
