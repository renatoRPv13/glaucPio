"""
Pipeline de avaliação multimodal do Projeto GlaucPIO..
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "glaucpio"

import json
import time
from datetime import datetime
from typing import Optional

import pandas as pd
from PIL import Image

from .config import (
    DATASET_MULTIMODAL, OUTPUT_AVALIACAO_CSV, OUTPUT_AVALIACAO_JSONL,
    BASE_DIR, IMAGES_DIR, SLEEP_BETWEEN_CALLS, CHECKPOINT_EVERY, MAX_CASES,
)
from .logger import get_logger
from .modelos import ModeloLVM
from .parser import comparar_com_ground_truth

log = get_logger(__name__)


def carregar_imagens(row: pd.Series) -> list[Image.Image]:
    imagens = []
    for col in ["OCT_Image_Path", "VF_Image_Path", "Fundus_Image_Path"]:
        if col in row and pd.notna(row[col]):
            raw = str(row[col]).strip()
            if not raw:
                continue
            raw_path = Path(raw).expanduser()
            candidatos = [raw_path] if raw_path.is_absolute() else [
                raw_path,
                BASE_DIR / raw_path,
            ]
            partes = raw_path.parts
            if partes[:2] == ("dataset", "images"):
                candidatos.append(IMAGES_DIR.joinpath(*partes[2:]))
            elif partes and partes[0] == "images":
                candidatos.append(IMAGES_DIR.joinpath(*partes[1:]))
            else:
                candidatos.append(IMAGES_DIR / raw_path)

            vistos = set()
            for path in candidatos:
                path = path.resolve()
                if path in vistos:
                    continue
                vistos.add(path)
                if not path.is_file():
                    continue
                try:
                    with Image.open(path) as imagem:
                        imagem.load()
                        imagens.append(imagem.copy())
                    break
                except (OSError, ValueError) as e:
                    log.warning(f"Falha ao abrir {path}: {e}")
            else:
                log.warning(f"Imagem não encontrada: {raw}")
    return imagens


def avaliar_caso(row: pd.Series, modelo: ModeloLVM) -> dict:
    case_id = str(row["Case_ID"])
    prompt = row["Multimodal_Prompt"]
    gt = row.get("Ground_Truth_Diagnosis", "")
    imagens = []

    try:
        imagens = carregar_imagens(row)
        if not imagens:
            raise FileNotFoundError("Nenhuma imagem válida foi encontrada para este caso.")
        resultado = modelo.chamar(prompt, imagens)
        parsed = comparar_com_ground_truth(resultado["text"], gt)

        return {
            "Case_ID": case_id,
            "Ground_Truth_Diagnosis": gt,
            "Modelo_Resposta": resultado["text"],
            "Classificacao_Extraida": parsed["classificacao_extraida"],
            "Correto": parsed["correto"],
            "Confianca_Declarada": parsed["confianca"],
            "Latency_s": resultado["latency_s"],
            "Input_Tokens": resultado["input_tokens"],
            "Output_Tokens": resultado["output_tokens"],
            "N_Images": len(imagens),
            "Timestamp": datetime.now().astimezone().isoformat(),
            "Status": "OK",
            "Prompt": prompt,
        }
    except Exception as e:
        log.error(f"[{case_id}] ERRO: {type(e).__name__}: {e}")
        return {
            "Case_ID": case_id,
            "Ground_Truth_Diagnosis": gt,
            "Modelo_Resposta": "",
            "Classificacao_Extraida": None,
            "Correto": None,
            "Confianca_Declarada": None,
            "Latency_s": None,
            "Input_Tokens": None,
            "Output_Tokens": None,
            "N_Images": len(imagens),
            "Timestamp": datetime.now().astimezone().isoformat(),
            "Status": f"ERROR: {type(e).__name__}: {e}",
            "Prompt": prompt,
        }
    finally:
        for imagem in imagens:
            imagem.close()


def _salvar(results, raw_records, output_csv, output_jsonl):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def avaliar_dataset(
        modelo: ModeloLVM,
        dataset_path: Path = DATASET_MULTIMODAL,
        output_csv: Path = OUTPUT_AVALIACAO_CSV,
        output_jsonl: Path = OUTPUT_AVALIACAO_JSONL,
        max_cases: Optional[int] = MAX_CASES,
        sleep: float = SLEEP_BETWEEN_CALLS,
) -> pd.DataFrame:
    log.info(f"Carregando dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    if max_cases:
        df = df.head(max_cases)
    log.info(f"Total de casos: {len(df)}")

    processados = set()
    results = []
    if output_csv.exists():
        df_existente = pd.read_csv(output_csv)
        if {"Status", "N_Images"}.issubset(df_existente.columns):
            imagens_validas = pd.to_numeric(
                df_existente["N_Images"], errors="coerce"
            ).fillna(0).gt(0)
            df_existente = df_existente[
                df_existente["Status"].eq("OK") & imagens_validas
            ]
        else:
            df_existente = df_existente.iloc[0:0]
        processados = set(df_existente["Case_ID"].astype(str))
        results = df_existente.to_dict("records")
        log.info(f"Retomando: {len(processados)} casos já processados.")

    raw_records = list(results)

    for idx, row in df.iterrows():
        case_id = str(row["Case_ID"])
        if case_id in processados:
            continue

        record = avaliar_caso(row, modelo)
        results.append(record)
        raw_records.append(record)

        log.info(
            f"[{idx+1}/{len(df)}] {case_id} | "
            f"{record['Latency_s']}s | "
            f"correto={record['Correto']} | "
            f"imgs={record['N_Images']}"
        )

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            _salvar(results, raw_records, output_csv, output_jsonl)
            log.info(f"Checkpoint salvo ({len(results)} casos).")

        time.sleep(sleep)

    _salvar(results, raw_records, output_csv, output_jsonl)
    log.info(f"✅ Avaliação concluída: {len(results)} casos.")
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("AVALIACAO.PY — Módulo carregado com sucesso")
    print("=" * 60)
    print("Use scripts/run_avaliacao_multimodal.py para executar.")


# """
# Pipeline de avaliação multimodal do Projeto GlaucPIO.
# """
# import json
# import sys
# import time
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Optional
#
# import pandas as pd
# from PIL import Image
#
# # Permite executar este arquivo diretamente, além de `python -m glaucpio.avaliacao`.
# if __package__ in (None, ""):
#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#     __package__ = "glaucpio"
#
# from .config import (
#     DATASET_MULTIMODAL, IMAGES_DIR, SLEEP_BETWEEN_CALLS, CHECKPOINT_EVERY,
#     MAX_CASES, caminhos_saida,
# )
# from .logger import get_logger
# from .modelos import ModeloLVM
# from .parser import comparar_com_ground_truth
#
# log = get_logger(__name__)
#
# STATUS_OK = "OK"
#
#
# def carregar_imagens(row: pd.Series) -> list:
#     """
#     Carrega imagens do caso (OCT, campo visual, fundo).
#     Resolve caminhos relativos ao diretório de imagens.
#     """
#     imagens = []
#     for col in ["OCT_Image_Path", "VF_Image_Path", "Fundus_Image_Path"]:
#         if col in row and pd.notna(row[col]):
#             raw = str(row[col]).strip()
#             # Tenta caminhos em diferentes bases
#             candidatos = [
#                 Path(raw),
#                 IMAGES_DIR.parent / raw,
#                 IMAGES_DIR / raw.removeprefix("dataset/images/"),
#             ]
#             for path in candidatos:
#                 if path.exists():
#                     try:
#                         imagens.append(Image.open(path))
#                         break
#                     except Exception as e:
#                         log.warning(f"Falha ao abrir {path}: {e}")
#             else:
#                 log.warning(f"Imagem não encontrada: {raw}")
#     return imagens
#
#
# def _agora() -> str:
#     return datetime.now(timezone.utc).isoformat()
#
#
# def avaliar_caso(
#     row: pd.Series,
#     modelo: ModeloLVM,
# ) -> dict:
#     """
#     Avalia um único caso e retorna dicionário de resultados.
#
#     Falhas de API/execução recebem Status "ERROR: ..." e Correto=None,
#     para não serem contadas como respostas erradas do modelo.
#     """
#     case_id = str(row["Case_ID"])
#     prompt = row["Multimodal_Prompt"]
#     gt = row.get("Ground_Truth_Diagnosis", "")
#
#     try:
#         imagens = carregar_imagens(row)
#         resultado = modelo.chamar(prompt, imagens)
#
#         # Parser: extrai classificação e compara com ground truth
#         parsed = comparar_com_ground_truth(resultado["text"], gt)
#
#         return {
#             "Case_ID": case_id,
#             "Modelo": modelo.nome,
#             "Ground_Truth_Diagnosis": gt,
#             "Ground_Truth_Classe": parsed["ground_truth"],
#             "Modelo_Resposta": resultado["text"],
#             "Classificacao_Extraida": parsed["classificacao_extraida"],
#             "Correto": parsed["correto"],
#             "Confianca_Declarada": parsed["confianca"],
#             "Latency_s": resultado["latency_s"],
#             "Input_Tokens": resultado["input_tokens"],
#             "Output_Tokens": resultado["output_tokens"],
#             "N_Images": len(imagens),
#             "Timestamp": _agora(),
#             "Status": STATUS_OK,
#             "Prompt": prompt,
#         }
#     except Exception as e:
#         log.error(f"[{case_id}] ERRO: {type(e).__name__}: {e}")
#         return {
#             "Case_ID": case_id,
#             "Modelo": modelo.nome,
#             "Ground_Truth_Diagnosis": gt,
#             "Ground_Truth_Classe": None,
#             "Modelo_Resposta": "",
#             "Classificacao_Extraida": None,
#             "Correto": None,
#             "Confianca_Declarada": None,
#             "Latency_s": None,
#             "Input_Tokens": None,
#             "Output_Tokens": None,
#             "N_Images": 0,
#             "Timestamp": _agora(),
#             "Status": f"ERROR: {type(e).__name__}",
#             "Prompt": prompt,
#         }
#
#
# def avaliar_dataset(
#     modelo: ModeloLVM,
#     dataset_path: Path = DATASET_MULTIMODAL,
#     output_csv: Optional[Path] = None,
#     output_jsonl: Optional[Path] = None,
#     max_cases: Optional[int] = MAX_CASES,
#     sleep: float = SLEEP_BETWEEN_CALLS,
# ) -> pd.DataFrame:
#     """
#     Avalia todos os casos do dataset com retomada automática.
#
#     Os arquivos de saída são separados por modelo (ver config.caminhos_saida).
#     Na retomada, apenas casos com Status OK são pulados; casos com erro
#     são reavaliados.
#     """
#     csv_padrao, jsonl_padrao = caminhos_saida(modelo.nome)
#     output_csv = output_csv or csv_padrao
#     output_jsonl = output_jsonl or jsonl_padrao
#
#     log.info(f"Modelo: {modelo.nome} | saída: {output_csv}")
#     log.info(f"Carregando dataset: {dataset_path}")
#     df = pd.read_csv(dataset_path)
#     if max_cases:
#         df = df.head(max_cases)
#     log.info(f"Total de casos: {len(df)}")
#
#     # Retomada: mantém só os casos OK; os com erro serão refeitos
#     results = []
#     if output_csv.exists():
#         df_existente = pd.read_csv(output_csv)
#         df_ok = df_existente[df_existente["Status"] == STATUS_OK]
#         results = df_ok.to_dict("records")
#         n_erros = len(df_existente) - len(df_ok)
#         log.info(
#             f"Retomando: {len(df_ok)} casos OK já processados"
#             + (f", {n_erros} com erro serão refeitos." if n_erros else ".")
#         )
#     processados = {str(r["Case_ID"]) for r in results}
#
#     novos = 0
#     for idx, row in df.iterrows():
#         case_id = str(row["Case_ID"])
#         if case_id in processados:
#             continue
#
#         record = avaliar_caso(row, modelo)
#         results.append(record)
#         novos += 1
#
#         log.info(
#             f"[{idx+1}/{len(df)}] {case_id} | "
#             f"{record['Latency_s'] if record['Latency_s'] is not None else '-'}s | "
#             f"status={record['Status']} | "
#             f"correto={record['Correto']} | "
#             f"imgs={record['N_Images']}"
#         )
#
#         # Checkpoint
#         if novos % CHECKPOINT_EVERY == 0:
#             _salvar(results, output_csv, output_jsonl)
#             log.info(f"Checkpoint salvo ({len(results)} casos).")
#
#         time.sleep(sleep)
#
#     _salvar(results, output_csv, output_jsonl)
#     df_res = pd.DataFrame(results)
#     _resumo(df_res)
#     return df_res
#
#
# def _resumo(df: pd.DataFrame) -> None:
#     """Loga acurácia apenas sobre casos OK com gabarito mapeável."""
#     if df.empty:
#         log.info("Nenhum caso avaliado.")
#         return
#     ok = df[df["Status"] == STATUS_OK]
#     avaliaveis = ok[ok["Correto"].notna()]
#     n_erros = len(df) - len(ok)
#     n_sem_gt = len(ok) - len(avaliaveis)
#
#     log.info(f"✅ Avaliação concluída: {len(df)} casos ({len(ok)} OK, {n_erros} com erro).")
#     if n_sem_gt:
#         log.warning(f"{n_sem_gt} casos com gabarito não mapeável foram excluídos da acurácia.")
#     if len(avaliaveis):
#         acertos = int(avaliaveis["Correto"].astype(bool).sum())
#         log.info(
#             f"Acurácia: {acertos}/{len(avaliaveis)} = "
#             f"{100 * acertos / len(avaliaveis):.1f}%"
#         )
#
#
# def _salvar(results, output_csv, output_jsonl):
#     """Salva CSV e JSONL."""
#     df = pd.DataFrame(results)
#     df.to_csv(output_csv, index=False)
#
#     with open(output_jsonl, "w", encoding="utf-8") as f:
#         for r in results:
#             f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
