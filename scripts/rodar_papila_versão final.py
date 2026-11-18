#!/usr/bin/env python3
"""
Avaliação do PAPILA com Gemini 3.6 Flash.

Uso:
    python scripts/rodar_papila.py --max-cases 3
    python scripts/rodar_papila.py  # todos os 488
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from PIL import Image
from google import genai
from google.genai import types

from glaucpio.config import RAW_DIR, CSV_DIR, GOOGLE_API_KEY, MODELOS
from glaucpio.logger import get_logger
from glaucpio.parser import comparar_com_ground_truth

log = get_logger("glaucpio.papila_eval")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliação do PAPILA com Gemini."
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=4.0)
    parser.add_argument(
        "--output",
        type=str,
        default=str(CSV_DIR / "papila_gemini_results.csv"),
    )
    return parser.parse_args()


def avaliar_caso(client, modelo, row, base_dir):
    """Avalia um caso do PAPILA."""
    case_id = row["Case_ID"]
    prompt = row["Multimodal_Prompt"]
    gt = row["Ground_Truth_Diagnosis"]
    img_path = row["Fundus_Image_Path"]

    try:
        imagens = []
        if img_path:
            path = base_dir / img_path
            if path.exists():
                imagens.append(Image.open(path))

        t0 = time.time()
        contents = [prompt]
        if imagens:
            contents.extend(imagens)

        response = client.models.generate_content(
            model=modelo,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        elapsed = time.time() - t0

        texto = response.text if hasattr(response, "text") else ""
        parsed = comparar_com_ground_truth(texto, gt)

        return {
            "Case_ID": case_id,
            "Olho": row["Olho"],
            "Ground_Truth_Diagnosis": gt,
            "Modelo_Resposta": texto,
            "Classificacao_Extraida": parsed["classificacao_extraida"],
            "Correto": parsed["correto"],
            "Confianca_Declarada": parsed["confianca"],
            "Latency_s": round(elapsed, 3),
            "N_Images": len(imagens),
            "Timestamp": datetime.utcnow().isoformat(),
            "Status": "OK",
        }
    except Exception as e:
        log.error(f"[{case_id}] ERRO: {type(e).__name__}: {e}")
        return {
            "Case_ID": case_id,
            "Olho": row["Olho"],
            "Ground_Truth_Diagnosis": gt,
            "Modelo_Resposta": "",
            "Classificacao_Extraida": None,
            "Correto": False,
            "Confianca_Declarada": None,
            "Latency_s": None,
            "N_Images": 0,
            "Timestamp": datetime.utcnow().isoformat(),
            "Status": f"ERROR: {type(e).__name__}",
        }


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("AVALIAÇÃO PAPILA — Gemini 3.6 Flash")
    log.info("=" * 60)

    dataset_path = RAW_DIR / "glaucoma_papila_dataset.csv"
    if not dataset_path.exists():
        log.error(f"Dataset não encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    if args.max_cases:
        df = df.head(args.max_cases)
    log.info(f"Casos a avaliar: {len(df)}")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    modelo = MODELOS["gemini"]["nome"]
    log.info(f"Modelo: {modelo}")

    base_dir = Path(__file__).resolve().parent.parent
    resultados = []
    output_path = Path(args.output)

    for idx, row in df.iterrows():
        r = avaliar_caso(client, modelo, row, base_dir)
        resultados.append(r)

        log.info(
            f"[{idx+1}/{len(df)}] {r['Case_ID']} | "
            f"{r['Latency_s']}s | "
            f"correto={r['Correto']} | "
            f"imgs={r['N_Images']}"
        )

        if (idx + 1) % 10 == 0:
            pd.DataFrame(resultados).to_csv(output_path, index=False)

        time.sleep(args.sleep)

    pd.DataFrame(resultados).to_csv(output_path, index=False)
    log.info(f"→ Resultados salvos: {output_path}")

    # Estatísticas
    df_res = pd.DataFrame(resultados)
    ok = df_res[df_res["Status"] == "OK"]
    if len(ok) > 0:
        acc = ok["Correto"].mean()
        log.info("-" * 60)
        log.info(f"Acurácia: {acc:.3f} ({int(acc*len(ok))}/{len(ok)})")
        log.info(f"Latência média: {ok['Latency_s'].mean():.2f}s")

        log.info("-" * 60)
        log.info("Acurácia por diagnóstico:")
        for diag, grupo in ok.groupby("Ground_Truth_Diagnosis"):
            acc_g = grupo["Correto"].mean()
            log.info(f"  {diag}: {acc_g:.3f} ({len(grupo)} casos)")

    log.info("=" * 60)
    log.info("✅ Avaliação concluída.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()