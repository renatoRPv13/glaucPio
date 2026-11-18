#!/usr/bin/env python3
"""
Integra o dataset PAPILA ao pipeline do GlaucPIO.

Lê ambos os olhos (OD e OS), pareia por paciente,
converte diagnósticos numéricos para texto.

Entrada:
    data/external/PAPILA/.../ClinicalData/patient_data_od.xlsx
    data/external/PAPILA/.../ClinicalData/patient_data_os.xlsx
    data/external/PAPILA/.../Images/*.jpg

Saída:
    data/raw/glaucoma_papila_dataset.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from glaucpio.config import RAW_DIR, BASE_DIR
from glaucpio.logger import get_logger

log = get_logger("glaucpio.papila")


# ============================================================
# MAPEAMENTO DE DIAGNÓSTICO (PAPILA usa códigos numéricos)
# ============================================================
MAPA_DIAGNOSTICO = {
    0: "Normal (NON)",
    1: "Glaucoma Suspect (SUSP)",
    2: "Glaucomatous Optic Neuropathy (GON)",
}


# ============================================================
# CARREGAR UM OLHO
# ============================================================
def carregar_olho(xlsx_path: Path, olho: str) -> pd.DataFrame:
    """
    Carrega metadados de um olho.

    Args:
        xlsx_path: caminho do arquivo .xlsx
        olho: 'OD' ou 'OS'

    Returns:
        DataFrame com colunas renomeadas + coluna 'Olho'
    """
    log.info(f"Lendo {olho}: {xlsx_path}")
    df = pd.read_excel(xlsx_path, header=1)

    # Renomear
    df = df.rename(columns={
        "Diagnosis": "Ground_Truth_Code",
        "Age": "Idade",
        "Gender": "Sexo",
        "Pneumatic": "IOP_Pneumatic",
        "Perkins": "IOP_Perkins",
        "Pachymetry": "Paquimetria",
        "Axial_Length": "Comprimento_Axial",
        "VF_MD": "VF_Mean_Deviation",
        "Phakic/Pseudophakic": "Status_Cristalino",
        "dioptre_1": "Dioptria_1",
        "dioptre_2": "Dioptria_2",
        "astigmatism": "Astigmatismo",
    })

    # Remover linhas com Diagnosis vazio
    df = df.dropna(subset=["Ground_Truth_Code"]).copy()

    # Converter código numérico para texto
    df["Ground_Truth_Diagnosis"] = (
        df["Ground_Truth_Code"].map(MAPA_DIAGNOSTICO)
    )

    # Adicionar coluna de olho
    df["Olho"] = olho

    # Criar Case_ID único
    df["Case_ID"] = [f"PAPILA-{i+1:04d}-{olho}" for i in range(len(df))]

    log.info(f"  {olho}: {len(df)} casos após limpeza")
    log.info(f"  Diagnósticos: {df['Ground_Truth_Diagnosis'].value_counts().to_dict()}")

    return df


# ============================================================
# PROCURAR IMAGENS
# ============================================================
def encontrar_imagem_para_caso(case_id: str, images_dir: Path, olho: str) -> str:
    """
    Procura imagem de fundo correspondente ao caso.

    PAPILA nomeia imagens como: {patient_id}_{olho}.jpg
    Ex.: 0001_OD.jpg, 0001_OS.jpg
    """
    if not images_dir.exists():
        return ""

    # Tentar padrões comuns
    paciente_id = case_id.split("-")[1]  # ex.: "0001"
    padroes = [
        f"{paciente_id}_{olho}.jpg",
        f"{paciente_id}_{olho}.png",
        f"{paciente_id}{olho}.jpg",
        f"{int(paciente_id)}_{olho}.jpg",
        f"{int(paciente_id):04d}_{olho}.jpg",
    ]
    for padrao in padroes:
        p = images_dir / padrao
        if p.exists():
            return str(p.relative_to(BASE_DIR))
    return ""


# ============================================================
# MAIN
# ============================================================
def main():
    log.info("=" * 60)
    log.info("INTEGRAÇÃO DO PAPILA — GlaucPIO")
    log.info("=" * 60)

    # Localizar arquivos
    papila_dir = BASE_DIR / "data" / "external" / "PAPILA"
    if not papila_dir.exists():
        log.error(f"PAPILA não encontrado: {papila_dir}")
        sys.exit(1)

    # Procurar os arquivos od.xlsx e os.xlsx
    od_paths = list(papila_dir.rglob("patient_data_od.xlsx"))
    os_paths = list(papila_dir.rglob("patient_data_os.xlsx"))

    if not od_paths or not os_paths:
        log.error("Arquivos od.xlsx e os.xlsx não encontrados.")
        log.error(f"od: {od_paths}")
        log.error(f"os: {os_paths}")
        sys.exit(1)

    # Carregar ambos os olhos
    df_od = carregar_olho(od_paths[0], "OD")
    df_os = carregar_olho(os_paths[0], "OS")

    # Concatenar
    df = pd.concat([df_od, df_os], ignore_index=True)
    log.info(f"Total após concatenação: {len(df)} casos")

    # Procurar pasta de imagens
    images_dir = None
    for candidate in papila_dir.rglob("Images"):
        if candidate.is_dir():
            images_dir = candidate
            break
    if images_dir is None:
        for candidate in papila_dir.rglob("images"):
            if candidate.is_dir():
                images_dir = candidate
                break

    if images_dir:
        log.info(f"Pasta de imagens: {images_dir}")
        df["Fundus_Image_Path"] = df.apply(
            lambda r: encontrar_imagem_para_caso(r["Case_ID"], images_dir, r["Olho"]),
            axis=1,
        )
        n_imgs = (df["Fundus_Image_Path"] != "").sum()
        log.info(f"  Imagens encontradas: {n_imgs}/{len(df)}")
    else:
        log.warning("Pasta de imagens não encontrada. Criando coluna vazia.")
        df["Fundus_Image_Path"] = ""

    # Prompt multimodal
    df["Multimodal_Prompt"] = df.apply(
        lambda r: (
            f"Analyze the fundus photograph and clinical data: "
            f"Age={r.get('Idade', 'N/A')}, Sex={r.get('Sexo', 'N/A')}, "
            f"IOP={r.get('IOP_Pneumatic', 'N/A')} mmHg, "
            f"Pachymetry={r.get('Paquimetria', 'N/A')} µm, "
            f"Axial Length={r.get('Comprimento_Axial', 'N/A')} mm, "
            f"VF MD={r.get('VF_Mean_Deviation', 'N/A')} dB. "
            f"Provide: 1) Diagnosis, 2) Glaucoma stage (if applicable), "
            f"3) Treatment plan."
        ),
        axis=1,
    )

    # Ordenar colunas
    colunas_prioritarias = [
        "Case_ID", "Olho", "Ground_Truth_Diagnosis", "Multimodal_Prompt",
        "Fundus_Image_Path",
        "Idade", "Sexo", "IOP_Pneumatic", "IOP_Perkins",
        "Paquimetria", "Comprimento_Axial", "VF_Mean_Deviation",
        "Status_Cristalino", "Dioptria_1", "Dioptria_2", "Astigmatismo",
        "Ground_Truth_Code",
    ]
    colunas_presentes = [c for c in colunas_prioritarias if c in df.columns]
    colunas_restantes = [c for c in df.columns if c not in colunas_presentes]
    df = df[colunas_presentes + colunas_restantes]

    # Salvar
    output_path = RAW_DIR / "glaucoma_papila_dataset.csv"
    df.to_csv(output_path, index=False)
    log.info(f"→ Salvo: {output_path}")
    log.info(f"Shape final: {df.shape}")

    # Estatísticas
    log.info("-" * 60)
    log.info("Distribuição de diagnósticos:")
    for diag, count in df["Ground_Truth_Diagnosis"].value_counts().items():
        log.info(f"  {diag}: {count} ({100*count/len(df):.1f}%)")

    log.info("-" * 60)
    log.info("Distribuição por olho:")
    for olho, count in df["Olho"].value_counts().items():
        log.info(f"  {olho}: {count}")

    log.info("-" * 60)
    log.info("Estatísticas clínicas:")
    for col in ["IOP_Pneumatic", "IOP_Perkins", "Paquimetria",
                "Comprimento_Axial", "VF_Mean_Deviation"]:
        if col in df.columns:
            log.info(f"  {col}: média={df[col].mean():.2f}, "
                     f"min={df[col].min():.2f}, max={df[col].max():.2f}")

    log.info("=" * 60)
    log.info("✅ Integração concluída.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()