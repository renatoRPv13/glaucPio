from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image


LABELS = {
    0: "normal",
    1: "glaucoma_suspect",
    2: "glaucoma",
}


def _fix_clinical_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "PatientID",
        "Age",
        "Gender",
        "Diagnosis",
        "dioptre_1",
        "dioptre_2",
        "astigmatism",
        "Phakic/Pseudophakic",
        "Pneumatic",
        "Perkins",
        "Pachymetry",
        "Axial_Length",
        "VF_MD",
    ]
    if df.shape[1] != len(columns):
        raise ValueError(
            f"Unexpected clinical spreadsheet format: expected {len(columns)} columns, "
            f"found {df.shape[1]}."
        )

    data = df.iloc[3:].copy()
    data.columns = columns
    data = data[data["PatientID"].astype(str).str.match(r"#\d+")]
    data = data.set_index("PatientID")
    data.index.name = "PatientID"
    return data


def load_clinical_data(root_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root_dir)
    df_od = pd.read_excel(root / "ClinicalData/patient_data_od.xlsx", header=None)
    df_os = pd.read_excel(root / "ClinicalData/patient_data_os.xlsx", header=None)
    return _fix_clinical_df(df_od), _fix_clinical_df(df_os)


def get_diagnosis(root_dir: str | Path):
    df_od, df_os = load_clinical_data(root_dir)

    index_od = np.ones(df_od.iloc[:, 2].values.shape, dtype=np.int8)
    index_os = np.zeros(df_os.iloc[:, 2].values.shape, dtype=np.int8)

    eye_id = np.array(list(zip(index_od, index_os))).reshape(-1)
    tag = np.array(list(zip(df_od.iloc[:, 2].values, df_os.iloc[:, 2].values))).reshape(-1)
    patient_id = np.array([[int(i.replace("#", ""))] * 2 for i in df_od.index]).reshape(-1)

    return tag, eye_id, patient_id


def build_dataset(root_dir: str | Path, eye: str | None = None) -> pd.DataFrame:
    root = Path(root_dir)
    df_od, df_os = load_clinical_data(root)
    records = []

    for eye_label, df in [("OD", df_od), ("OS", df_os)]:
        if eye is not None and eye_label.lower() != eye.lower():
            continue

        for patient_id, row in df.iterrows():
            pat_number = str(patient_id).replace("#", "")
            image_name = f"RET{pat_number}{eye_label}.jpg"
            image_path = root / "FundusImages" / image_name

            if not image_path.exists():
                continue

            diagnosis_value = pd.to_numeric(row.get("Diagnosis"), errors="coerce")
            if pd.isna(diagnosis_value):
                raise ValueError(
                    f"Missing diagnosis for patient {patient_id} ({eye_label}) "
                    f"in {image_name} clinical data."
                )
            diagnosis = int(diagnosis_value)
            if diagnosis not in LABELS:
                raise ValueError(
                    f"Unknown diagnosis {diagnosis} for patient {patient_id} ({eye_label}); "
                    f"expected one of {sorted(LABELS)}."
                )

            records.append(
                {
                    "patient_id": pat_number,
                    "eye": eye_label,
                    "image_path": str(image_path),
                    "age": row.get("Age"),
                    "gender": row.get("Gender"),
                    "diagnosis": diagnosis,
                    "diagnosis_label": LABELS[diagnosis],
                    "dioptre_1": row.get("dioptre_1"),
                    "dioptre_2": row.get("dioptre_2"),
                    "pachymetry": row.get("Pachymetry"),
                    "axial_length": row.get("Axial_Length"),
                    "vf_md": row.get("VF_MD"),
                }
            )

    return pd.DataFrame(records)


def preprocess_image(image_path: str | Path, target_size=(224, 224), normalize: bool = True):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    array = np.asarray(image, dtype=np.float32)
    if normalize:
        array = array / 255.0
    return array


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    df_od, df_os = load_clinical_data(root_dir)
    dataset = build_dataset(root_dir)

    print("Clinical Data OD shape:", df_od.shape)
    print("Clinical Data OS shape:", df_os.shape)
    print("Dataset rows:", len(dataset))
    print(dataset.head(5).to_string(index=False))
    print("Diagnosis counts:")
    print(dataset["diagnosis"].value_counts().sort_index().to_string())

    sample_path = dataset.iloc[0]["image_path"]
    sample_image = preprocess_image(sample_path)
    print("Sample image shape:", sample_image.shape)
    print("Sample min/max:", float(sample_image.min()), float(sample_image.max()))
