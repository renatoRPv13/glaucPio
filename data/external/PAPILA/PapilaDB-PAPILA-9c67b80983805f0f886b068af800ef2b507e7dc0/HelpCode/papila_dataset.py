from __future__ import annotations

from pathlib import Path
from typing import Tuple
import openpyxl
import numpy as np
import pandas as pd
from PIL import Image


LABELS = {
    0: "normal",
    1: "glaucoma_suspect",
    2: "glaucoma",
}


def _fix_clinical_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["ID"], axis=0, errors="ignore")
    df.columns = df.iloc[0, :]
    df = df.drop([np.nan], axis=0, errors="ignore")
    df.columns.name = "ID"
    df.index = df.index.rename("PatientID")
    return df


def load_clinical_data(root_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root_dir)
    df_od = pd.read_excel(root / "ClinicalData/patient_data_od.xlsx", index_col=[0])
    df_os = pd.read_excel(root / "ClinicalData/patient_data_os.xlsx", index_col=[0])
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
    rows = []

    for eye_label, df in [("OD", df_od), ("OS", df_os)]:
        if eye is not None and eye_label.lower() != eye.lower():
            continue

        for patient_id, row in df.iterrows():
            patient_number = str(patient_id).replace("#", "")
            image_name = f"RET{patient_number}{eye_label}.jpg"
            image_path = root / "FundusImages" / image_name

            if not image_path.exists():
                continue

            diagnosis = int(row.get("Diagnosis"))
            rows.append(
                {
                    "patient_id": patient_number,
                    "eye": eye_label,
                    "image_path": str(image_path),
                    "age": row.get("Age"),
                    "gender": row.get("Gender"),
                    "diagnosis": diagnosis,
                    "diagnosis_label": LABELS.get(diagnosis, "unknown"),
                    "dioptre_1": row.get("dioptre_1"),
                    "dioptre_2": row.get("dioptre_2"),
                    "astigmatism": row.get("astigmatism"),
                    "phakic_pseudophakic": row.get("Phakic/Pseudophakic"),
                    "pachymetry": row.get("Pachymetry"),
                    "axial_length": row.get("Axial_Length"),
                    "vf_md": row.get("VF_MD"),
                }
            )

    return pd.DataFrame(rows).reset_index(drop=True)


def load_classification_matrix(root_dir: str | Path, target_size: tuple[int, int] = (64, 64), eye: str | None = None):
    df = build_dataset(root_dir=root_dir, eye=eye)
    if df.empty:
        raise ValueError("No data found for the selected configuration.")

    X = []
    for path in df["image_path"]:
        image = Image.open(path).convert("RGB").resize(target_size)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        X.append(image_arr.reshape(-1))

    y = df["diagnosis"].astype(int).to_numpy()
    return np.stack(X), y, df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    dataset = build_dataset(root)
    print(f"Dataset rows: {len(dataset)}")
    print(dataset.head().to_string(index=False))
    X, y, df = load_classification_matrix(root, target_size=(64, 64))
    print("X.shape:", X.shape)
    print("y unique:", np.unique(y, return_counts=True))
