from unittest.mock import Mock

import pandas as pd
import pytest
from PIL import Image

import glaucpio.avaliacao as avaliacao
import scripts.run_gerar_figuras as figuras
from glaucpio.config import caminhos_saida


def test_caminhos_saida_slugifica_nome_do_modelo():
    csv_path, jsonl_path = caminhos_saida("my model-v2")

    assert csv_path.name == "my_model_v2_eval_results.csv"
    assert jsonl_path.name == "my_model_v2_eval_raw.jsonl"


def test_caminho_dataset_images_e_resolvido_sem_lstrip(tmp_path, monkeypatch):
    images_dir = tmp_path / "dataset" / "images"
    image_path = images_dir / "oct" / "case.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), "white").save(image_path)
    monkeypatch.setattr("glaucpio.avaliacao.IMAGES_DIR", images_dir)
    monkeypatch.setattr("glaucpio.avaliacao.BASE_DIR", tmp_path)

    imagens = avaliacao.carregar_imagens(
        pd.Series({"OCT_Image_Path": "dataset/images/oct/case.png"})
    )

    assert len(imagens) == 1
    assert imagens[0].size == (2, 2)
    imagens[0].close()


def test_caso_sem_imagem_nao_chama_modelo_e_fica_com_erro():
    modelo = Mock()
    row = pd.Series({
        "Case_ID": "case-1",
        "Multimodal_Prompt": "Analise o caso.",
        "Ground_Truth_Diagnosis": "normal",
    })

    resultado = avaliacao.avaliar_caso(row, modelo)

    modelo.chamar.assert_not_called()
    assert resultado["N_Images"] == 0
    assert resultado["Correto"] is None
    assert resultado["Status"].startswith("ERROR: FileNotFoundError:")


def test_avaliacao_fecha_imagens_depois_da_chamada(monkeypatch):
    imagem = Image.new("RGB", (2, 2), "white")
    modelo = Mock()
    modelo.chamar.return_value = {
        "text": "Diagnóstico: normal",
        "latency_s": 0.1,
        "input_tokens": 1,
        "output_tokens": 1,
    }
    monkeypatch.setattr(avaliacao, "carregar_imagens", lambda case: [imagem])
    row = pd.Series({
        "Case_ID": "case-1",
        "Multimodal_Prompt": "Analise o caso.",
        "Ground_Truth_Diagnosis": "normal",
    })

    resultado = avaliacao.avaliar_caso(row, modelo)

    assert resultado["Status"] == "OK"
    assert resultado["N_Images"] == 1
    with pytest.raises(ValueError, match="closed image"):
        imagem.getpixel((0, 0))


def test_retoma_avaliacao_reprocessando_registro_sem_imagem(tmp_path, monkeypatch):
    dataset = tmp_path / "cases.csv"
    saida = tmp_path / "results.csv"
    jsonl = tmp_path / "results.jsonl"
    pd.DataFrame([{
        "Case_ID": "case-1",
        "Multimodal_Prompt": "Analise o caso.",
        "Ground_Truth_Diagnosis": "normal",
    }]).to_csv(dataset, index=False)
    pd.DataFrame([{
        "Case_ID": "case-1",
        "Status": "OK",
        "N_Images": 0,
        "Correto": True,
    }]).to_csv(saida, index=False)
    modelo = Mock()
    modelo.chamar.return_value = {
        "text": "Diagnóstico: normal",
        "latency_s": 0.1,
        "input_tokens": 1,
        "output_tokens": 1,
    }
    monkeypatch.setattr(
        avaliacao,
        "carregar_imagens",
        lambda case: [Image.new("RGB", (2, 2), "white")],
    )

    resultados = avaliacao.avaliar_dataset(
        modelo,
        dataset_path=dataset,
        output_csv=saida,
        output_jsonl=jsonl,
        max_cases=None,
        sleep=0,
    )

    modelo.chamar.assert_called_once()
    assert resultados.loc[0, "N_Images"] == 1
    assert resultados.loc[0, "Status"] == "OK"


def test_figura_de_diferenca_usa_dados_pareados_reais(monkeypatch):
    observados = {}

    def gerar(llm, residentes, salvar):
        observados["llm"] = llm.tolist()
        observados["residentes"] = residentes.tolist()
        observados["salvar"] = salvar
        return "figura.png"

    monkeypatch.setattr("scripts.run_gerar_figuras.figura2_diferenca_item", gerar)
    dados = pd.DataFrame({
        "llm_correct": [1, 0],
        "resident_mean_accuracy": [0.75, 0.5],
    })

    assert figuras.gerar_figura2(dados) == "figura.png"
    assert observados == {
        "llm": [1, 0],
        "residentes": [0.75, 0.5],
        "salvar": True,
    }


def test_figura_de_diferenca_rejeita_dados_nao_pareados():
    with pytest.raises(ValueError, match="CSV pareado precisa"):
        figuras.gerar_figura2(pd.DataFrame({"llm_correct": [1]}))
