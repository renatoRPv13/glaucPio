# 🩺 GlaucPIO

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-orange)]()

> **Avaliação Multimodal e Calibrada de Modelos de Deep Learning para Diagnóstico de Glaucoma**

O **Projeto GlaucPIO** é um framework completo para avaliação de modelos de Inteligência Artificial no diagnóstico de glaucoma, utilizando o dataset público **PAPILA** (488 retinografias). O projeto vai além dos LLMs textuais ao integrar:

- ✅ **Imagens reais de fundo de olho**
- ✅ **Deep Learning (ResNet-18)**
- ✅ **Calibração estatística (ECE, Brier Score)**
- ✅ **Diagnóstico tripartido (Normal, Suspeito, Glaucoma)**
- ✅ **Reprodutibilidade total (seed fixo, código aberto)**

---

## 📋 Sumário

- [Motivação](#-motivação)
- [Resultados Principais](#-resultados-principais)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Como Rodar](#-como-rodar)
- [Dataset](#-dataset)
- [Referências](#-referências)
- [Como Citar](#-como-citar)
- [Licença](#-licença)

---

## 🎯 Motivação

O glaucoma é a **principal causa de cegueira irreversível no mundo**, com projeção de **111,8 milhões de casos até 2040** (Tham et al., 2014).

Estudos recentes (Gobira et al., 2026) demonstraram que o **ChatGPT-5** atinge **86,8%** de acurácia em questões teóricas de glaucoma, superando residentes humanos (62,9%). No entanto, essa abordagem é **100% textual** e ignora:

- Imagens médicas (fundoscopia, OCT)
- Calibração de incerteza (o modelo sabe quando não sabe?)
- A classe ambígua **"Suspeito"**, crucial na prática clínica

**O GlaucPIO preenche essa lacuna** ao propor uma avaliação multimodal e calibrada, integrando imagens reais e métricas rigorosas de confiabilidade.

---

## 📊 Resultados Principais

### Comparação de Modelos

| Modelo | F1 Macro | Acurácia | AUC Macro | ECE |
|---|---|---|---|---|
| Logistic Regression | 0,487 | 67,0% | — | — |
| SVM (RBF) | 0,433 | 67,0% | — | — |
| Random Forest | 0,424 | 66,0% | — | — |
| **ResNet-18 (GlaucPIO)** | **0,693** | **79,6%** | **0,865** | **0,098** |

### Desempenho por Classe (ResNet-18)

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| Normal | 0,87 | 0,90 | 0,88 | 67 |
| Suspeito | 0,71 | 0,59 | 0,65 | 17 |
| Glaucoma | 0,53 | 0,57 | 0,55 | 14 |

### Calibração

| Métrica | Valor |
|---|---|
| **ECE (Expected Calibration Error)** | **0,098** ✅ |
| **Brier Score** | **0,157** ✅ |
| **AUC Macro** | **0,865** ✅ |

> **Interpretação:** ECE < 0,10 indica **excelente calibração**. O modelo sabe quando não sabe.

---

## 📁 Estrutura do Projeto
glaucPIO/
│
├── data/
│ ├── raw/ # Dados brutos (não versionados)
│ │ └── glaucoma_papila_dataset.csv
│ ├── processed/ # Dados processados
│ └── external/
│ └── PAPILA/ # Dataset original (download separado)
│ ├── ClinicalData/
│ ├── FundusImages/
│ └── ...
│
├── src/
│ └── glaucpio/
│ ├── init.py
│ ├── config.py # Configurações centrais
│ ├── logger.py # Logging estruturado
│ ├── metricas.py # Wilson CI, McNemar, ECE, Brier
│ ├── parser.py # Parser de respostas de LLMs
│ ├── modelos.py # Wrappers (Gemini, GPT, Claude)
│ ├── avaliacao.py # Pipeline de avaliação
│ └── visualizacao.py # Figuras e gráficos
│
├── scripts/
│ ├── integrar_papila.py # Integra o dataset PAPILA
│ ├── preprocessar_dataset.py # Limpeza e enriquecimento
│ ├── fase_2_baseline_v2.py # ML clássico (LR, SVM, RF)
│ ├── fase_3_deep_learning_v2.py # ResNet-18 (treinamento)
│ ├── avaliar_calibracao_resnet.py # ECE, Brier, ROC, matriz
│ ├── plot_resnet_curvas.py # Curvas de aprendizado
│ ├── gerar_figura_combinada.py # Painel 2×2 para artigo
│ └── run_replicacao_baseline.py # Replicação do Gobira et al.
│
├── outputs/
│ ├── csv/ # Resultados (não versionados)
│ ├── figures/ # Figuras (.png, .pdf)
│ └── logs/ # Logs de execução
│
├── notebooks/
│ └── analise_exploratoria.ipynb
│
├── tests/ # Testes unitários (pytest)
│ ├── test_metricas.py
│ ├── test_parser.py
│ └── test_avaliacao.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md


---

## ⚙️ Pré-requisitos

- **Python 3.10+** (recomendado 3.12)
- **pip** e **venv**
- **GPU (opcional)** para acelerar o treinamento da ResNet-18

### Bibliotecas principais

| Biblioteca | Versão mínima | Uso |
|---|---|---|
| `numpy` | 1.26 | Computação numérica |
| `pandas` | 2.2 | Manipulação de dados |
| `scikit-learn` | 1.5 | ML clássico |
| `torch` | 2.0 | Deep Learning (ResNet-18) |
| `torchvision` | 0.15 | Modelos pré-treinados |
| `matplotlib` | 3.8 | Visualização |
| `seaborn` | 0.13 | Gráficos estatísticos |
| `scipy` | 1.13 | Estatística (Wilcoxon, Shapiro) |
| `statsmodels` | 0.14 | Testes estatísticos |
| `Pillow` | 10.0 | Processamento de imagens |
| `imbalanced-learn` | 0.12 | SMOTE (balanceamento) |
| `tenacity` | 8.5 | Retry em chamadas de API |
| `python-dotenv` | 1.0 | Variáveis de ambiente |

---

## 🚀 Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/renatoRPv13/glaucPio.git
cd glaucPio

2. Criar e ativar o ambiente virtual
Linux/macOS:

bash
python3 -m venv .venv
source .venv/bin/activate
Windows:

bash
python -m venv .venv
.venv\Scripts\activate
3. Atualizar o pip
bash
python -m pip install --upgrade pip setuptools wheel
4. Instalar as dependências
bash
pip install -r requirements.txt

5. Baixar o dataset PAPILA
O dataset PAPILA não está incluído no repositório (por questões de tamanho e licença).

Acesse: https://doi.org/10.6084/m9.figshare.14798004

Baixe o arquivo PAPILA.zip

Extraia em data/external/PAPILA/

Estrutura esperada:

text
data/external/PAPILA/
├── ClinicalData/
│   ├── patient_data_od.xlsx
│   └── patient_data_os.xlsx
└── FundusImages/
    ├── RET002OD.jpg
    ├── RET002OS.jpg
    └── ...


6. Configurar variáveis de ambiente (opcional)
Crie um arquivo .env na raiz do projeto:

bash
# .env
GOOGLE_API_KEY=sua_chave_google_aqui
OPENAI_API_KEY=sua_chave_openai_aqui
ANTHROPIC_API_KEY=sua_chave_anthropic_aqui


Como Rodar
Todos os scripts devem ser executados da raiz do projeto com o ambiente virtual ativado.

Pipeline Completo (Ordem Recomendada
1️⃣ Integrar o dataset PAPILA
bash
python scripts/integrar_papila.py
Saída esperada:

text
data/raw/glaucoma_papila_dataset.csv
Este script:

Lê os Excels patient_data_od.xlsx e patient_data_os.xlsx

Consolida os 488 casos

Vincula as imagens de fundo de olho

Salva o CSV final

2️⃣ Pré-processar o dataset
bash
python scripts/preprocessar_dataset.py
Saída esperada:

text
data/processed/dataset_limpo.csv
data/processed/dataset_com_features.csv
data/processed/estatisticas_descritivas.csv
3️⃣ Rodar o baseline de ML clássico
bash
python scripts/fase_2_baseline_v2.py
Saída esperada:

text
outputs/csv/fase2_v2_resultados.csv
Este script treina:

Logistic Regression (PCA + SMOTE)

SVM (RBF)

Random Forest

Tempo estimado: ~2–3 minutos (CPU)

4️⃣ Treinar a ResNet-18
bash
python scripts/fase_3_deep_learning_v2.py --epochs 30 --patience 5
Saída esperada:

text
outputs/papila_resnet18_best.pt
outputs/csv/resnet_historico.csv
outputs/csv/resnet_predicoes.csv
Este script:

Usa seed fixo (42) para reprodutibilidade

Aplica early stopping por F1 macro

Salva o melhor modelo

Tempo estimado: ~30–45 min (CPU) | ~5–10 min (GPU)

5️⃣ Avaliar a calibração da ResNet-18
bash
python scripts/avaliar_calibracao_resnet.py
Saída esperada:

text
outputs/csv/resnet_calibracao.csv
outputs/figures/fig_resnet_reliability.png
outputs/figures/fig_resnet_roc.png
outputs/figures/fig_resnet_matriz_confusao.png
Este script calcula:

ECE (Expected Calibration Error)

Brier Score

AUC-ROC multiclasse

Matriz de confusão

6️⃣ Gerar as curvas de aprendizado
bash
python scripts/plot_resnet_curvas.py
Saída esperada:

text
outputs/figures/fig_resnet_curvas.png
outputs/figures/fig_resnet_curvas_artigo.png
7️⃣ Gerar o painel combinado (2×2) para o artigo
bash
python scripts/gerar_figura_combinada.py
Saída esperada:

text
outputs/figures/fig_resnet_combinada.png
outputs/figures/fig_resnet_combinada.pdf
Rodar os Testes
bash
pytest tests/ -v
Saída esperada:

text
tests/test_metricas.py::test_wilson_ci PASSED
tests/test_parser.py::TestClasses::test_tres_classes_definidas PASSED
tests/test_avaliacao.py::test_avaliacao_fecha_imagens_depois_da_chamada PASSED
...
Replicar o Estudo Original (Gobira et al., 2026)
bash
python scripts/run_replicacao_baseline.py
Saída esperada:

text
outputs/csv/tabela1_acuracia.csv
outputs/csv/tabela2_mcnemar.csv
outputs/figures/figura1_acuracia_wilson.png
outputs/figures/figura2_diferenca_item.png
Este script:

Reproduz as Tabelas 1 e 2 do artigo original

Gera as Figuras 1 e 2

Valida o pipeline estatístico (Wilson CI, McNemar, Haldane-Anscombe)

📊 Dataset
PAPILA
Fonte: Figshare

Autores: Kovalyk et al. (2022)

Descrição: 488 retinografias de 244 pacientes (Hospital Reina Sofía, Espanha)

Classes: Normal (68,2%), Suspeito (17,8%), Glaucoma (13,9%)

Licença: CC-BY 4.0

Nota: O dataset não é versionado neste repositório. Faça o download separadamente e coloque em data/external/PAPILA/.

📚 Referências
Tham, Y.-C. et al. (2014). Global prevalence of glaucoma and projections of glaucoma burden through 2040. Ophthalmology, 121(11), 2081–2090.

Kovalyk, O. et al. (2022). PAPILA: Dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment. Scientific Data, 9, 291.

Gobira, M. et al. (2026). Assessing a large language model for glaucoma knowledge: ChatGPT-5 versus residents. Arquivos Brasileiros de Oftalmologia, 89(4), e2025-0283.

Li, Z. et al. (2018). Efficacy of a deep learning system for detecting glaucomatous optic neuropathy based on color fundus photographs. Ophthalmology, 125(8), 1199–1206.

Ting, D. S. et al. (2019). Artificial intelligence and deep learning in ophthalmology. British Journal of Ophthalmology, 103(2), 167–175.