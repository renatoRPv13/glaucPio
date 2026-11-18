cd /home/rpvselva/IdeaProjects/glaucPIO
source .venv/bin/activate

# Rodar todos os testes do parser
pytest tests/test_parser.py -v

# Rodar teste específico
pytest tests/test_parser.py::TestComparacaoComGroundTruth::test_resposta_pt_com_gabarito_en -v

# Rodar com cobertura
pytest tests/test_parser.py --cov=glaucpio.parser -v