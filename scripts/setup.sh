#!/usr/bin/env bash
# ============================================================
# Setup do Projeto GlaucIA
# ============================================================
set -e

echo "============================================================"
echo "SETUP — Projeto GlaucIA"
echo "============================================================"

# 1. Verificar Python
if ! command -v python3.12 &> /dev/null; then
    echo "⚠️  Python 3.12 não encontrado. Instale antes de continuar."
    echo "   Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    exit 1
fi
echo "✅ Python 3.12 encontrado: $(python3.12 --version)"

# 2. Criar virtualenv
if [ ! -d ".venv" ]; then
    echo "→ Criando virtualenv..."
    python3.12 -m venv .venv
fi

# 3. Ativar
source .venv/bin/activate

# 4. Atualizar pip
echo "→ Atualizando pip..."
python -m pip install --upgrade pip setuptools wheel

# 5. Instalar dependências
if [ -f "requirements.txt" ]; then
    echo "→ Instalando dependências..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt não encontrado."
fi

# 6. Criar estrutura de pastas
echo "→ Criando estrutura de pastas..."
mkdir -p data/raw data/processed
mkdir -p outputs/csv outputs/figures outputs/logs
mkdir -p dataset/images/oct dataset/images/fundus dataset/images/visual_field

# 7. Criar .env se não existir
if [ ! -f ".env" ]; then
    echo "→ Criando .env de exemplo..."
    cat > .env << 'EOF'
# Projeto GlaucIA — Variáveis de ambiente
GOOGLE_API_KEY=sua_chave_google_aqui
OPENAI_API_KEY=sua_chave_openai_aqui
ANTHROPIC_API_KEY=sua_chave_anthropic_aqui
EOF
    echo "⚠️  Edite o .env com suas chaves de API."
fi

echo "============================================================"
echo "✅ Setup concluído!"
echo "============================================================"
echo ""
echo "Próximos passos:"
echo "  1. source .venv/bin/activate"
echo "  2. Edite .env com suas chaves de API"
echo "  3. python scripts/preprocessar_dataset.py"
echo "  4. python scripts/split_train_test.py"
echo "  5. python scripts/redimensionar_imagens.py"
echo ""