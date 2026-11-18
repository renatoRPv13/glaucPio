"""Torna o pacote em src/ importável nos testes sem instalação."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
