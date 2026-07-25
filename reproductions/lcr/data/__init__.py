"""Dataset loaders for the LCR reproduction (BEIR / TREC)."""

from .beir_loader import load_beir
from .trec_loader import load_trec_qrels, load_trec_run

__all__ = ["load_beir", "load_trec_qrels", "load_trec_run"]
