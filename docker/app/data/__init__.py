"""
Data package initialization.
"""

from .mlm_dataset import WikiText2MLMDataset, load_wikitext2_for_mlm

__all__ = ["WikiText2MLMDataset", "load_wikitext2_for_mlm"]
