"""OCR-level evaluation metrics (requirement 15). Interfaces only for now —
filled in once we have ground-truth transcriptions to benchmark against.
"""


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    raise NotImplementedError("TODO: implement once ground-truth transcriptions are available")


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    raise NotImplementedError("TODO: implement once ground-truth transcriptions are available")


def critical_field_accuracy(expected_fields: dict, extracted_fields: dict) -> float:
    """Fraction of critical fields (EINs, dates, prices, IDs, etc.) that
    match exactly between expected and OCR-extracted values."""
    raise NotImplementedError("TODO: implement once a critical-field validation layer exists")


def layout_preservation_score(reference_structure: dict, extracted_structure: dict) -> float:
    """Compares heading/section/table structure between a reference layout
    and what the OCR + chunker produced."""
    raise NotImplementedError("TODO: define a structure-similarity metric")
