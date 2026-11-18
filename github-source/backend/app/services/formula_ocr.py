from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ..config import get_settings


def _plain(value: Any) -> Any:
    """Convert PaddleOCR/numpy values to JSON-safe Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _result_json(result: Any) -> dict:
    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if isinstance(value, dict):
        return value
    if isinstance(result, dict):
        return result
    return {}


def extract_pdf_formulas(filename: str, content: bytes) -> dict:
    """Recognize equation regions in a PDF and return their LaTeX and locations."""
    settings = get_settings()
    if not settings.formula_ocr_enabled:
        return {"status": "disabled", "model": settings.formula_ocr_model, "formulas": []}

    try:
        # On Windows, loading Paddle before Torch can make Torch's shm.dll fail
        # to resolve. Keep the verified import order: Torch first, then PaddleOCR.
        import torch  # noqa: F401
        from paddleocr import FormulaRecognitionPipeline
    except ImportError:
        return {
            "status": "unavailable",
            "model": settings.formula_ocr_model,
            "formulas": [],
            "warning": "PaddleOCR is not installed; formula OCR was skipped.",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "model": settings.formula_ocr_model,
            "formulas": [],
            "warning": f"PaddleOCR could not be loaded ({exc.__class__.__name__}: {str(exc)[:240]}); formula OCR was skipped.",
        }

    try:
        with tempfile.TemporaryDirectory(prefix="research-copilot-formula-") as temp_dir:
            pdf_path = Path(temp_dir) / Path(filename).name
            pdf_path.write_bytes(content)
            pipeline = FormulaRecognitionPipeline(
                formula_recognition_model_name=settings.formula_ocr_model
            )
            results = pipeline.predict(str(pdf_path))
            formulas = []
            for result_index, result in enumerate(results):
                payload = _result_json(result)
                res = payload.get("res", payload)
                if not isinstance(res, dict):
                    continue
                page_index = res.get("page_index", result_index)
                try:
                    page_number = int(page_index) + 1
                except (TypeError, ValueError):
                    page_number = result_index + 1
                entries = res.get("formula_res_list") or []
                for formula_index, entry in enumerate(entries, start=1):
                    if not isinstance(entry, dict):
                        continue
                    latex = str(entry.get("rec_formula") or "").strip()
                    if not latex:
                        continue
                    formulas.append({
                        "page": page_number,
                        "formula_index": formula_index,
                        "latex": latex,
                        "region_id": _plain(entry.get("formula_region_id")),
                        "bbox": _plain(entry.get("dt_polys")),
                    })
    except Exception as exc:
        return {
            "status": "failed",
            "model": settings.formula_ocr_model,
            "formulas": [],
            "warning": f"Formula OCR failed ({exc.__class__.__name__}: {str(exc)[:240]}); PDF ingestion continued.",
        }

    return {
        "status": "completed",
        "model": settings.formula_ocr_model,
        "count": len(formulas),
        "formulas": formulas,
    }
