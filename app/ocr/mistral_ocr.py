"""Mistral OCR wrapper.

Produces structured (Markdown) OCR output per image and attaches the
document/page metadata that must survive through chunking, embedding,
retrieval, and the final answer (requirement 2 / 11).

NOTE: verify field names on `client.ocr.process(...)` against the installed
`mistralai` SDK version — the OCR API is relatively new and response shapes
have shifted across releases.
"""
import base64
import uuid
from dataclasses import dataclass, field
from typing import List

from mistralai.client import Mistral

from app.config import settings


@dataclass
class OCRPageResult:
    page_number: int
    markdown: str


@dataclass
class OCRDocumentResult:
    document_id: str
    filename: str
    source: str
    pages: List[OCRPageResult] = field(default_factory=list)

    @property
    def full_markdown(self) -> str:
        return "\n\n".join(p.markdown for p in self.pages)


class MistralOCRClient:
    def __init__(self, api_key: str = settings.MISTRAL_API_KEY, model: str = settings.MISTRAL_OCR_MODEL):
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self._client = Mistral(api_key=api_key)
        self._model = model

    @staticmethod
    def _encode_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def process_image(
        self,
        image_bytes: bytes,
        filename: str,
        mime_type: str = "image/jpeg",
        document_id: str | None = None,
    ) -> OCRDocumentResult:
        """Run OCR on a single image and return structured, metadata-tagged output."""
        document_id = document_id or str(uuid.uuid4())
        data_uri = self._encode_image(image_bytes, mime_type)

        response = self._client.ocr.process(
            model=self._model,
            document={
                "type": "image_url",
                "image_url": data_uri,
            },
        )

        pages = [
            OCRPageResult(page_number=idx + 1, markdown=page.markdown)
            for idx, page in enumerate(response.pages)
        ]

        return OCRDocumentResult(
            document_id=document_id,
            filename=filename,
            source=filename,
            pages=pages,
        )

    def process_images(self, images: List[tuple[bytes, str]]) -> List[OCRDocumentResult]:
        """images: list of (image_bytes, filename). Each image is treated as its own document."""
        return [self.process_image(img_bytes, filename) for img_bytes, filename in images]
