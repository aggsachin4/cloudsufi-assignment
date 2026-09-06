from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from rag import GroundedAnswer, QuestionRequest, extract_documents


def test_extract_documents_keeps_page_metadata_and_splits_with_langchain():
    page = Mock()
    page.extract_text.return_value = "A useful sentence from the document. " * 10
    fake_reader = Mock(pages=[page])

    with patch("rag.PdfReader", return_value=fake_reader):
        documents = extract_documents(
            [("example.pdf", BytesIO(b"pdf bytes"))], chunk_size=80, chunk_overlap=20
        )

    assert len(documents) > 1
    assert documents[0].metadata == {
        "source": "example.pdf",
        "page": 1,
        "chunk_number": 1,
    }
    assert documents[1].metadata["chunk_number"] == 2


def test_structured_models_validate_question_and_citation_shape():
    answer = GroundedAnswer(answer="Supported answer.", citations=[{"source_number": 1}])
    assert answer.citations[0].source_number == 1

    with pytest.raises(ValidationError):
        QuestionRequest(question="")
