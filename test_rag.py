from io import BytesIO
from unittest.mock import Mock, patch

from rag import _split_text, extract_chunks


def test_split_text_keeps_overlap_and_stops_at_end():
    words = " ".join(f"word{i}" for i in range(20))
    chunks = _split_text(words, chunk_size=8, overlap=2)

    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
    assert chunks[-1].split()[-1] == "word19"


def test_extract_chunks_preserves_source_and_page_metadata():
    page = Mock()
    page.extract_text.return_value = "A useful sentence from the document."
    fake_reader = Mock(pages=[page])

    with patch("rag.PdfReader", return_value=fake_reader):
        chunks = extract_chunks([("example.pdf", BytesIO(b"pdf bytes"))])

    assert chunks[0].source == "example.pdf"
    assert chunks[0].page == 1
    assert chunks[0].chunk_number == 1
