"""Unit tests for PdfTableExtractor."""

from pathlib import Path
import pytest
import pandas as pd

from data.extractors.pdf import PdfTableExtractor, ExtractedPdfTable

SAMPLE_PDF_PATH = Path(__file__).resolve().parent.parent / "samples" / "S100S37P_2.pdf"


@pytest.fixture
def extractor() -> PdfTableExtractor:
    return PdfTableExtractor()


def test_pdf_not_found(extractor: PdfTableExtractor):
    with pytest.raises(FileNotFoundError):
        extractor.extract_tables("non_existent_file.pdf")


def test_extract_tables_bordered_grid(extractor: PdfTableExtractor):
    """Test extracting fully-enclosed bordered table from Page 2 (Financial Highlights)."""
    assert SAMPLE_PDF_PATH.exists(), f"Sample PDF not found at {SAMPLE_PDF_PATH}"

    tables = extractor.extract_tables(SAMPLE_PDF_PATH, page_numbers=[2])
    assert len(tables) >= 2, f"Expected at least 2 tables on Page 2, got {len(tables)}"

    t1 = tables[0]
    assert isinstance(t1, ExtractedPdfTable)
    assert t1.page_number == 2
    assert t1.row_count >= 10
    assert t1.col_count >= 4
    assert "<table" in t1.html
    assert "</table>" in t1.html

    # Check financial contents
    df_str = t1.df.to_string()
    assert "売上高" in df_str
    assert "経常利益" in df_str


def test_extract_tables_open_horizontal_rules(extractor: PdfTableExtractor):
    """Test extracting open table with horizontal-only borders from Page 13 (Notes on notes payable)."""
    assert SAMPLE_PDF_PATH.exists()

    tables = extractor.extract_tables(SAMPLE_PDF_PATH, page_numbers=[13])
    assert len(tables) >= 1, f"Expected at least 1 table on Page 13, got {len(tables)}"

    t = tables[0]
    assert t.page_number == 13
    assert t.row_count >= 3
    assert t.col_count >= 2
    assert "<table" in t.html

    df_str = t.df.to_string()
    assert ("手形" in df_str or "債権" in df_str or "割引" in df_str)


def test_extract_as_html_and_dataframe(extractor: PdfTableExtractor):
    """Test extract_tables_as_dataframe and extract_tables_as_html helper methods."""
    dfs = extractor.extract_tables_as_dataframe(SAMPLE_PDF_PATH, page_numbers=[2])
    assert isinstance(dfs, list)
    assert len(dfs) >= 2
    assert isinstance(dfs[0], pd.DataFrame)

    htmls = extractor.extract_tables_as_html(SAMPLE_PDF_PATH, page_numbers=[2])
    assert isinstance(htmls, list)
    assert len(htmls) >= 2
    assert htmls[0].startswith('<table class="pdf-extracted-table"')
