"""FR-01: input validation with understandable errors."""
from pathlib import Path
import pandas as pd


class KeywordInputError(ValueError):
    """Raised for any problem with the input file that the user can fix."""


def load_keywords(path, column: str = "keyword") -> pd.DataFrame:
    """Load a UTF-8 CSV and return a one-column DataFrame named 'keyword' (all strings).

    Blank lines are kept (as empty strings) so they can be counted in the data-quality report.
    """
    p = Path(path)
    if not p.exists():
        raise KeywordInputError(f"Input file not found: {p}")
    if not p.is_file():
        raise KeywordInputError(f"Input path is not a file: {p}")
    if p.stat().st_size == 0:
        raise KeywordInputError(f"Input file is empty: {p}")
    try:
        df = pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False,
                         skip_blank_lines=False)
    except UnicodeDecodeError as e:
        raise KeywordInputError(f"File is not valid UTF-8 ({e.reason}). Re-save it as UTF-8 CSV.") from e
    except pd.errors.EmptyDataError as e:
        raise KeywordInputError(f"Input file has no content: {p}") from e
    except pd.errors.ParserError as e:
        raise KeywordInputError(f"Could not parse CSV: {e}") from e

    cols = {str(c).strip().lower(): c for c in df.columns}
    if column not in cols:
        raise KeywordInputError(
            f"Required column '{column}' not found. Columns present: {list(df.columns)}")
    out = df[[cols[column]]].rename(columns={cols[column]: "keyword"}).reset_index(drop=True)
    if out.empty:
        raise KeywordInputError("CSV has a header but no data rows.")
    return out
