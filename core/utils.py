"""Utility functions for data extraction and formatting."""

from typing import Any


def extract_uri_suffix(value: Any) -> str | None:
    """Extract the last segment from a URI or return the value as-is.

    Handles: str, dict with _about/_value, list of str/dict.

    Args:
        value: The value to extract from (str, dict, list, or None).

    Returns:
        The extracted suffix or None.

    Examples:
        >>> extract_uri_suffix("http://example.com/foo/bar")
        'bar'
        >>> extract_uri_suffix({"_about": "http://example.com/foo"})
        'foo'
        >>> extract_uri_suffix("simple_value")
        'simple_value'
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value.split("/")[-1] if "/" in value else value

    if isinstance(value, dict):
        # Try _about first, then _value
        about = value.get("_about", "")
        if about and "/" in about:
            return about.split("/")[-1]
        return value.get("_value") or value.get("label")

    if isinstance(value, list) and value:
        return extract_uri_suffix(value[0])

    return None


def extract_dict_value(
    obj: Any, keys: list[str] | None = None, default: Any = None
) -> Any:
    """Extract value from dict trying multiple keys, or return as-is.

    Args:
        obj: Object to extract from (dict, str, or other).
        keys: List of keys to try in order. Defaults to ["_value", "label", "name"].
        default: Default value if extraction fails.

    Returns:
        Extracted value or original object.
    """
    if keys is None:
        keys = ["_value", "label", "name"]

    if obj is None:
        return default

    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                val = obj[key]
                # Handle nested dict
                if isinstance(val, dict):
                    return extract_dict_value(val, keys, default)
                return val
        # Fallback to _about suffix
        about = obj.get("_about")
        if about:
            return extract_uri_suffix(about)
        return default

    return obj


# Format normalization mapping
FORMAT_MAPPINGS: dict[str, str] = {
    "csv": "CSV",
    "comma-separated": "CSV",
    "json": "JSON",
    "xml": "XML",
    "xls": "Excel",
    "xlsx": "Excel",
    "excel": "Excel",
    "spreadsheet": "Excel",
    "pdf": "PDF",
    "rdf": "RDF",
    "html": "HTML",
    "api": "API",
    "zip": "ZIP",
    "shp": "Shapefile",
    "shapefile": "Shapefile",
    "geojson": "GeoJSON",
    "tsv": "TSV",
    "tab-separated": "TSV",
    "txt": "TXT",
    "text/plain": "TXT",
}


def normalize_format(fmt: str) -> str:
    """Normalize a format string to a standard name.

    Args:
        fmt: Raw format string from API.

    Returns:
        Normalized format name (e.g., "CSV", "JSON", "Excel").

    Examples:
        >>> normalize_format("text/csv")
        'CSV'
        >>> normalize_format("application/json")
        'JSON'
        >>> normalize_format("application/vnd.ms-excel")
        'Excel'
    """
    if not fmt:
        return fmt

    fmt_lower = fmt.lower()

    for pattern, normalized in FORMAT_MAPPINGS.items():
        if pattern in fmt_lower:
            return normalized

    # Fallback: clean up the format name
    clean_fmt = fmt.split("/")[-1] if "/" in fmt else fmt
    return clean_fmt.upper() if len(clean_fmt) <= 5 else clean_fmt.title()
