"""MCP server for datos.gob.es open data catalog API."""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

try:
    # When running as a package
    from .client import DatosGobClient, DatosGobClientError
    from .models import DatasetSummary, DistributionSummary, PaginationParams
except ImportError:
    # When running directly (e.g., fastmcp inspect)
    from client import DatosGobClient, DatosGobClientError
    from models import DatasetSummary, DistributionSummary, PaginationParams

# Initialize FastMCP server
mcp = FastMCP(
    name="datos-gob-es",
    instructions="Access Spain's open data catalog from datos.gob.es. "
    "Search and explore thousands of public datasets from Spanish government institutions.",
)

# Shared client instance
client = DatosGobClient()


def _format_response(data: dict[str, Any], summary_type: str | None = None) -> str:
    """Format API response for readable output.

    Args:
        data: Raw API response.
        summary_type: Type of summary to generate ('dataset', 'distribution', or None).

    Returns:
        Formatted JSON string.
    """
    result = data.get("result", {})
    items = result.get("items", [])

    output = {
        "total_in_page": len(items),
        "page": result.get("page", 0),
        "items_per_page": result.get("itemsPerPage", 10),
    }

    if summary_type == "dataset":
        output["datasets"] = [
            DatasetSummary.from_api_item(item).model_dump(exclude_none=True) for item in items
        ]
    elif summary_type == "distribution":
        output["distributions"] = [
            DistributionSummary.from_api_item(item).model_dump(exclude_none=True) for item in items
        ]
    else:
        output["items"] = items

    return json.dumps(output, ensure_ascii=False, indent=2)


def _handle_error(e: Exception) -> str:
    """Format error message."""
    if isinstance(e, DatosGobClientError):
        return json.dumps({"error": e.message, "status_code": e.status_code}, ensure_ascii=False)
    return json.dumps({"error": str(e)}, ensure_ascii=False)


# =============================================================================
# DATASET TOOLS
# =============================================================================


@mcp.tool()
async def list_datasets(
    page: int = 0,
    page_size: int = 10,
    sort: str | None = "-modified",
) -> str:
    """List datasets from the Spanish open data catalog.

    Browse all available datasets with pagination. Use this to discover
    what public data is available from Spanish government institutions.

    Args:
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).
        sort: Sort field. Use '-' prefix for descending. Examples: '-modified', 'title', '-issued'.

    Returns:
        JSON with datasets including title, description, publisher, and available formats.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size, sort=sort)
        data = await client.list_datasets(pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_dataset(dataset_id: str) -> str:
    """Get detailed information about a specific dataset.

    Retrieve complete metadata for a dataset including all its distributions
    (downloadable files), description, publisher, and update frequency.

    Args:
        dataset_id: The dataset identifier (slug from the URL or URI).

    Returns:
        JSON with full dataset details including download URLs.
    """
    try:
        data = await client.get_dataset(dataset_id)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def search_datasets_by_title(
    title: str,
    page: int = 0,
    page_size: int = 10,
) -> str:
    """Search datasets by title text.

    Find datasets whose title contains the search term. Useful for finding
    specific topics like 'presupuesto', 'empleo', 'educacion', etc.

    Args:
        title: Text to search in dataset titles (partial match supported).
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with matching datasets.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.search_datasets_by_title(title, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_datasets_by_publisher(
    publisher_id: str,
    page: int = 0,
    page_size: int = 10,
) -> str:
    """Get datasets from a specific publisher/organization.

    Find all datasets published by a government institution. Use list_publishers
    to discover available publisher IDs.

    Args:
        publisher_id: Publisher identifier (e.g., 'A16003011' for INE).
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with datasets from the specified publisher.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.get_datasets_by_publisher(publisher_id, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_datasets_by_theme(
    theme_id: str,
    page: int = 0,
    page_size: int = 10,
) -> str:
    """Get datasets by category/theme.

    Browse datasets in a specific topic area. Common themes include:
    economia, hacienda, educacion, salud, medio-ambiente, transporte, turismo.

    Args:
        theme_id: Theme identifier (e.g., 'economia', 'hacienda', 'educacion').
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with datasets in the specified category.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.get_datasets_by_theme(theme_id, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_datasets_by_format(
    format_id: str,
    page: int = 0,
    page_size: int = 10,
) -> str:
    """Get datasets available in a specific file format.

    Find datasets that have distributions in formats like CSV, JSON, XML,
    RDF, etc. Useful when you need data in a particular format.

    Args:
        format_id: Format identifier (e.g., 'csv', 'json', 'xml', 'rdf', 'xlsx').
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with datasets available in the specified format.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.get_datasets_by_format(format_id, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_datasets_by_keyword(
    keyword: str,
    page: int = 0,
    page_size: int = 10,
) -> str:
    """Get datasets tagged with a specific keyword.

    Search datasets by their assigned tags/keywords. Keywords are more specific
    than themes and help find focused datasets.

    Args:
        keyword: Keyword/tag to search (e.g., 'presupuesto', 'gastos', 'poblacion').
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with datasets tagged with the keyword.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.get_datasets_by_keyword(keyword, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_datasets_by_spatial(
    spatial_type: str,
    spatial_value: str,
    page: int = 0,
    page_size: int = 10,
) -> str:
    """Get datasets by geographic coverage.

    Find datasets that cover a specific geographic area. Examples:
    - Autonomia/Pais-Vasco, Autonomia/Cataluna, Autonomia/Comunidad-Madrid
    - Provincia/Madrid, Provincia/Barcelona

    Args:
        spatial_type: Geographic type (e.g., 'Autonomia', 'Provincia').
        spatial_value: Geographic value (e.g., 'Pais-Vasco', 'Madrid').
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with datasets covering the specified area.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.get_datasets_by_spatial(spatial_type, spatial_value, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_datasets_by_date_range(
    begin_date: str,
    end_date: str,
    page: int = 0,
    page_size: int = 10,
) -> str:
    """Get datasets modified within a date range.

    Find datasets that were updated between two dates. Useful for tracking
    recent updates or finding historical data changes.

    Args:
        begin_date: Start date in format 'YYYY-MM-DDTHH:mmZ' (e.g., '2024-01-01T00:00Z').
        end_date: End date in format 'YYYY-MM-DDTHH:mmZ' (e.g., '2024-12-31T23:59Z').
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with datasets modified in the date range.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.get_datasets_by_date_range(begin_date, end_date, pagination)
        return _format_response(data, "dataset")
    except Exception as e:
        return _handle_error(e)


# =============================================================================
# DISTRIBUTION TOOLS
# =============================================================================


@mcp.tool()
async def list_distributions(
    page: int = 0,
    page_size: int = 10,
) -> str:
    """List all available data distributions (downloadable files).

    Browse individual downloadable files across all datasets. Each distribution
    is a specific file format (CSV, JSON, etc.) for a dataset.

    Args:
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with distributions including download URLs and formats.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.list_distributions(pagination)
        return _format_response(data, "distribution")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_distributions_by_dataset(
    dataset_id: str,
    page: int = 0,
    page_size: int = 10,
) -> str:
    """Get all downloadable files for a specific dataset.

    List all available formats and download URLs for a dataset.
    Use this after finding an interesting dataset to get its files.

    Args:
        dataset_id: Dataset identifier.
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with distributions for the dataset.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.get_distributions_by_dataset(dataset_id, pagination)
        return _format_response(data, "distribution")
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_distributions_by_format(
    format_id: str,
    page: int = 0,
    page_size: int = 10,
) -> str:
    """Get distributions (files) in a specific format.

    Find downloadable files in a particular format across all datasets.

    Args:
        format_id: Format identifier (e.g., 'csv', 'json', 'xml').
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with distributions in the specified format.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.get_distributions_by_format(format_id, pagination)
        return _format_response(data, "distribution")
    except Exception as e:
        return _handle_error(e)


# =============================================================================
# METADATA TOOLS
# =============================================================================


@mcp.tool()
async def list_publishers(
    page: int = 0,
    page_size: int = 50,
) -> str:
    """List all data publishers (government organizations).

    Get a list of all institutions that publish data on datos.gob.es.
    Use the publisher IDs to filter datasets by organization.

    Args:
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with publisher organizations and their IDs.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.list_publishers(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def list_spatial_coverage(
    page: int = 0,
    page_size: int = 50,
) -> str:
    """List all geographic coverage options.

    Get available geographic areas that datasets can cover.
    Includes autonomous regions, provinces, and municipalities.

    Args:
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with geographic coverage options.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.list_spatial_coverage(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def list_themes(
    page: int = 0,
    page_size: int = 50,
) -> str:
    """List all dataset categories/themes.

    Get all topic categories used to classify datasets.
    Common themes: economia, hacienda, educacion, salud, transporte, etc.

    Args:
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with available themes and their labels.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.list_themes(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


# =============================================================================
# NTI TOOLS (Norma Técnica de Interoperabilidad)
# =============================================================================


@mcp.tool()
async def list_public_sectors(
    page: int = 0,
    page_size: int = 50,
) -> str:
    """List all public sectors from NTI taxonomy.

    Get sectors defined by Spain's Technical Interoperability Standard.
    Includes: comercio, educacion, salud, justicia, etc.

    Args:
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with public sector categories.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.list_public_sectors(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_public_sector(sector_id: str) -> str:
    """Get details about a specific public sector.

    Retrieve information about a sector from the NTI taxonomy.

    Args:
        sector_id: Sector identifier (e.g., 'comercio', 'educacion', 'salud').

    Returns:
        JSON with sector details.
    """
    try:
        data = await client.get_public_sector(sector_id)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def list_provinces(
    page: int = 0,
    page_size: int = 50,
) -> str:
    """List all Spanish provinces.

    Get the 50 provinces of Spain plus Ceuta and Melilla.

    Args:
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with Spanish provinces.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.list_provinces(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_province(province_id: str) -> str:
    """Get details about a specific Spanish province.

    Args:
        province_id: Province name (e.g., 'Madrid', 'Barcelona', 'Sevilla').

    Returns:
        JSON with province details.
    """
    try:
        data = await client.get_province(province_id)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def list_autonomous_regions(
    page: int = 0,
    page_size: int = 20,
) -> str:
    """List all Spanish autonomous regions (Comunidades Autónomas).

    Get Spain's 17 autonomous communities plus Ceuta and Melilla.

    Args:
        page: Page number (starting from 0).
        page_size: Number of results per page (max 50).

    Returns:
        JSON with autonomous regions.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)
        data = await client.list_autonomous_regions(pagination)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_autonomous_region(region_id: str) -> str:
    """Get details about a specific autonomous region.

    Args:
        region_id: Region identifier (e.g., 'Comunidad-Madrid', 'Cataluna', 'Andalucia').

    Returns:
        JSON with region details.
    """
    try:
        data = await client.get_autonomous_region(region_id)
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


@mcp.tool()
async def get_country_spain() -> str:
    """Get information about Spain as a country.

    Retrieve Spain's country-level information from the NTI geographic taxonomy.

    Returns:
        JSON with Spain country data.
    """
    try:
        data = await client.get_country_spain()
        return _format_response(data)
    except Exception as e:
        return _handle_error(e)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
