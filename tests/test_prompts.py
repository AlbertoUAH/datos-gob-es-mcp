"""Tests for MCP prompts."""

import pytest
from prompts import buscar_por_tema, datasets_recientes, explorar_catalogo, analisis_dataset


class TestBuscarPorTemaPrompt:
    """Tests for the buscar_por_tema prompt."""

    def test_generate_prompt_default(self):
        """Test prompt generation with default values."""
        result = buscar_por_tema.generate_prompt()

        assert "economia" in result
        assert "csv" in result
        assert "10" in result
        assert "Búsqueda de Datos Abiertos" in result

    def test_generate_prompt_custom_values(self):
        """Test prompt generation with custom values."""
        result = buscar_por_tema.generate_prompt(
            tema="salud",
            formato="json",
            max_resultados=25
        )

        assert "salud" in result
        assert "json" in result
        assert "25" in result

    def test_prompt_contains_instructions(self):
        """Test that prompt contains search instructions."""
        result = buscar_por_tema.generate_prompt()

        assert "catalog://themes" in result
        assert "get_datasets_by_theme" in result
        assert "get_datasets_by_format" in result

    def test_prompt_contains_format_descriptions(self):
        """Test that prompt contains format descriptions."""
        result = buscar_por_tema.generate_prompt()

        assert "CSV" in result.upper() or "csv" in result
        assert "JSON" in result.upper() or "json" in result


class TestDatasetsRecientesPrompt:
    """Tests for the datasets_recientes prompt."""

    def test_generate_prompt_default(self):
        """Test prompt generation with default values."""
        result = datasets_recientes.generate_prompt()

        assert "30" in result
        assert "15" in result
        assert "Búsqueda de Datasets Actualizados" in result

    def test_generate_prompt_with_theme(self):
        """Test prompt generation with theme filter."""
        result = datasets_recientes.generate_prompt(
            dias=7,
            tema="educacion",
            max_resultados=10
        )

        assert "7" in result
        assert "educacion" in result
        assert "10" in result

    def test_generate_prompt_without_theme(self):
        """Test prompt generation without theme filter."""
        result = datasets_recientes.generate_prompt(dias=14)

        assert "14" in result
        assert "get_datasets_by_date_range" in result

    def test_prompt_contains_date_instructions(self):
        """Test that prompt contains date format instructions."""
        result = datasets_recientes.generate_prompt()

        assert "YYYY-MM-DD" in result


class TestExplorarCatalogoPrompt:
    """Tests for the explorar_catalogo prompt."""

    def test_generate_prompt_default(self):
        """Test prompt generation with default interest."""
        result = explorar_catalogo.generate_prompt()

        assert "datos económicos de España" in result
        assert "Exploración Guiada" in result

    def test_generate_prompt_custom_interest(self):
        """Test prompt generation with custom interest."""
        result = explorar_catalogo.generate_prompt(
            interes="datos de turismo en Barcelona"
        )

        assert "datos de turismo en Barcelona" in result

    def test_prompt_contains_phases(self):
        """Test that prompt contains exploration phases."""
        result = explorar_catalogo.generate_prompt()

        assert "Fase 1" in result
        assert "Fase 2" in result
        assert "Fase 3" in result
        assert "Fase 4" in result

    def test_prompt_contains_catalog_resources(self):
        """Test that prompt references catalog resources."""
        result = explorar_catalogo.generate_prompt()

        assert "catalog://themes" in result
        assert "catalog://publishers" in result

    def test_prompt_contains_search_tools(self):
        """Test that prompt references search tools."""
        result = explorar_catalogo.generate_prompt()

        assert "search_datasets_by_title" in result
        assert "get_datasets_by_theme" in result
        assert "get_datasets_by_keyword" in result


class TestAnalisisDatasetPrompt:
    """Tests for the analisis_dataset prompt."""

    def test_generate_prompt_default(self):
        """Test prompt generation with default values."""
        result = analisis_dataset.generate_prompt()

        assert "Análisis Detallado" in result
        assert "el dataset especificado" in result

    def test_generate_prompt_with_dataset_id(self):
        """Test prompt generation with dataset ID."""
        result = analisis_dataset.generate_prompt(
            dataset_id="test-dataset-123"
        )

        assert "test-dataset-123" in result

    def test_generate_prompt_with_distributions(self):
        """Test prompt with distribution analysis enabled."""
        result = analisis_dataset.generate_prompt(
            dataset_id="test",
            incluir_distribuciones=True
        )

        assert "Análisis de Distribuciones" in result
        assert "get_distributions_by_dataset" in result

    def test_generate_prompt_without_distributions(self):
        """Test prompt with distribution analysis disabled."""
        result = analisis_dataset.generate_prompt(
            dataset_id="test",
            incluir_distribuciones=False
        )

        assert "Análisis de Distribuciones" not in result

    def test_generate_prompt_with_quality_evaluation(self):
        """Test prompt with quality evaluation enabled."""
        result = analisis_dataset.generate_prompt(
            dataset_id="test",
            evaluar_calidad=True
        )

        assert "Evaluación de Calidad" in result
        assert "Puntuación" in result

    def test_generate_prompt_without_quality_evaluation(self):
        """Test prompt with quality evaluation disabled."""
        result = analisis_dataset.generate_prompt(
            dataset_id="test",
            evaluar_calidad=False
        )

        assert "Evaluación de Calidad" not in result

    def test_prompt_contains_methodology(self):
        """Test that prompt contains analysis methodology."""
        result = analisis_dataset.generate_prompt(dataset_id="test")

        assert "Metodología" in result
        assert "Fase 1" in result
        assert "Fase 2" in result

    def test_prompt_contains_use_cases(self):
        """Test that prompt mentions use cases."""
        result = analisis_dataset.generate_prompt(dataset_id="test")

        assert "Casos de uso" in result or "casos de uso" in result


class TestPromptMetadata:
    """Tests for prompt module metadata."""

    def test_buscar_por_tema_metadata(self):
        """Test buscar_por_tema module metadata."""
        assert hasattr(buscar_por_tema, "PROMPT_NAME")
        assert hasattr(buscar_por_tema, "PROMPT_DESCRIPTION")
        assert buscar_por_tema.PROMPT_NAME == "buscar_datos_por_tema"

    def test_datasets_recientes_metadata(self):
        """Test datasets_recientes module metadata."""
        assert hasattr(datasets_recientes, "PROMPT_NAME")
        assert hasattr(datasets_recientes, "PROMPT_DESCRIPTION")
        assert datasets_recientes.PROMPT_NAME == "datasets_recientes"

    def test_explorar_catalogo_metadata(self):
        """Test explorar_catalogo module metadata."""
        assert hasattr(explorar_catalogo, "PROMPT_NAME")
        assert hasattr(explorar_catalogo, "PROMPT_DESCRIPTION")
        assert explorar_catalogo.PROMPT_NAME == "explorar_catalogo"

    def test_analisis_dataset_metadata(self):
        """Test analisis_dataset module metadata."""
        assert hasattr(analisis_dataset, "PROMPT_NAME")
        assert hasattr(analisis_dataset, "PROMPT_DESCRIPTION")
        assert analisis_dataset.PROMPT_NAME == "analisis_dataset"
