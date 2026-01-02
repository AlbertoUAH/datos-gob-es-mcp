"""MCP Prompts para datos.gob.es - Guias de busqueda detalladas."""

from . import buscar_por_tema, datasets_recientes, explorar_catalogo, analisis_dataset, guia_herramientas, buscar_estadisticas


def register_prompts(mcp):
    """
    Registra todos los prompts en el servidor MCP.

    Args:
        mcp: Instancia de FastMCP donde registrar los prompts
    """

    @mcp.prompt()
    def prompt_buscar_datos_por_tema(
        tema: str = "economia",
        formato: str = "csv",
        max_resultados: int = 10
    ) -> str:
        """
        Búsqueda guiada de datasets por temática y formato.

        Este prompt te guía para encontrar datasets de una temática específica
        disponibles en un formato determinado.

        Args:
            tema: Temática a buscar (economia, salud, educacion, hacienda, etc.)
            formato: Formato deseado (csv, json, xml, xlsx, rdf)
            max_resultados: Número máximo de resultados a mostrar
        """
        return buscar_por_tema.generate_prompt(tema, formato, max_resultados)

    @mcp.prompt()
    def prompt_datasets_recientes(
        dias: int = 30,
        tema: str | None = None,
        max_resultados: int = 15
    ) -> str:
        """
        Búsqueda de datasets actualizados recientemente.

        Este prompt te guía para encontrar los datasets que han sido
        actualizados o publicados en los últimos días.

        Args:
            dias: Número de días hacia atrás para buscar actualizaciones
            tema: Temática opcional para filtrar resultados
            max_resultados: Número máximo de resultados a mostrar
        """
        return datasets_recientes.generate_prompt(dias, tema, max_resultados)

    @mcp.prompt()
    def prompt_explorar_catalogo(
        interes: str = "datos económicos de España"
    ) -> str:
        """
        Exploración guiada del catálogo de datos abiertos.

        Este prompt proporciona una guía completa para explorar el catálogo
        de datos.gob.es y encontrar datasets relevantes para un interés específico.

        Args:
            interes: Descripción del área de interés para la búsqueda
        """
        return explorar_catalogo.generate_prompt(interes)

    @mcp.prompt()
    def prompt_analisis_dataset(
        dataset_id: str = "",
        incluir_distribuciones: bool = True,
        evaluar_calidad: bool = True
    ) -> str:
        """
        Análisis detallado de un dataset específico.

        Este prompt te guía para realizar un análisis exhaustivo de un dataset
        concreto del catálogo datos.gob.es, incluyendo metadatos, distribuciones,
        calidad de datos y posibles usos.

        Args:
            dataset_id: Identificador del dataset a analizar
            incluir_distribuciones: Si incluir análisis de distribuciones
            evaluar_calidad: Si incluir evaluación de calidad de datos
        """
        return analisis_dataset.generate_prompt(
            dataset_id, incluir_distribuciones, evaluar_calidad
        )

    @mcp.prompt()
    def prompt_guia_herramientas(
        tool_category: str = "all",
        include_examples: bool = True
    ) -> str:
        """
        Guia interactiva de herramientas MCP con ejemplos de uso.

        Este prompt proporciona documentacion completa de todas las herramientas
        disponibles del servidor MCP de datos.gob.es, organizadas por categoria
        y con ejemplos practicos de uso.

        Args:
            tool_category: Categoria a mostrar ('all', 'search', 'metadata', 'external', 'utilities')
            include_examples: Si incluir ejemplos de uso practicos
        """
        return guia_herramientas.generate_prompt(tool_category, include_examples)

    @mcp.prompt()
    def prompt_buscar_estadisticas(
        tema: str = "empleo",
        incluir_ine: bool = True,
        incluir_datos_gob: bool = True
    ) -> str:
        """
        Busqueda guiada de estadisticas oficiales espanolas.

        Este prompt guia para buscar datos estadisticos en el INE (Instituto
        Nacional de Estadistica) y en datos.gob.es. El INE es la fuente
        PRINCIPAL de estadisticas oficiales en Espana.

        Args:
            tema: Tema estadistico (empleo, poblacion, precios, turismo, PIB, etc.)
            incluir_ine: Buscar en INE - fuente principal de estadisticas (default: True)
            incluir_datos_gob: Buscar tambien en datos.gob.es (default: True)
        """
        return buscar_estadisticas.generate_prompt(tema, incluir_ine, incluir_datos_gob)


__all__ = ["register_prompts"]
