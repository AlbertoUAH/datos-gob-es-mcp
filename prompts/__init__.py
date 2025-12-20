"""MCP Prompts para datos.gob.es - Guías de búsqueda detalladas."""

from . import buscar_por_tema, datasets_recientes, explorar_catalogo


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


__all__ = ["register_prompts"]
