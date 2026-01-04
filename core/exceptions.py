"""Base exceptions for API clients."""


class APIClientError(Exception):
    """Base exception for all API client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DatosGobClientError(APIClientError):
    """Exception for datos.gob.es API errors."""

    pass


class INEClientError(APIClientError):
    """Exception for INE API errors."""

    pass


class BOEClientError(APIClientError):
    """Exception for BOE API errors."""

    pass


class AEMETClientError(APIClientError):
    """Exception for AEMET API errors."""

    pass
