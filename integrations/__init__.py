"""Integrations with external Spanish government APIs."""

from .aemet import AEMETClient, register_aemet_tools
from .boe import BOEClient, register_boe_tools
from .ine import INEClient, register_ine_tools

__all__ = [
    "INEClient",
    "AEMETClient",
    "BOEClient",
    "register_ine_tools",
    "register_aemet_tools",
    "register_boe_tools",
]
