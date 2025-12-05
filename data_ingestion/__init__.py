"""
Data Ingestion Package for Glass Production System
Entry point for all data collection and ingestion modules
"""

# Version info
__version__ = "1.0.0"

# Lazy import of main components to avoid dependency issues
# Import main components only when needed
def __getattr__(name):
    if name == "DataCollector":
        from .data_collector import DataCollector
        return DataCollector
    elif name == "DataRouter":
        from .data_router import DataRouter
        return DataRouter
    elif name == "DataBuffer":
        from .data_router import DataBuffer
        return DataBuffer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "DataCollector",
    "DataRouter",
    "DataBuffer"
]