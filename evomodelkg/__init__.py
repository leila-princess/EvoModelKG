"""README extraction self-evolution package."""

__all__ = ["EvolutionConfig", "EvolutionOrchestrator"]


def __getattr__(name: str):
    if name == "EvolutionConfig":
        from evomodelkg.config import EvolutionConfig

        return EvolutionConfig
    if name == "EvolutionOrchestrator":
        from evomodelkg.orchestrator import EvolutionOrchestrator

        return EvolutionOrchestrator
    raise AttributeError(name)
