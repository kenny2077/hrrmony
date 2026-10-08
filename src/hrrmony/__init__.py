"""hrrmony: turn any song into a Minecraft villager cover."""

__version__ = "0.3.0"

__all__ = ["__version__", "CoverOptions", "CoverResult", "make_cover"]


def __getattr__(name: str):  # lazy: keep `import hrrmony` light
    if name in {"CoverOptions", "CoverResult", "make_cover"}:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
