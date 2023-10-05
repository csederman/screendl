try:
    from ._version import __version__  # pyright: ignore
except ImportError:
    try:
        from importlib.metadata import version, PackageNotFoundError

        try:
            __version__ = version("screendl")
        except PackageNotFoundError:
            __version__ = "0.0.0"
    except ImportError:
        __version__ = "0.0.0"
