"""Generic name -> component registry with entry-point plugin discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, cast

from raw_pipeline.core.errors import RegistryError

T = TypeVar("T")
C = TypeVar("C", bound=type[Any])

ENTRY_POINT_PREFIX = "raw_pipeline."


@dataclass(frozen=True)
class BuildEnv:
    """Everything a component factory may need to build itself.

    Attributes:
        config: The validated root configuration. Components read their
            section through :func:`raw_pipeline.config.app_config`.
        output_dir: Directory where outputs of this run are written.
        overwrite: Whether existing output files may be replaced.
        options: Free-form runtime options (e.g. CLI flags) keyed by name.
    """

    config: object
    output_dir: Path
    overwrite: bool = False
    options: Mapping[str, object] = field(default_factory=dict)


class Buildable(Protocol):
    """A registered class: it knows how to construct itself from a :class:`BuildEnv`."""

    @classmethod
    def from_env(cls, env: BuildEnv) -> object:
        """Build an instance, injecting collaborators according to ``env.config``."""
        ...


class Registry(Generic[T]):
    """Maps configuration names to component classes of one extension point.

    Args:
        kind: Extension point name, e.g. ``"stages"``. Plugins register under
            the entry-point group ``raw_pipeline.<kind>``.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[[BuildEnv], T]] = {}

    @property
    def entry_point_group(self) -> str:
        """Entry-point group scanned by :meth:`load_entry_points`."""
        return f"{ENTRY_POINT_PREFIX}{self.kind}"

    def register(self, name: str) -> Callable[[C], C]:
        """Class decorator registering a :class:`Buildable` class under ``name``.

        Raises:
            RegistryError: If ``name`` is already taken.
        """

        def decorator(cls: C) -> C:
            self.add(name, cast("Callable[[BuildEnv], T]", cast("Buildable", cls).from_env))
            return cls

        return decorator

    def add(self, name: str, factory: Callable[[BuildEnv], T]) -> None:
        """Register ``factory`` under ``name`` (decorator-free variant)."""
        if name in self._factories:
            msg = f"{self.kind}: '{name}' is already registered"
            raise RegistryError(msg)
        self._factories[name] = factory

    def names(self) -> list[str]:
        """Return the registered names, sorted."""
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def create(self, name: str, env: BuildEnv) -> T:
        """Build the component registered as ``name``.

        Raises:
            RegistryError: If ``name`` is unknown; the message lists the options.
        """
        try:
            factory = self._factories[name]
        except KeyError:
            msg = f"unknown {self.kind} '{name}'; registered: {', '.join(self.names()) or '(none)'}"
            raise RegistryError(msg) from None
        return factory(env)

    def load_entry_points(self) -> list[str]:
        """Import third-party plugins of this extension point and return their names.

        Each entry point must reference a :class:`Buildable` class; its entry
        point name is the registration name.
        """
        loaded: list[str] = []
        for ep in entry_points(group=self.entry_point_group):
            if ep.name in self._factories:
                continue
            self.add(ep.name, cast("Callable[[BuildEnv], T]", cast("Buildable", ep.load()).from_env))
            loaded.append(ep.name)
        return loaded
