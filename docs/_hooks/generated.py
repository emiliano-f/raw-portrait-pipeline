"""MkDocs hook: fill generated sections from the code so the docs cannot drift.

Markers replaced in page Markdown:

- ``<!-- generated:config-reference -->``: every option of ``AppConfig``,
  rendered from :func:`raw_pipeline.config.schema.config_json_schema`.
- ``<!-- generated:config-defaults -->``: the complete default configuration as YAML.
- ``<!-- generated:components -->``: every registered component, per registry.
- ``<!-- generated:import-contracts -->``: the import-linter contracts in ``pyproject.toml``.

The build fails if a config field has no description or if a page holding the
components table does not mention every registered component name.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from mkdocs.exceptions import PluginError

from raw_pipeline.config.models import AppConfig
from raw_pipeline.config.schema import config_json_schema
from raw_pipeline.runtime.components import load_components

Schema = Mapping[str, Any]

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
MARKER = re.compile(r"<!-- generated:(?P<name>[a-z-]+) -->")
CONSTRAINTS = (
    ("minimum", "≥"),
    ("exclusiveMinimum", ">"),
    ("maximum", "≤"),
    ("exclusiveMaximum", "<"),
)


def on_page_markdown(markdown: str, **_: object) -> str:
    """Replace every ``<!-- generated:... -->`` marker with its rendered Markdown."""

    def render(match: re.Match[str]) -> str:
        name = match.group("name")
        renderer = RENDERERS.get(name)
        if renderer is None:
            msg = f"unknown generated section '{name}'; known: {', '.join(sorted(RENDERERS))}"
            raise PluginError(msg)
        return renderer(markdown)

    return MARKER.sub(render, markdown)


# --- configuration reference -------------------------------------------------


def render_config_reference(_: str) -> str:
    """One section per top-level key, one table row per option."""
    schema = config_json_schema()
    defs: Mapping[str, Schema] = schema["$defs"]
    blocks: list[str] = []
    for key, prop in schema["properties"].items():
        sub = _resolve(prop, defs)
        if "properties" in sub:
            blocks.extend(_render_section(key, sub, defs, level=2))
        else:
            blocks.append(f"## `{key}`\n\n{_require_description(key, prop)}\n\n{_table([(key, prop)], defs)}")
    return "\n\n".join(blocks)


def _render_section(path: str, schema: Schema, defs: Mapping[str, Schema], level: int) -> Iterator[str]:
    heading = "#" * level
    leaves: list[tuple[str, Schema]] = []
    nested: list[tuple[str, Schema]] = []
    for key, prop in schema["properties"].items():
        child = _resolve(prop, defs)
        (nested if "properties" in child and "$ref" in prop else leaves).append((key, prop))
    yield f"{heading} `{path}`\n\n{schema.get('description', '').strip()}\n\n{_table(leaves, defs)}"
    for key, prop in nested:
        yield from _render_section(f"{path}.{key}", _resolve(prop, defs), defs, level + 1)
    for key, prop in leaves:
        items = prop.get("items", {})
        if "$ref" in items:
            yield from _render_section(f"{path}.{key}[]", _resolve(items, defs), defs, level + 1)


def _table(fields: list[tuple[str, Schema]], defs: Mapping[str, Schema]) -> str:
    rows = ["| Option | Type | Default | Allowed | Description |", "| --- | --- | --- | --- | --- |"]
    for key, prop in fields:
        rows.append(
            "| "
            + " | ".join(
                (
                    f"`{key}`",
                    _cell(_type(prop, defs)),
                    _cell(_default(prop)),
                    _cell(_constraints(prop)),
                    _cell(_require_description(key, prop)),
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _resolve(prop: Schema, defs: Mapping[str, Schema]) -> Schema:
    ref = prop.get("$ref")
    return defs[ref.rsplit("/", 1)[-1]] if ref else prop


def _type(prop: Schema, defs: Mapping[str, Schema]) -> str:  # noqa: PLR0911 - one return per schema shape
    if "anyOf" in prop:
        return " or ".join(_type(option, defs) for option in prop["anyOf"])
    if "$ref" in prop:
        return "section"
    if "enum" in prop:
        return "one of " + ", ".join(f"`{value}`" for value in prop["enum"])
    if "const" in prop:
        return f"`{prop['const']}`"
    kind = prop.get("type", "any")
    if kind == "array":
        items = prop.get("items", {})
        inner = "entries (see below)" if "$ref" in items else _type(items, defs)
        return f"list of {inner}"
    if kind == "object":
        return "mapping"
    return {"integer": "int", "number": "float", "boolean": "bool", "string": "str"}.get(kind, kind)


def _default(prop: Schema) -> str:
    if "default" not in prop:
        return "required"
    return f"`{json.dumps(prop['default'], ensure_ascii=False)}`"


def _constraints(prop: Schema) -> str:
    sources = [prop, *prop.get("anyOf", [])]
    return ", ".join(f"{symbol} {source[key]}" for source in sources for key, symbol in CONSTRAINTS if key in source)


def _require_description(key: str, prop: Schema) -> str:
    description = str(prop.get("description", "")).strip()
    if not description:
        msg = f"config field '{key}' has no description; add one to its pydantic Field"
        raise PluginError(msg)
    return description


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ") or "—"


def render_config_defaults(_: str) -> str:
    """The default configuration, as a complete YAML file."""
    dumped = yaml.safe_dump(AppConfig().model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    return f"```yaml\n{dumped}```"


# --- registered components -----------------------------------------------------


def render_components(markdown: str) -> str:
    """Table of every registered component; fails if the page omits one of them."""
    registries = load_components()
    rows = ["| Registry | Name | Class |", "| --- | --- | --- |"]
    missing: list[str] = []
    for registry in registries.all():
        for name in registry.names():
            factory = registry._factories[name]  # Registry exposes no factory accessor; read-only here
            owner = getattr(factory, "__self__", factory)
            rows.append(f"| `{registry.kind}` | `{name}` | `{owner.__module__}.{owner.__qualname__}` |")
            if f"`{name}`" not in MARKER.sub("", markdown):
                missing.append(f"{registry.kind}:{name}")
    if missing:
        msg = f"components not documented on the page: {', '.join(missing)}"
        raise PluginError(msg)
    return "\n".join(rows)


# --- import-linter contracts ---------------------------------------------------


def render_import_contracts(_: str) -> str:
    """The import-linter contracts, straight from ``pyproject.toml``."""
    contracts = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["importlinter"]["contracts"]
    blocks: list[str] = []
    for contract in contracts:
        lines = [f"**{contract['name']}** (`{contract['type']}`)", ""]
        if contract["type"] == "layers":
            lines += [f"{i}. `{layer}`" for i, layer in enumerate(contract["layers"], start=1)]
        else:
            lines.append(f"- Source: {', '.join(f'`{m}`' for m in contract['source_modules'])}")
            lines.append(f"- Forbidden: {', '.join(f'`{m}`' for m in contract['forbidden_modules'])}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


RENDERERS: Mapping[str, Callable[[str], str]] = {
    "config-reference": render_config_reference,
    "config-defaults": render_config_defaults,
    "components": render_components,
    "import-contracts": render_import_contracts,
}
