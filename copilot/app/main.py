"""Streamlit UI entrypoint for the primer design copilot.

Run with: ``streamlit run copilot/app/main.py`` (per CLAUDE.md).

``streamlit`` is an optional dependency (see the ``copilot`` extra in
``pyproject.toml``), so the import is deferred into ``main()`` rather than
performed at module scope — this module must remain importable without
``streamlit`` installed (e.g. for a plain import smoke-test).
"""

from __future__ import annotations


def render_app() -> None:
    """Render the Streamlit app: design goal input, ranked results, memo.

    Requires ``streamlit``, imported lazily inside this function so the
    module remains importable without the optional ``copilot`` extra
    installed.

    Raises:
        ImportError: If ``streamlit`` is not installed.
    """
    raise NotImplementedError


def main() -> None:
    """Entrypoint invoked by ``streamlit run copilot/app/main.py``."""
    render_app()


if __name__ == "__main__":
    main()
