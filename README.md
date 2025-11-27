# Real-Time Sign Language Translator

## Prerequisites
* **uv** (Required): [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
* **Make** (Recommended): [https://community.chocolatey.org/packages/make](https://community.chocolatey.org/packages/make) (Windows users)

## Quick Reference

| Goal | Command | When to run? |
| :--- | :--- | :--- |
| **Setup / Update** | `make setup` | First time, after `git pull`, or switching branches. |
| **Run App** | `make run` | To start the application. |
| **Add Library** | `uv add <package>` | To install new tools. **DO NOT** use `pip install`. |

**IMPORTANT:** If you use `uv add`, you MUST commit both `pyproject.toml` and `uv.lock`.

## Windows / No-Make Users
If you cannot install `Make`, run these raw commands instead:

* **Setup:** `uv sync`
* **Run:** `uv run streamlit run src/app.py`