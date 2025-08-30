# python
from __future__ import annotations
from typing import Dict, Optional, Union
from pathlib import Path
import os
import re

_PathLike = Union[str, Path]


def load_env_file(path: _PathLike = ".env", *, overwrite: bool = False, encoding: str = "utf-8") -> Dict[str, str]:
    """
    Load variables from a .env file into the current process environment (os.environ).

    Args:
        path: path to the .env file (default: "./.env").
        overwrite: if True existing environment variables will be replaced.
        encoding: file encoding to use when reading the .env.

    Returns:
        A dict of variables that this function set in os.environ.
    """
    p = Path(path)
    if not p.exists():
        return {}

    # Try to use python-dotenv's parser if available (it handles more edge cases).
    try:
        from dotenv import dotenv_values  # type: ignore

        parsed = dict(dotenv_values(str(p)))  # returns {key: value or None}
    except Exception:
        parsed = {}
        key_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        with p.open("r", encoding=encoding) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # split at first '='
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                if not key_re.match(key):
                    continue
                val = val.strip()
                # strip surrounding quotes if present
                if (len(val) >= 2) and ((val[0] == val[-1]) and val[0] in ("'", '"')):
                    val = val[1:-1]
                parsed[key] = val

    loaded: Dict[str, str] = {}
    for k, v in parsed.items():
        if v is None:
            # dotenv_values returns None for blank or malformed; skip
            continue
        # if not overwriting and key already exists, skip setting it
        if (not overwrite) and (k in os.environ):
            continue
        os.environ[k] = v
        loaded[k] = v

    return loaded


# ---------- pytest tests ----------
def test_load_env_file_basic(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text('GEMINI_API_KEY="sk_test_123"\nFOO=bar\n# comment\n')
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("FOO", None)

    loaded = load_env_file(p)
    assert loaded["GEMINI_API_KEY"] == "sk_test_123"
    assert loaded["FOO"] == "bar"
    assert os.getenv("GEMINI_API_KEY") == "sk_test_123"
    assert os.getenv("FOO") == "bar"


def test_load_env_file_no_overwrite(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text('FOO="new"\n')
    os.environ["FOO"] = "old"
    # should not overwrite
    loaded = load_env_file(p, overwrite=False)
    assert "FOO" not in loaded
    assert os.environ["FOO"] == "old"
    # now allow overwrite
    loaded2 = load_env_file(p, overwrite=True)
    assert loaded2["FOO"] == "new"
    assert os.environ["FOO"] == "new"
