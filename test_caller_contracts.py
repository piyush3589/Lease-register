"""Guards against interface drift: every call site of a documents.* function must match its signature.

Why this exists: streamlit_app.py once called text_from_pdf_bytes(raw, max_pages=..., max_chars=...) and
unpacked (text, errors) after the function had become text_from_pdf_bytes(data) -> str. api.py had tests,
streamlit_app.py had none, so the PDF upload broke without any test failing.

The Streamlit file runs UI code on import, so these tests read its source with `ast` instead of importing it.
"""
import ast
import inspect
import pathlib

import documents

ROOT = pathlib.Path(__file__).parent
CALLERS = ["streamlit_app.py", "api.py", "app.py"]


def _calls_to(tree, func_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == func_name:
                yield node


def test_every_call_to_text_from_pdf_bytes_matches_the_signature():
    sig = inspect.signature(documents.text_from_pdf_bytes)
    checked = 0
    for fname in CALLERS:
        path = ROOT / fname
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _calls_to(tree, "text_from_pdf_bytes"):
            kwargs = {kw.arg: None for kw in call.keywords if kw.arg}
            try:
                sig.bind(*[None] * len(call.args), **kwargs)
            except TypeError as err:
                raise AssertionError(f"{fname}:{call.lineno} calls text_from_pdf_bytes wrongly: {err}") from err
            checked += 1
    assert checked >= 2  # api.py and streamlit_app.py both call it


def test_callers_do_not_unpack_a_tuple_from_text_from_pdf_bytes():
    """The function returns a str. `text, errors = ...` would iterate characters or fail."""
    for fname in CALLERS:
        path = ROOT / fname
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
                if any(True for _ in _calls_to(ast.Module(body=[ast.Expr(node.value)], type_ignores=[]), "text_from_pdf_bytes")):
                    raise AssertionError(f"{fname}:{node.lineno} unpacks a tuple from text_from_pdf_bytes, which returns str")
