"""Shims so the converted notebooks run outside Colab.

* ``userdata.get(name)`` used to read Colab Secrets; here it reads environment
  variables (see ``.env.example``) and raises a clear error if one is missing.
* ``display(obj)`` is the IPython convenience; here it falls back to ``print``.
"""
import os


class _UserData:
    def get(self, name, default=None):
        val = os.environ.get(name, default)
        if val is None:
            raise KeyError(
                f"Environment variable {name!r} is not set. The original notebook read it "
                "from Colab Secrets; export it in your shell or put it in a .env file."
            )
        return val


userdata = _UserData()

try:  # inside Jupyter / IPython the real display is available
    from IPython.display import display  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    def display(*objs, **kwargs):  # noqa: D103
        for o in objs:
            print(o)
