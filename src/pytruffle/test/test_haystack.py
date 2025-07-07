import pytruffle
import types
from unittest.mock import MagicMock, patch
import sys


def test_haystack_pipeline(monkeypatch, tmp_path):
    class DummyComponent:
        def __call__(self, obj=None, **kwargs):
            return obj

        def output_types(self, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    dummy = types.SimpleNamespace(component=DummyComponent(), Document=dict)
    monkeypatch.setitem(sys.modules, "haystack", dummy)

    with patch("asyncio.run", return_value=[]):
        combined_store_type = pytruffle.get_haystack_interface()
        store = combined_store_type(
            tmp_path,
            openai_client=MagicMock(),
            model="gpt",
            cache_summaries=False,
        )
        assert store.run("test") == []
