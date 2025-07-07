import json
import subprocess
import pytruffle
from unittest.mock import MagicMock, patch


def test_main():
    weird_filenames = [
        "normal.txt",  # Just a regular file
        "file.txt",  # Just a regular file
        " leading-space.txt",  # Leading space
        "trailing-space.txt ",  # Trailing space (usually tricky on Windows Explorer, but can exist)
        "file with spaces.txt",  # Spaces in the name
        ".dotfile",  # Hidden file on Unix-like systems, valid on Windows
        "multiple...dots.txt",  # Multiple dots
        "semi;colon.txt",  # Semicolon
        "equal=sign.txt",  # Equal sign
        "carat^top.txt",  # Caret
        "excl!mation.txt",  # Exclamation mark
        "hash#tag.txt",  # Hash symbol
        "plus+minus-.txt",  # Plus and minus
        "underscores_.txt",  # Underscore
        "résumé.txt",  # Accented characters
        "smiley_😊.txt",  # Emoji / Unicode symbol
        "Пример.txt",  # Cyrillic characters
        "例.txt",  # Japanese characters
        "special(chars){here}.txt",  # Parentheses and curly braces
        "ampersand&file.txt",  # Ampersand
        "percent%file.txt",  # Percent sign
        "file-name@",  # Symbol at the end, no extension
        "singlequote'.txt",  # Single quote
        'doublequote".txt',  # Double quote
    ]
    cls, _ = pytruffle._get_file_selection_class(weird_filenames)
    schema = cls.model_json_schema()
    json.loads(json.dumps(schema))


def test_read_and_format(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("print('hi')\nprint('bye')\n")
    assert pytruffle._read(f, (1, 2)) == "print('bye')\n"

    ff = pytruffle.FileFragmentFormatter(show_line_nums=True, format_as_code_block=False)
    llm_cfg = pytruffle._LLMConfig(
        model="gpt",
        llm_query_args={},
        llm_sum_args={},
        openai_client=MagicMock(),
        prompts=pytruffle.Prompts(),
    )
    frag = pytruffle._FileFragment(f, tmp_path, ff, llm_cfg, lines=(0, 1))
    frag.identifier = "frag"
    out = ff.format(frag)
    assert "print('hi')" in out


def test_store_git_command(monkeypatch, tmp_path):
    completed = subprocess.CompletedProcess([], 0, stdout="")

    def fake_run(args, check, stdout, text):
        fake_run.called = args
        return completed

    monkeypatch.setattr(pytruffle.subprocess, "run", fake_run)
    with patch("asyncio.run"), patch.object(pytruffle, "_Directory") as dummy_dir:
        dummy_hash = type("H", (), {"hex": lambda self: "1" * 64})()
        dummy_dir.return_value.hash_ = dummy_hash
        dummy_dir.return_value.to_dict.return_value = {}
        dummy_dir.return_value.size = 0
        pytruffle.Store(
            tmp_path,
            openai_client=MagicMock(),
            model="gpt",
            cache_summaries=False,
        )
    assert fake_run.called == ["git", "-C", str(tmp_path), "ls-files"]
