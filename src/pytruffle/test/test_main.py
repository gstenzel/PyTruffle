import pytruffle
import json


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
