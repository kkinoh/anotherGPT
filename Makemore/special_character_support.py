"""

This implementation supports special characters, namely '[...]'.
This includes emojis, speakers, etc.


"""

import re
from pathlib import Path

from data import import_words

_EMOJI_FILE = Path(__file__).with_name("emojis.txt")
SPECIAL_CHARS = import_words(_EMOJI_FILE, encoding_method="utf-8")
SPECIAL_CHARS = list(dict.fromkeys(SPECIAL_CHARS + ["K", "C", "Q"]))

SPECIAL_TO_CHAR = {f"[{name}]": chr(0xE000 + i) for i, name in enumerate(SPECIAL_CHARS)}
CHAR_TO_SPECIAL = {char: emoji for emoji, char in SPECIAL_TO_CHAR.items()}

_patterns = re.compile(
    "|".join(
        re.escape(k) for k in 
        sorted(SPECIAL_TO_CHAR, key = len, reverse=True)
        # Sorting special characters by length in reversed order
        # So characters like 微 and 微笑 will not be misplaced.
    )
)

def encode_specials(words: str) -> str:
    return _patterns.sub(lambda x: SPECIAL_TO_CHAR[x.group(0)], words)

def decode_specials(words: str) -> str:
    return ''.join(CHAR_TO_SPECIAL.get(char, char) for char in words)