import re # To search for emojis and substitute them.

WECHAT_EMOJIS = ['强']
# And probably others.

EMOJI_TO_CHAR = {f"[{name}]": chr(0xE000 + i) for i, name in enumerate(WECHAT_EMOJIS)}
CHAR_TO_EMOJI = {char: emoji for emoji, char in EMOJI_TO_CHAR.items()}

_patterns = re.compile(
    "|".join(
        re.escape(k) for k in 
        sorted(EMOJI_TO_CHAR, key = len, reverse=True)
        # Sorting emojis by length in reversed order, so characters like 微 and 微笑 will not be misplaced.
    )
)

def encode_emojis(words: str) -> str:
    return _patterns.sub(lambda x: EMOJI_TO_CHAR[x.group(0)], words)

def decode_emojis(words: str) -> str:
    return ''.join(CHAR_TO_EMOJI.get(char, char) for char in words)