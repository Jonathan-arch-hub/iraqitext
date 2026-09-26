"""Tiny terminal demo for iraqitext.

    python examples/cli_translator.py
"""
from iraqitext import IraqiText

it = IraqiText()
text = input("code : ")
print("=" * 70)
print(it.to_fusha(text))
print("=" * 70)
print(it.to_iraqi(text))

# print(it.detect("شلونك شخبارك؟"))
# print(it.normalize("شــــلـونَك؟"))
# print(it.tokenize("شلونك حبيبي؟"))
# print(it.keywords("شلونك حبيبي وين رايح؟"))
