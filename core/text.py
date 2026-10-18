"""Script-normalised lexical classes for Hindi / Hinglish trailing tokens.

Streaming ASR returns Hindi in Devanagari, romanised Hinglish or a mix (Pulse `hi` is
documented as "bilingual, mixed-script"; Vosk emits Devanagari). The classes below are the
ones Section 3.2 of the problem statement names:

* HOLD (projects more speech): fillers and connectives such as matlab, woh, ki, toh, aur,
  lekin, kyunki, jaise, so, actually, basically.
* VERB (clause-final verb group, the usual SOV turn end): hai, tha, dunga, dijiye, gaya...
* TAG (yields the floor): tag questions and polite closers, na, hai na, theek hai, ji, bas.
* DIGIT: numerals (digit / entity read-outs, Section 3.1: "account number hai [900 ms] 4 5 7 2").
* BACKCHANNEL: haan, hmm, achha, ji, ok, theek (used by the barge-in arbiter and labels).

Every lexicon entry is listed in both scripts rather than transliterated on the fly, because
rule-based Devanagari to Latin transliteration is ambiguous for short function words
(ki / ko / ke all collapse to "k" under consonant-skeleton matching).
"""

from __future__ import annotations

import re
import unicodedata

HOLD = {
    # fillers
    "matlab", "मतलब", "woh", "wo", "वो", "वह", "um", "umm", "uh", "अं", "उम", "आ",
    "yaani", "यानी", "jaise", "जैसे", "basically", "बेसिकली", "actually", "एक्चुअली",
    # connectives / complementisers
    "ki", "कि", "toh", "to", "तो", "aur", "और", "lekin", "लेकिन", "par", "पर", "magar", "मगर",
    "kyunki", "kyonki", "क्योंकि", "क्यूंकि", "agar", "अगर", "phir", "fir", "फिर", "ya", "या",
    "so", "सो", "and", "एंड", "but", "बट", "because", "बिकॉज़", "or", "ऑर",
    # postpositions that cannot end a clause
    "ka", "का", "ke", "के", "ko", "को", "se", "से", "mein", "में", "ne", "ने", "liye", "लिए",
    "mera", "मेरा", "meri", "मेरी", "mere", "मेरे", "apna", "अपना", "apne", "अपने", "the",
}
VERB = {
    "hai", "है", "hain", "हैं", "tha", "था", "thi", "थी", "थे", "hoon", "hun", "हूँ", "हूं",
    "ho", "हो", "gaya", "गया", "gayi", "गई", "gaye", "गए", "diya", "दिया", "liya", "लिया",
    "dijiye", "दीजिए", "दीजिये", "दिजिए", "kijiye", "कीजिए", "कीजिये", "chahiye", "चाहिए",
    "sakte", "सकते", "sakta", "सकता", "sakti", "सकती", "raha", "रहा", "rahi", "रही", "rahe", "रहे",
    "karna", "करना", "karunga", "करूंगा", "करूँगा", "dunga", "दूंगा", "दूँगा", "lunga", "लूंगा",
    "hoga", "होगा", "hogi", "होगी", "milega", "मिलेगा", "bataiye", "बताइए", "बताइये",
}
TAG = {
    "na", "ना", "न", "ji", "जी", "bas", "बस", "theek", "ठीक", "thik", "dhanyavaad", "धन्यवाद",
    "shukriya", "शुक्रिया", "thanks", "थैंक्स", "please", "प्लीज", "प्लीज़", "ok", "okay", "ओके",
    "hai_na", "haan", "हाँ", "हां", "nahi", "नहीं", "achha", "accha", "अच्छा",
}
DIGIT_WORDS = {
    "shunya", "शून्य", "zero", "ज़ीरो", "जीरो", "ek", "एक", "do", "दो", "teen", "तीन", "char", "चार",
    "paanch", "पांच", "पाँच", "chhe", "छह", "छः", "saat", "सात", "aath", "आठ", "nau", "नौ", "das", "दस",
    "one", "वन", "two", "टू", "three", "थ्री", "four", "फोर", "five", "फाइव", "six", "सिक्स",
    "seven", "सेवन", "eight", "एट", "nine", "नाइन", "double", "डबल", "triple", "ट्रिपल",
}
BACKCHANNEL = {
    "haan", "ha", "han", "हाँ", "हां", "हा", "hmm", "hm", "हम्म", "हूं", "achha", "accha", "acha",
    "अच्छा", "ji", "जी", "haanji", "हांजी", "हाँजी", "ok", "okay", "ओके", "theek", "ठीक", "thik", "है",
}
QUESTION_WORDS = {
    "kya", "क्या", "kab", "कब", "kahan", "कहाँ", "कहां", "kaise", "कैसे", "kitna", "कितना", "kitne",
    "कितने", "kaun", "कौन", "kyun", "क्यों",
}
CLASSES = ("HOLD", "VERB", "TAG", "DIGIT", "OTHER", "NONE")

_TOKEN_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_DIGIT_RE = re.compile(r"^[0-9०-९]+$")


def normalise(text: str) -> list[str]:
    """Lower-case, NFC-normalise, strip nukta variants and punctuation, split into tokens."""
    text = unicodedata.normalize("NFC", text.lower()).replace("़", "")
    return _TOKEN_RE.findall(text)


def token_class(tok: str) -> str:
    if _DIGIT_RE.match(tok) or tok in DIGIT_WORDS:
        return "DIGIT"
    if tok in HOLD:
        return "HOLD"
    if tok in TAG:
        return "TAG"
    if tok in VERB or tok.endswith(("ेगा", "ेगी", "ूंगा", "ूँगा", "ega", "egi", "unga")):
        return "VERB"
    return "OTHER"


def trailing_classes(tokens: list[str], n: int = 2) -> list[str]:
    """Classes of the last `n` tokens, most recent first, padded with NONE."""
    cls = [token_class(t) for t in tokens[-n:]][::-1]
    return cls + ["NONE"] * (n - len(cls))


def is_tag_question(tokens: list[str]) -> bool:
    """'hai na', 'theek hai', '... na' at the end of the turn."""
    tail = tokens[-2:]
    return bool(tail) and (
        tail[-1] in {"na", "ना", "न"} or (len(tail) == 2 and tail[0] in {"theek", "ठीक", "thik"} and tail[1] in {"hai", "है"})
    )


def is_backchannel_only(tokens: list[str]) -> bool:
    return bool(tokens) and all(t in BACKCHANNEL for t in tokens)


def ends_with_question(text: str | None) -> bool:
    """Agent-context feature: did the last agent utterance ask something?"""
    if not text:
        return False
    if text.strip().endswith(("?", "؟")):
        return True
    toks = normalise(text)
    return any(t in QUESTION_WORDS for t in toks[-6:])
