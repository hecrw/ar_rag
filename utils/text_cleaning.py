import re
import unicodedata


# Arabic diacritics (tashkeel) Unicode range
_DIACRITICS = re.compile(
    "["
    "\u0610-\u061A"  # Arabic signs
    "\u064B-\u065F"  # Arabic fathatan through wavy hamza below
    "\u0670"         # Arabic letter superscript alef
    "\u06D6-\u06DC"  # Arabic small high ligature
    "\u06DF-\u06E4"  # Arabic small high rounded zero
    "\u06E7-\u06E8"  # Arabic small high yeh/noon
    "\u06EA-\u06ED"  # Arabic empty centre low/high stop
    "]+"
)

# Alef variants to normalize
_ALEF_VARIANTS = re.compile("[\u0622\u0623\u0625]")  # آ أ إ → ا

# Tatweel (kashida) - decorative elongation
_TATWEEL = re.compile("\u0640+")

# Multiple whitespace
_MULTI_SPACE = re.compile(r"\s+")

# Multiple newlines
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel)."""
    return _DIACRITICS.sub("", text)


def normalize_alef(text: str) -> str:
    """Normalize alef variants (آ أ إ) to bare alef (ا)."""
    return _ALEF_VARIANTS.sub("\u0627", text)


def remove_tatweel(text: str) -> str:
    """Remove tatweel/kashida characters."""
    return _TATWEEL.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into single space, multiple newlines into double."""
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def clean_arabic_text(text: str, keep_diacritics: bool = True) -> str:
    """Clean and normalize Arabic text.

    Args:
        text: Raw Arabic text.
        keep_diacritics: If True, preserve tashkeel marks (useful for Quran/poetry).
                        If False, strip all diacritics.
    """
    if not text:
        return ""

    # Remove zero-width characters
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\ufeff", "")  # BOM

    # Normalize Unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Remove tatweel
    text = remove_tatweel(text)

    if not keep_diacritics:
        text = remove_diacritics(text)

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text


def clean_html_text(soup_element) -> str:
    """Extract and clean text from a BeautifulSoup element."""
    if soup_element is None:
        return ""

    # Get text with newlines between block elements
    text = soup_element.get_text(separator="\n")

    # Clean up
    text = clean_arabic_text(text)

    return text
