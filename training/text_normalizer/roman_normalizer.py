import re


# ============================================================
# ENGLISH WORDS
# Used mainly to prevent English sentences from being
# incorrectly classified as Hindi/Marathi.
# ============================================================

ENGLISH_WORDS = {
    "the", "to", "of", "and", "in", "is", "are", "was", "were",
    "be", "been", "being", "you", "your", "yours", "we", "our",
    "they", "their", "he", "him", "his", "she", "her",
    "it", "this", "that", "these", "those", "for", "from",
    "with", "by", "on", "at", "as", "an", "a", "or", "but",
    "not", "can", "could", "will", "would", "should",
    "have", "has", "had", "do", "does", "did",
    "if", "then", "than", "so", "because",
    "what", "which", "when", "where", "why", "how", "who",
    "one", "two", "three", "four", "five",
    "more", "most", "very", "only", "all", "any", "some",
    "about", "into", "over", "under", "after", "before",
    "also", "just", "like", "well", "now", "here", "there",
    "going", "come", "get", "give", "make", "know", "see",
    "look", "check", "follow", "fill", "click",
    "question", "questions", "answer", "answers",
    "description", "box", "comment", "comments",
    "price", "prices", "target", "ideal", "lead",
    "thought", "split", "details", "properly",
    "first", "last", "next", "previous",
    "income", "tax", "investment", "market",
    "share", "shares", "stock", "stocks",
    "company", "companies", "business",
    "calculation", "calculate", "practical",
    "approach", "file", "files",
}


# ============================================================
# HINDI
# IMPORTANT:
# Do NOT include ambiguous English words such as:
# the, to, me, he, main
# ============================================================

HINDI_WORDS = {
    "aaj": "आज",
    "ab": "अब",
    "aap": "आप",
    "aapko": "आपको",
    "aapka": "आपका",
    "aapki": "आपकी",
    "aapke": "आपके",

    "apna": "अपना",
    "apne": "अपने",
    "apni": "अपनी",

    "hum": "हम",
    "ham": "हम",
    "hume": "हमें",
    "humein": "हमें",
    "humko": "हमको",

    "hai": "है",
    "hain": "हैं",
    "tha": "था",
    "thi": "थी",
    "the": "थे",  # NOTE: handled carefully below

    "ye": "ये",
    "yeh": "यह",
    "vo": "वो",
    "woh": "वो",

    "ka": "का",
    "ki": "की",
    "ke": "के",
    "ko": "को",
    "se": "से",
    "par": "पर",
    "pe": "पे",
    "mein": "में",
    "me": "में",  # protected in token processing
    "baare": "बारे",
    "baat": "बात",

    "kar": "कर",
    "karo": "करो",
    "karna": "करना",
    "karne": "करने",
    "karta": "करता",
    "karte": "करते",
    "karti": "करती",
    "karenge": "करेंगे",
    "karunga": "करूंगा",
    "karungi": "करूंगी",

    "hoga": "होगा",
    "hogi": "होगी",
    "honge": "होंगे",

    "aapko": "आपको",
    "bhi": "भी",
    "hi": "ही",
    "toh": "तो",
    "kyun": "क्यों",
    "kyon": "क्यों",
    "kya": "क्या",
    "kaise": "कैसे",
    "kab": "कब",
    "kahan": "कहाँ",

    "yeh": "यह",
    "yah": "यह",

    "jo": "जो",
    "jis": "जिस",
    "jiska": "जिसका",
    "jisko": "जिसको",

    "ek": "एक",
    "do": "दो",
    "teen": "तीन",

    "aur": "और",
    "ya": "या",
    "lekin": "लेकिन",
    "agar": "अगर",
    "phir": "फिर",
    "sirf": "सिर्फ",
    "bahut": "बहुत",
    "sab": "सब",
    "sabhi": "सभी",

    "samajh": "समझ",
    "samajhna": "समझना",
    "samjhenge": "समझेंगे",

    "dekho": "देखो",
    "dekhen": "देखें",
    "dekhta": "देखता",

    "hona": "होना",
    "hota": "होता",
    "hote": "होते",

    "liye": "लिए",
    "liye": "लिए",
    "wala": "वाला",
    "wale": "वाले",
    "wali": "वाली",

    "agar": "अगर",
    "toh": "तो",

    "income": "इनकम",
    "tax": "टैक्स",
}


# ============================================================
# MARATHI
# ============================================================

MARATHI_WORDS = {
    "aaj": "आज",
    "apan": "आपण",
    "aapan": "आपण",
    "tumhi": "तुम्ही",
    "tumhala": "तुम्हाला",
    "tumcha": "तुमचा",
    "tumchi": "तुमची",
    "tumche": "तुमचे",

    "cha": "चा",
    "chi": "ची",
    "che": "चे",
    "chya": "च्या",

    "kasa": "कसा",
    "kashi": "कशी",
    "kase": "कसे",

    "kay": "काय",
    "kaya": "काय",

    "karaycha": "करायचा",
    "karaychi": "करायची",
    "karayche": "करायचे",
    "karayla": "करायला",
    "karava": "करावा",
    "karavi": "करावी",

    "asel": "असेल",
    "astil": "असतील",
    "ahe": "आहे",
    "aahe": "आहे",
    "aahot": "आहोत",

    "tar": "तर",
    "mag": "मग",
    "te": "ते",
    "tya": "त्या",
    "tyacha": "त्याचा",
    "tyachi": "त्याची",
    "tyache": "त्याचे",

    "apan": "आपण",
    "aplya": "आपल्या",
    "aaplya": "आपल्या",

    "samjun": "समजून",
    "samajun": "समजून",
    "ghenar": "घेणार",
    "ghenaar": "घेणार",

    "sathi": "साठी",
    "madhe": "मध्ये",
    "madhye": "मध्ये",
    "pasun": "पासून",
    "paryant": "पर्यंत",

    "mhanje": "म्हणजे",
    "mhantat": "म्हणतात",
    "mhanun": "म्हणून",

    "kiti": "किती",
    "kuthe": "कुठे",
    "kadhi": "कधी",
    "ka": "का",

    "baddal": "बद्दल",
    "vishayi": "विषयी",

    "pahije": "पाहिजे",
    "pahijech": "पाहिजेच",

    "nahi": "नाही",
    "nahitar": "नाहीतर",

    "aani": "आणि",
    "ani": "आणि",
    "kinva": "किंवा",

    "ek": "एक",
    "don": "दोन",
    "teen": "तीन",

    "file": "फाइल",
}


# ============================================================
# DISTINCTIVE WORDS
# Used for automatic language detection.
# Ambiguous words are intentionally excluded.
# ============================================================

HINDI_DISTINCTIVE = {
    "aaj",
    "aap",
    "aapko",
    "aapka",
    "aapki",
    "aapke",
    "apna",
    "apne",
    "apni",
    "hum",
    "ham",
    "hume",
    "humein",
    "humko",
    "hai",
    "hain",
    "baare",
    "baat",
    "karna",
    "karne",
    "karta",
    "karte",
    "karti",
    "karenge",
    "karunga",
    "karungi",
    "hoga",
    "hogi",
    "honge",
    "bhi",
    "toh",
    "kyun",
    "kyon",
    "kya",
    "kaise",
    "kab",
    "kahan",
    "aur",
    "lekin",
    "agar",
    "phir",
    "sirf",
    "bahut",
    "sabhi",
    "samajh",
    "samajhna",
    "samjhenge",
    "dekho",
    "dekhen",
    "hona",
    "hota",
    "hote",
    "liye",
    "wala",
    "wale",
    "wali",
}


MARATHI_DISTINCTIVE = {
    "apan",
    "aapan",
    "tumhi",
    "tumhala",
    "tumcha",
    "tumchi",
    "tumche",
    "cha",
    "chi",
    "che",
    "chya",
    "kasa",
    "kashi",
    "kase",
    "kay",
    "karaycha",
    "karaychi",
    "karayche",
    "karayla",
    "karava",
    "karavi",
    "asel",
    "astil",
    "ahe",
    "aahe",
    "aahot",
    "tar",
    "mag",
    "tya",
    "tyacha",
    "tyachi",
    "tyache",
    "aplya",
    "aaplya",
    "samjun",
    "samajun",
    "ghenar",
    "ghenaar",
    "sathi",
    "madhe",
    "madhye",
    "pasun",
    "paryant",
    "mhanje",
    "mhantat",
    "mhanun",
    "kiti",
    "kuthe",
    "kadhi",
    "baddal",
    "vishayi",
    "pahije",
    "nahi",
    "nahitar",
    "aani",
    "ani",
    "kinva",
}


# ============================================================
# PROTECTED WORDS
# These are common English words that must NEVER be converted
# by the Roman Hindi/Marathi dictionaries.
# ============================================================

PROTECTED_ENGLISH = {
    "the",
    "to",
    "me",
    "he",
    "main",
    "is",
    "and",
    "of",
    "in",
    "on",
    "for",
    "from",
    "with",
    "by",
    "a",
    "an",
    "or",
    "but",
    "if",
    "that",
    "this",
    "these",
    "those",
    "you",
    "your",
    "we",
    "our",
    "they",
    "their",
    "it",
    "was",
    "were",
    "are",
    "will",
    "can",
    "could",
    "would",
    "should",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
    "one",
    "two",
    "three",
    "first",
    "last",
    "next",
    "more",
    "most",
    "very",
    "only",
    "all",
    "any",
    "some",
    "just",
    "about",
    "into",
    "over",
    "under",
    "after",
    "before",
    "also",
    "going",
    "come",
    "get",
    "give",
    "make",
    "know",
    "see",
    "look",
    "check",
    "follow",
    "fill",
    "click",
    "question",
    "answer",
    "description",
    "box",
    "comment",
    "comments",
    "price",
    "prices",
    "target",
    "ideal",
    "lead",
    "thought",
    "split",
    "details",
    "properly",
    "income",
    "tax",
    "investment",
    "market",
    "share",
    "shares",
    "stock",
    "stocks",
    "company",
    "companies",
    "business",
    "calculation",
    "calculate",
    "practical",
    "approach",
    "file",
    "files",
}


# ============================================================
# TOKENIZATION
# ============================================================

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+")


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def has_latin(text):
    return bool(re.search(r"[A-Za-z]", text))


def has_devanagari(text):
    return bool(re.search(r"[\u0900-\u097F]", text))


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):
    """
    Conservative automatic language detection.

    Returns:
        "en"       -> English
        "hi"       -> Hindi
        "mr"       -> Marathi
        "unknown"  -> do not normalize
    """

    if not text or not text.strip():
        return "unknown"

    # Native-script text should not be modified.
    if not has_latin(text):
        return "unknown"

    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    ]

    if not words:
        return "unknown"

    # English score
    english_hits = sum(1 for w in words if w in ENGLISH_WORDS)

    # Distinctive Hindi / Marathi scores
    hindi_hits = sum(1 for w in words if w in HINDI_DISTINCTIVE)
    marathi_hits = sum(1 for w in words if w in MARATHI_DISTINCTIVE)

    total = len(words)

    english_ratio = english_hits / total
    hindi_ratio = hindi_hits / total
    marathi_ratio = marathi_hits / total

    # --------------------------------------------------------
    # Strong pure English
    # --------------------------------------------------------

    if english_hits >= 2 and hindi_hits == 0 and marathi_hits == 0:
        return "en"

    if english_ratio >= 0.35 and hindi_hits == 0 and marathi_hits == 0:
        return "en"

    # --------------------------------------------------------
    # Hindi
    # Need meaningful Hindi evidence.
    # --------------------------------------------------------

    if hindi_hits >= 2 and hindi_hits >= marathi_hits + 1:
        return "hi"

    if hindi_hits >= 3 and hindi_ratio >= 0.20:
        return "hi"

    # --------------------------------------------------------
    # Marathi
    # --------------------------------------------------------

    if marathi_hits >= 2 and marathi_hits >= hindi_hits + 1:
        return "mr"

    if marathi_hits >= 3 and marathi_ratio >= 0.20:
        return "mr"

    # --------------------------------------------------------
    # Mixed cases
    # --------------------------------------------------------

    if hindi_hits >= 2 and hindi_hits > marathi_hits:
        return "hi"

    if marathi_hits >= 2 and marathi_hits > hindi_hits:
        return "mr"

    # If only English evidence exists, preserve English.
    if english_hits >= 1 and hindi_hits == 0 and marathi_hits == 0:
        return "en"

    # Not enough evidence -> safest option is unchanged.
    return "unknown"


# ============================================================
# TOKEN NORMALIZATION
# ============================================================

def normalize_tokens(text, dictionary):
    tokens = tokenize(text)
    output = []

    for token in tokens:
        # Preserve spaces, punctuation, numbers, etc.
        if not re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token):
            output.append(token)
            continue

        original = token
        lower = original.lower()

        # ----------------------------------------------------
        # Never touch protected English words.
        # ----------------------------------------------------

        if lower in PROTECTED_ENGLISH:
            output.append(original)
            continue

        # ----------------------------------------------------
        # Convert only if explicitly present in dictionary.
        # ----------------------------------------------------

        if lower in dictionary:
            output.append(dictionary[lower])
        else:
            output.append(original)

    return "".join(output)


# ============================================================
# HINDI NORMALIZATION
# ============================================================

def normalize_hindi(text):
    if not text or not text.strip():
        return text

    # Native Hindi/Devanagari -> untouched.
    if not has_latin(text):
        return text

    return normalize_tokens(text, HINDI_WORDS)


# ============================================================
# MARATHI NORMALIZATION
# ============================================================

def normalize_marathi(text):
    if not text or not text.strip():
        return text

    # Native Devanagari -> untouched.
    if not has_latin(text):
        return text

    return normalize_tokens(text, MARATHI_WORDS)


# ============================================================
# MAIN NORMALIZER
# ============================================================

def normalize_text(text, language_hint=None):
    """
    Main normalization entry point.

    language_hint:
        "en" -> preserve completely
        "hi" -> Hindi normalization
        "mr" -> Marathi normalization
        None  -> conservative automatic detection
    """

    if not text or not text.strip():
        return text

    # --------------------------------------------------------
    # Native script -> never modify.
    # --------------------------------------------------------

    if not has_latin(text):
        return text

    # --------------------------------------------------------
    # Explicit dataset language hint.
    # --------------------------------------------------------

    if language_hint == "en":
        return text

    if language_hint == "hi":
        return normalize_hindi(text)

    if language_hint == "mr":
        return normalize_marathi(text)

    # --------------------------------------------------------
    # Automatic detection.
    # --------------------------------------------------------

    detected = detect_language(text)

    if detected == "hi":
        return normalize_hindi(text)

    if detected == "mr":
        return normalize_marathi(text)

    # English / unknown -> safest option is unchanged.
    return text


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    examples = [
        "Aaj hum income tax ke baare mein baat karenge.",
        "Aapko apna ITR file karna hai.",
        "Aaj apan income tax cha calculation kasa karaycha te samjun ghenar aahot.",
        "Tumhala ITR file karaycha asel tar first tumhi your income details properly check kara.",
        "three to one split",
        "comment and in the description box you just have to fill",
        "RBI's target is 4% and them",
        "He thought PPP",
        "what is it going to lead to",
    ]

    print("=" * 70)
    print("ROMAN NORMALIZER V0.4 TEST")
    print("=" * 70)

    for text in examples:
        detected = detect_language(text)
        normalized = normalize_text(text)

        print("\nINPUT     :", text)
        print("DETECTED  :", detected)
        print("OUTPUT    :", normalized)