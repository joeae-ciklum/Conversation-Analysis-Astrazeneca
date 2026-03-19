"""
conv_parser.py
══════════════════════════════════════════════════════════════════════
Reads the 3-column Excel file:
    conversation_id | conversation_starttime | conversation_transcript

Every feature is derived purely from the transcript text itself.

CONVERSATION TYPES:
  1. Button Click / Notification Noise — button-click reactions, empty messages
  2. URL Share             — bare URL pasted, no question asked
  3. Feedback / Negative   — "not helpful", rating reactions
  4. System / Meta         — language change, clear history
  5. HR Transactional      — Workday task: title, hours, leave, payslip
  6. IT / Access           — IDM, VPN, password, tickets, systems
  7. Knowledge Query       — genuine question wanting an answer
  8. Science / R&D         — pharma / lab / clinical queries
  9. Multilingual          — real question in non-English language
 10. General / Unclear     — real but unclassifiable

OUTCOME DETECTION (two-tier):
  Tier 1 — Transactional Resolved: Workday completion phrase in bot reply
  Tier 2 — Informational Resolved: bot gave a direct useful answer
           (link, ID, definition, address, steps) — only for real
           conversations, never for button-click noise records

ADVANCED FEATURES:
  - noise_reason: plain-English reason why a noise record was flagged
  - why_other: explains WHY this conversation ended up in "Other"
  - bot_answer_quality: rates the bot's response (Direct/Partial/Deflected/Blocked/Empty)
  - link_relevance: checks whether a URL in the bot reply matches what the user asked for
  - frustration_score: 0-5 signal derived from user message patterns
  - conversation_health: composite Green / Amber / Red rating
"""

import re
import pandas as pd
import numpy as np
from collections import Counter
from urllib.parse import urlparse

# ══════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════

LANG_MARKERS = {
    "Spanish":    ["hola","que","como","donde","puedo","tengo","para","una","los",
                   "las","por","del","hay","también","saber","bueno","saberlo",
                   "reconocido","es bueno","cómo","dónde","también","están"],
    "Swedish":    ["hej","vad","hur","kan","jag","och","att","det","den","bra",
                   "veta","bekräftat","som","till","med","man","har","inte","vill"],
    "German":     ["ich","das","die","der","und","ist","wie","kann","bitte","gut",
                   "zur","kenntnis","habe","eine","nicht","für","haben","werden"],
    "Polish":     ["jak","czy","gdzie","jest","proszę","dobrze","wiedzieć",
                   "konto","nie","mogę","pracownikowi","można","chcę"],
    "Italian":    ["ciao","dove","trovo","posso","grazie","come","non","sono",
                   "per","che","della","questo","voglio"],
    "French":     ["bonjour","merci","comment","puis","avoir","faire","est-ce",
                   "je","vous","nous","pouvez","trouver"],
    "Portuguese": ["bom","saber","como","posso","onde","encontro","obrigado",
                   "por","favor","para","você","está"],
    "Turkish":    ["merhaba","nasıl","için","bir","ile","bu","ne","var","yok",
                   "yapabilir","istiyorum"],
    "Chinese":    ["知道","高兴","感谢","好的","您好","谢谢","请问","怎么","哪里","可以"],
}

def detect_language(text: str) -> str:
    t = str(text).lower()
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii > 3:
        for lang, markers in LANG_MARKERS.items():
            if any(m in t for m in markers):
                return lang
        return "Non-English (other)"
    for lang, markers in LANG_MARKERS.items():
        if sum(1 for m in markers if m in t) >= 2:
            return lang
    return "English"


# ══════════════════════════════════════════════════════════════════════
# CONVERSATION TYPE SIGNALS
# ══════════════════════════════════════════════════════════════════════

NOISE_PATTERNS = [
    r"^good to know[!.]?$", r"^acknowledged[!.]?$",
    r"^bom saber[!.]?$", r"^¡?bueno saberlo[!.]?$",
    r"^gut zu wissen[!.]?$", r"^bra att veta[!.]?$",
    r"^dobrze wiedzieć[!.]?$", r"^bekräftat$",
    r"^zur kenntnis genommen$", r"^reconocido$",
    r"^notado$", r"^ok+[.!]?$", r"^noted[.!]?$",
    r"^great[.!]?$", r"^thanks[.!]?$", r"^thank you[.!]?$",
    r"^ok thanks[.!]?$", r"^:[a-z_]+:.*",
    r"^user took an action$", r"^approve$", r"^deny$",
    r"^\?+$", r"^很高兴知道", r"^\s*$",
    r"^hi+[.!]?$", r"^hello+[.!]?$", r"^hola[.!]?$",
    r"^hej[.!]?$", r"^ciao[.!]?$", r"^bonjour[.!]?$",
]
NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]
URL_RE   = re.compile(r'^https?://\S+$', re.IGNORECASE)
URL_ANY  = re.compile(r'https?://\S+', re.IGNORECASE)

SYSTEM_PATTERNS = [
    "clear history", "clear copilot", "copilot history", "clear chat",
    "set my language", "change language", "language to english",
    "language settings", "ställ in mitt språk", "set language",
    "show me my tickets", "show tickets", "my tickets",
    "user took an action", "reset chat",
]

HR_SIGNALS = [
    "change title", "job title", "update title", "title change", "job role",
    "weekly hours", "change hours", "working hours", "hours per week", "fte",
    "annual leave", "sick leave", "holiday request", "time off", "absence",
    "maternity", "paternity", "payslip", "pay slip", "salary slip",
    "p60", "p45", "p11d", "tax document", "tax form",
    "performance review", "appraisal", "objective", "feedback form",
    "reporting line", "org change", "department change", "transfer",
    "bank detail", "personal detail", "name change", "surname",
    "new joiner", "onboard", "new starter", "first day",
    "workday", "my hr", "hr advisory", "hr business partner",
    "contract", "offer letter", "benefits", "pension", "bonus",
]

IT_SIGNALS = [
    "vpn", "password", "login issue", "access denied", "sso",
    "system access", "idm", "identity manager", "reset password",
    "cannot log", "mfa", "2fa", "authenticator",
    "servicenow", "service now", "incident", "raise a ticket",
    "myaz", "sharepoint", "teams issue", "outlook",
    "laptop", "desktop", "software install", "it support",
]

SCIENCE_SIGNALS = [
    "ng/ml", "mg/ml", "μl", "µl", "dilution", "concentration",
    "protocol", "assay", "clinical trial", "drug compound",
    "molecule", "laboratory", "vlp", "virus-like", "vaccine",
    "antibody", "protein expression", "pharmacology", "chemistry",
    "pipeline drug", "regulatory submission", "clinical data",
]

KNOWLEDGE_SIGNALS = [
    "what is", "what are", "what does", "what do", "what was",
    "who is", "who are", "where can", "where is", "where do",
    "how do", "how can", "how to", "how does", "how many",
    "can you tell", "tell me about", "give me", "show me",
    "find me", "i want to know", "do you know", "is there",
    "link to", "link for", "website for", "send me the link",
    "what's the", "whats the", "prid", "prid id",
    "address of", "office address", "phone number of",
]

FEEDBACK_SIGNALS = [
    "not helpful", "unhelpful", "this is wrong", "wrong answer",
    "that's not right", "incorrect", "bad answer", "useless",
    "terrible", "awful", "rubbish", "waste of time",
]


# ══════════════════════════════════════════════════════════════════════
# NOISE REASON
# ══════════════════════════════════════════════════════════════════════

def classify_noise_reason(msg: str) -> str:
    """
    For conversations classified as Button Click / Notification Noise or
    URL Share, return a plain-English reason explaining exactly why.
    Returns empty string for non-noise conversations.
    """
    m  = str(msg).strip()
    ml = m.lower().strip()

    if not m or ml in ("nan", "none", ""):
        return "Empty session — no message was typed by the user"

    if URL_RE.match(m):
        return "Notification link click — user clicked a URL in a broadcast message; no question was typed"

    for pattern in NOISE_RE:
        if pattern.match(ml):
            if any(x in ml for x in [
                "good to know", "bom saber", "bueno saberlo", "gut zu wissen",
                "bra att veta", "dobrze", "bekräftat", "zur kenntnis",
                "reconocido", "notado", "très bien", "bien noté", "compris",
                "很高兴", "好的",
            ]):
                return "Acknowledgement button click — user clicked 'Good to know!' or equivalent on a broadcast message"
            if ml in ("approve", "deny"):
                return "Workflow approval button click — user approved or denied a workflow request via button"
            if "user took an action" in ml:
                return "System-generated log entry — automated record, not typed by a human"
            if ml.startswith(":"):
                return "Emoji reaction button click — user reacted with an emoji to a bot message"
            if re.match(r"^hi+[.!]?$|^hello+[.!]?$|^hola[.!]?$|^hej[.!]?$|^ciao[.!]?$|^bonjour[.!]?$", ml):
                return "Greeting only — user opened a chat session but typed only a greeting with no follow-up question"
            if any(x in ml for x in ["acknowledged", "noted", "thanks", "thank you", "great", "ok"]):
                return "Acknowledgement reaction — user acknowledged a message without asking a question"
            return "Noise pattern — message matched a known non-question pattern"
    return ""


# ══════════════════════════════════════════════════════════════════════
# WHY DID THIS CONVERSATION LAND IN "OTHER"?
# ══════════════════════════════════════════════════════════════════════

def explain_why_other(conv_type: str, language: str,
                      first_user_msg: str, all_user_text: str,
                      all_bot_text: str, n_total_turns: int) -> str:
    msg = str(first_user_msg).lower().strip()
    bt  = str(all_bot_text).lower()

    if conv_type == "Button Click / Notification Noise":
        return "Button click or notification reaction — no real user question was asked"
    if conv_type == "URL Share":
        return "URL-only message — notification link click, no question asked"
    if conv_type == "Feedback / Negative Reaction":
        return "Feedback signal — user rated bot as unhelpful"
    if language != "English":
        return f"Non-English query ({language}) — classifier trained primarily on English text"

    words = len(msg.split())
    if words <= 2 and n_total_turns <= 3:
        return f"Vague opener ({words} words) — too little context for classifier at turn 1"
    if conv_type == "Science / R&D":
        return "Pharma / R&D topic — genuinely outside HR, IT, Finance scope"
    if conv_type == "System / Meta":
        return "System / meta action — language settings, history clear; no domain intent"
    if conv_type == "HR Transactional":
        return "HR intent detected but may have used phrasing not in classifier training vocabulary"
    if conv_type == "IT / Access":
        return "IT intent detected but may have used phrasing not in classifier training vocabulary"
    if any(p in bt for p in ["i am unsure", "i'm not sure", "unable to help",
                               "please click on get help", "get help below"]):
        return "Bot had no matching knowledge — zero domain signal in both user message and bot reply"
    if n_total_turns >= 5 and words <= 5:
        return "Intent emerged late (turn 3+) — classifier assigned Other at turn 1 before topic was clear"
    if conv_type == "Knowledge Query":
        return "General knowledge question — no specific HR/IT/Finance keyword to trigger correct domain"
    return "Classifier confidence too low — query did not match any trained domain topic closely enough"


# ══════════════════════════════════════════════════════════════════════
# BOT ANSWER QUALITY
# ══════════════════════════════════════════════════════════════════════

def rate_bot_answer(bot_text: str, user_text: str) -> str:
    bt = str(bot_text).lower().strip()
    if not bt or bt in ("nan", "none", ""):
        return "Empty"

    has_direct = any(s in bt for s in [
        "http://", "https://", "your prid", "prid is",
        "the address is", "here is the link", "here's the link",
        "here are the steps", "here is how", "here's how",
        "successfully submitted", "has been processed", "has been completed",
        "has been updated", "has been submitted",
        "stands for", "refers to", "is defined as",
    ])
    has_deflect = any(s in bt for s in [
        "connect you to", "transfer you to", "reach out to",
        "contact your", "get help below", "click on get help",
        "i am unsure of how to help", "please review the provided",
    ])
    has_block = any(s in bt for s in [
        "cannot", "can't", "unable to", "not able to",
        "couldn't find", "could not find", "not supported",
    ])
    has_clarify = any(s in bt for s in [
        "please provide", "could you please", "please confirm",
        "can you clarify", "i need a bit more", "please share",
    ])

    if has_block and not has_direct:   return "Blocked"
    if has_deflect and not has_direct: return "Deflected"
    if has_direct and (has_deflect or has_clarify): return "Partial"
    if has_direct:   return "Direct"
    if has_clarify:  return "Clarifying"
    return "Deflected"


# ══════════════════════════════════════════════════════════════════════
# LINK RELEVANCE CHECK
# ══════════════════════════════════════════════════════════════════════

LINK_RELEVANCE_RULES = [
    (["degreed", "learning", "course", "training"],       ["degreed"]),
    (["servicenow", "ticket", "incident", "raise"],        ["service-now", "servicenow", "esc"]),
    (["workvivo", "news", "post", "announcement"],         ["workvivo"]),
    (["workday", "hr", "payslip", "leave", "absence"],     ["workday", "myhr"]),
    (["idm", "identity", "access", "sso"],                 ["idm", "iam", "identity"]),
    (["teams", "microsoft", "office365"],                  ["microsoft", "teams", "office"]),
    (["benefits", "pension", "benify"],                    ["benify", "benefits", "flex"]),
    (["translate", "az translate"],                        ["translate"]),
    (["brightidea", "innovation", "idea"],                 ["brightidea", "az.bright"]),
    (["nucleus", "video", "media"],                        ["nucleusvideo", "nucleus"]),
    (["forms", "survey", "questionnaire"],                 ["forms.office", "forms.microsoft"]),
    (["prid", "employee id", "employee number"],           ["prid", "employee"]),
]

def check_link_relevance(user_text: str, bot_text: str) -> str:
    ut = str(user_text).lower()
    bt = str(bot_text).lower()

    bot_urls = URL_ANY.findall(bt)
    if not bot_urls:
        asked_for_link = any(s in ut for s in [
            "link", "url", "website", "portal", "where can i find",
            "how do i access", "where is", "send me"
        ])
        return "Not asked" if not asked_for_link else "No link"

    asked_for_link = any(s in ut for s in [
        "link", "url", "website", "portal", "where can i find",
        "where is", "send me", "access", "how do i get to",
        "how do i find", "how to access",
    ])
    if not asked_for_link:
        return "No link"

    combined_urls = " ".join(bot_urls).lower()
    for user_keywords, url_keywords in LINK_RELEVANCE_RULES:
        user_match = any(k in ut for k in user_keywords)
        url_match  = any(k in combined_urls for k in url_keywords)
        if user_match and url_match:   return "Relevant"
        if user_match and not url_match: return "Irrelevant"

    return "Relevant"


# ══════════════════════════════════════════════════════════════════════
# FRUSTRATION SCORE  (0–5)
# ══════════════════════════════════════════════════════════════════════

def frustration_score(turns: list) -> int:
    user_msgs = [t.lower() for sp, t in turns if sp == "User"]
    if not user_msgs:
        return 0

    score = 0
    if len(user_msgs) != len(set(user_msgs)):
        score += 1

    urgency = ["urgent", "asap", "deadline", "eod", "by today",
               "immediately", "as soon as possible", "emergency"]
    if any(u in m for m in user_msgs for u in urgency):
        score += 1

    frustration_words = ["not working", "still not", "already told",
                         "again", "useless", "terrible", "wrong",
                         "ridiculous", "frustrated", "annoying", "waste"]
    if any(f in m for m in user_msgs for f in frustration_words):
        score += 1

    if len(user_msgs) >= 3:
        first_words = len(user_msgs[0].split())
        last_words  = len(user_msgs[-1].split())
        if last_words > first_words * 1.8:
            score += 1

    if len(user_msgs) >= 4:
        score += 1

    return min(score, 5)


# ══════════════════════════════════════════════════════════════════════
# CONVERSATION HEALTH
# ══════════════════════════════════════════════════════════════════════

def conversation_health(outcome: str, bot_answer_quality: str,
                        frustration: int, n_turns: int) -> str:
    if outcome in ("Resolved", "Informational Resolved") and frustration <= 1:
        return "Green"
    if outcome in ("Blocked / Policy", "Geo-Blocked"):
        return "Red"
    if frustration >= 3:
        return "Red"
    if outcome == "Deflected / Unknown" and bot_answer_quality == "Deflected":
        return "Red"
    if frustration == 2 or outcome == "Async / Pending":
        return "Amber"
    if bot_answer_quality in ("Partial", "Clarifying"):
        return "Amber"
    if outcome in ("Resolved", "Informational Resolved"):
        return "Green"
    return "Amber"


# ══════════════════════════════════════════════════════════════════════
# OUTCOME DETECTION  (two-tier)
# ══════════════════════════════════════════════════════════════════════

PHANTOM_BOT_REPLIES = [
    "glad you found this helpful",
    "glad that was helpful",
    "glad i could help",
]

INFORMATIONAL_SIGNALS = [
    "your prid", "your employee id", "your id is", "prid is",
    "here is the link", "here's the link", "the link is",
    "you can access", "you can find it at",
    "http://", "https://",
    "the address is", "office address", "located at",
    "stands for", "refers to", "is defined as",
    "here are the steps", "you can do this by",
    "here is how", "here's how",
    "here is the information", "here's the information",
    "here are the details", "here's the detail",
    "the answer is",
    "aquí está", "aqui esta", "hier ist", "här är",
    "voici le lien", "ecco il link",
]


def detect_outcome(bot_text: str,
                   conv_type: str = None,
                   first_user_msg: str = None) -> str:
    bt  = str(bot_text).lower()
    msg = str(first_user_msg or "").lower().strip()

    if any(p in bt for p in [
        "successfully submitted", "has been submitted", "has been completed",
        "has been updated", "change to be reflected", "has been processed",
        "request has been", "update has been",
    ]):
        return "Resolved"

    if any(p in bt for p in [
        "located in france", "located in germany", "located in spain",
        "located in italy", "located in poland", "located in china",
        "cannot be processed through this system",
        "not supported in your region",
    ]):
        return "Geo-Blocked"

    if any(p in bt for p in [
        "can't", "cannot", "unable to", "i'm unable", "i am unable",
        "not able to", "couldn't find", "could not find",
    ]):
        return "Blocked / Policy"

    if any(p in bt for p in [
        "is still ongoing", "is currently ongoing", "will be notified",
        "process to retrieve", "process to update", "please allow",
    ]):
        return "Async / Pending"

    if any(p in bt for p in [
        "connect you to", "transfer you to", "reach out to", "contact your",
    ]):
        return "Deflected to Human"

    if any(p in bt for p in [
        "please provide", "could you please",
        "i need more information", "need more information from you",
        "please confirm", "can you clarify", "please share",
    ]):
        return "Incomplete - needs info"

    is_phantom_conv  = conv_type in ("Button Click / Notification Noise", "URL Share")
    is_phantom_reply = any(p in bt for p in PHANTOM_BOT_REPLIES)

    if not is_phantom_conv and not is_phantom_reply:
        if any(s in bt for s in INFORMATIONAL_SIGNALS):
            return "Informational Resolved"

    return "Deflected / Unknown"


# ══════════════════════════════════════════════════════════════════════
# SENTIMENT
# ══════════════════════════════════════════════════════════════════════

def detect_sentiment(text: str) -> str:
    t = str(text).lower()
    neg = ["not working", "still not", "i already", "frustrated", "urgent",
           "asap", "deadline", "no one", "annoying", "terrible", "useless",
           "broken", "not helpful", "wrong", "incorrect", "bad", "error",
           "failed", "waste", "ridiculous"]
    pos = ["thank", "thanks", "great", "perfect", "helpful", "brilliant",
           "excellent", "appreciate", "sorted", "resolved", "wonderful"]
    neg_score = sum(1 for w in neg if w in t)
    pos_score = sum(1 for w in pos if w in t)
    if neg_score > pos_score: return "Negative"
    if pos_score > 0:         return "Positive"
    return "Neutral"


# ══════════════════════════════════════════════════════════════════════
# TRANSCRIPT PARSING
# ══════════════════════════════════════════════════════════════════════

def parse_turns(raw_text) -> list:
    if pd.isna(raw_text) or str(raw_text).strip() == "":
        return []
    turns = []
    segments = re.split(r'\n(?=User:|Bot:)', str(raw_text).strip())
    for seg in segments:
        seg = seg.strip()
        if seg.startswith("User:"):
            turns.append(("User", seg[5:].strip()))
        elif seg.startswith("Bot:"):
            turns.append(("Bot", seg[4:].strip()))
    return turns


def classify_conversation_type(first_user_msg: str,
                                all_user_text: str) -> str:
    msg       = str(first_user_msg).strip()
    msg_lower = msg.lower().strip()

    if not msg or msg.lower() in ("nan", "none", ""):
        return "Button Click / Notification Noise"
    for pattern in NOISE_RE:
        if pattern.match(msg_lower):
            return "Button Click / Notification Noise"
    if URL_RE.match(msg.strip()):
        return "URL Share"
    if any(s in msg_lower for s in FEEDBACK_SIGNALS):
        return "Feedback / Negative Reaction"
    if any(s in msg_lower for s in SYSTEM_PATTERNS):
        return "System / Meta"
    if any(s in msg_lower for s in SCIENCE_SIGNALS):
        return "Science / R&D"
    if any(s in msg_lower for s in HR_SIGNALS):
        return "HR Transactional"
    if any(s in msg_lower for s in IT_SIGNALS):
        return "IT / Access"
    lang = detect_language(msg)
    if lang != "English":
        return f"Multilingual ({lang})"
    if any(s in msg_lower for s in KNOWLEDGE_SIGNALS):
        return "Knowledge Query"
    return "General / Unclear"


# ══════════════════════════════════════════════════════════════════════
# DATETIME PARSER
# ══════════════════════════════════════════════════════════════════════

def _safe_parse_dt(val):
    if val is None: return None
    if isinstance(val, pd.Timestamp):
        return val if not pd.isna(val) else None
    if isinstance(val, float):
        if np.isnan(val): return None
        try: return pd.Timestamp("1899-12-30") + pd.Timedelta(days=val)
        except: return None
    s = str(val).strip()
    if s in ("", "nan", "NaT", "None", "NaN"): return None
    try: return pd.to_datetime(s, infer_datetime_format=True)
    except: return None


# ══════════════════════════════════════════════════════════════════════
# MAIN LOAD FUNCTION
# ══════════════════════════════════════════════════════════════════════

def load_excel(filepath: str) -> pd.DataFrame:
    print(f"Reading {filepath}...")
    df = pd.read_excel(filepath, engine="openpyxl")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "transcript" in cl: col_map[col] = "conversation_transcript"
        elif "start" in cl:    col_map[col] = "conversation_starttime"
        elif "id" in cl:       col_map[col] = "conversation_id"
    df = df.rename(columns=col_map)
    df = df.dropna(subset=["conversation_transcript"])
    df = df[df["conversation_transcript"].astype(str).str.strip() != ""]

    print(f"Parsing {len(df):,} transcripts...")
    df["turns"]          = df["conversation_transcript"].apply(parse_turns)
    df["first_user_msg"] = df["turns"].apply(lambda t: next((txt for sp, txt in t if sp == "User"), ""))
    df["last_user_msg"]  = df["turns"].apply(lambda t: next((txt for sp, txt in reversed(t) if sp == "User"), ""))
    df["first_bot_msg"]  = df["turns"].apply(lambda t: next((txt for sp, txt in t if sp == "Bot"), ""))
    df["all_user_text"]  = df["turns"].apply(lambda t: " ".join(txt for sp, txt in t if sp == "User"))
    df["all_bot_text"]   = df["turns"].apply(lambda t: " ".join(txt for sp, txt in t if sp == "Bot"))

    df["n_total_turns"]    = df["turns"].apply(len)
    df["n_user_turns"]     = df["turns"].apply(lambda t: sum(1 for s, _ in t if s == "User"))
    df["n_bot_turns"]      = df["turns"].apply(lambda t: sum(1 for s, _ in t if s == "Bot"))
    df["t1_user_words"]    = df["first_user_msg"].apply(lambda x: len(str(x).split()))
    df["avg_user_words"]   = df["all_user_text"].apply(lambda x: round(len(str(x).split()) / max(1, str(x).count("\n") + 1), 1))
    df["avg_bot_words"]    = df["all_bot_text"].apply(lambda x: round(len(str(x).split()) / max(1, str(x).count("\n") + 1), 1))
    df["total_user_words"] = df["all_user_text"].apply(lambda x: len(str(x).split()))
    df["total_bot_words"]  = df["all_bot_text"].apply(lambda x: len(str(x).split()))

    print("Classifying conversation types...")
    df["conv_type"]    = df.apply(lambda r: classify_conversation_type(r["first_user_msg"], r["all_user_text"]), axis=1)
    df["language"]     = df["first_user_msg"].apply(detect_language)
    df["noise_reason"] = df["first_user_msg"].apply(classify_noise_reason)

    df["outcome"] = df.apply(lambda r: detect_outcome(r["all_bot_text"], conv_type=r["conv_type"], first_user_msg=r["first_user_msg"]), axis=1)
    df["why_other"] = df.apply(lambda r: explain_why_other(r["conv_type"], r["language"], r["first_user_msg"], r["all_user_text"], r["all_bot_text"], r["n_total_turns"]), axis=1)
    df["bot_answer_quality"] = df.apply(lambda r: rate_bot_answer(r["all_bot_text"], r["all_user_text"]), axis=1)
    df["link_relevance"]     = df.apply(lambda r: check_link_relevance(r["all_user_text"], r["all_bot_text"]), axis=1)
    df["frustration_score"]  = df["turns"].apply(frustration_score)
    df["sentiment"]          = df["all_user_text"].apply(detect_sentiment)
    df["conv_health"]        = df.apply(lambda r: conversation_health(r["outcome"], r["bot_answer_quality"], r["frustration_score"], r["n_total_turns"]), axis=1)

    df["has_urgency"]       = df["all_user_text"].str.lower().str.contains(r'\b(urgent|asap|deadline|immediately|by today|by friday|eod)\b', regex=True, na=False)
    user_msgs_list          = df["turns"].apply(lambda t: [txt.lower().strip() for sp, txt in t if sp == "User"])
    df["n_rephrases"]       = user_msgs_list.apply(lambda m: len(m) - len(set(m)))
    df["has_repeated_user"] = df["n_rephrases"] > 0
    df["filler_count"]      = df["all_bot_text"].str.lower().str.count("feel free to").fillna(0).astype(int)
    df["multi_bot_resp"]    = (df["filler_count"] >= 3)
    df["bot_blocked"]       = df["all_bot_text"].str.lower().str.contains("can't|cannot|unable to|couldn't find|could not find", regex=True, na=False)
    df["geo_blocked"]       = df["all_bot_text"].str.lower().str.contains("located in france|located in germany|located in spain|located in italy|located in poland|cannot be processed through this system", regex=True, na=False)
    df["bot_async"]         = df["all_bot_text"].str.lower().str.contains("still ongoing|currently ongoing|will be notified|process to", regex=True, na=False)
    df["contradictory"]     = (df["all_bot_text"].str.lower().str.contains("successfully submitted|has been submitted", na=False) & df["all_bot_text"].str.lower().str.contains("still ongoing|currently ongoing", na=False))

    df["start_datetime"] = df["conversation_starttime"].apply(_safe_parse_dt)
    df["hour"]       = df["start_datetime"].apply(lambda x: int(x.hour) if x else None)
    df["day_of_week"] = df["start_datetime"].apply(lambda x: x.day_name() if x else None)
    df["month"]      = df["start_datetime"].apply(lambda x: x.strftime("%Y-%m") if x else None)
    df["week"]       = df["start_datetime"].apply(lambda x: x.strftime("%Y-W%V") if x else None)
    df["date"]       = df["start_datetime"].apply(lambda x: x.date() if x else None)
    df["time_slot"]  = df["hour"].apply(
        lambda h: ("Morning (6-12)" if h is not None and 6<=h<12
              else "Afternoon (12-17)" if h is not None and 12<=h<17
              else "Evening (17-22)"   if h is not None and 17<=h<22
              else "Night (22-6)"      if h is not None else None))

    df["raw_transcript"] = df["conversation_transcript"].astype(str)

    keep = [
        "conversation_id", "start_datetime", "hour", "time_slot",
        "day_of_week", "month", "week", "date",
        "conv_type", "language", "noise_reason", "outcome", "why_other",
        "bot_answer_quality", "link_relevance",
        "frustration_score", "sentiment", "conv_health",
        "n_user_turns", "n_bot_turns", "n_total_turns",
        "t1_user_words", "avg_user_words", "avg_bot_words",
        "total_user_words", "total_bot_words",
        "n_rephrases", "has_repeated_user", "has_urgency",
        "bot_blocked", "geo_blocked", "bot_async",
        "multi_bot_resp", "contradictory", "filler_count",
        "first_user_msg", "last_user_msg", "first_bot_msg",
        "raw_transcript",
    ]
    keep = [c for c in keep if c in df.columns]
    print(f"Done. {len(df):,} conversations processed.")
    return df[keep]