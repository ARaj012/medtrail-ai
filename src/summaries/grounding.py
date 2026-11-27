"""
Grounding check: is every sentence of a summary supported by the facts it cites?

For each sentence it checks that:
  1. the cited fact ids exist;
  2. every number appears in the cited facts (small rounding allowed);
  3. every date (or year) appears in the cited facts;
  4. every report id (e.g. A-R10) belongs to the cited facts;
  5. condition words (diabetes, kidney disease, ...) appear in the cited facts;
  6. there is no medicine advice (start/stop/increase a medicine, prescribing);
  7. sentences that cite no facts contain no numbers, dates or medical claims;
  8. patient summaries contain no ML estimates; in doctor summaries, ML estimates are
     called estimates and stay in the ML section.

It also checks meaning in two simple ways: a test named in a sentence must be cited, and a
sentence may not call results "within range"/"normal" while citing results outside their range.

Limits (stated honestly): a wrong number that also appears in a cited fact, numbers written
as words, and other changes of meaning are not detected, so summaries still need human review.
"""
import re
import unicodedata


def nfc(text):
    """One Unicode form for all text: Hindi letters with a nukta (e.g. ज़) can be stored in two ways
    that look the same but do not match each other in code."""
    return unicodedata.normalize("NFC", text) if isinstance(text, str) else text

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
MONTH_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"\b(\d{2})-(\d{2})-(\d{4})\b"), lambda m: (int(m[3]), int(m[2]), int(m[1]))),
    (re.compile(rf"\b(\d{{1,2}}) {MONTH_RE},? (\d{{4}})\b", re.I), lambda m: (int(m[3]), MONTHS[m[2][:3].lower()], int(m[1]))),
    (re.compile(rf"\b{MONTH_RE} (\d{{4}})\b", re.I), lambda m: (int(m[2]), MONTHS[m[1][:3].lower()], None)),
    (re.compile(r"\b(19\d{2}|20\d{2})\b"), lambda m: (int(m[1]), None, None)),
]
UNIT_RE = re.compile(r"[A-Za-z\[\]]+/[^\s,;()]+")           # mg/dL, mL/min/1.73m2, U/L
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
REPORT_RE = re.compile(r"\b[A-Z]-R\d{2}\b")

CLAIM_WORDS = ["prediabetes", "diabetes", "hypertension", "anaemia", "anemia", "kidney disease", "renal",
               "liver disease", "hepatitis", "cirrhosis", "fatty liver", "heart disease", "heart failure",
               "cancer", "stroke", "infection", "neuropathy", "retinopathy", "obesity", "thyroid"]
ADVICE_RE = [
    re.compile(r"\byou should (start|stop|take|increase|decrease|reduce|change)\b", re.I),
    re.compile(r"(?<!not )(?<!n't )\b(start|stop|increase|decrease|reduce|double) (taking|your)\b", re.I),
    re.compile(r"\bprescrib(e|es|ing)\b", re.I),
    re.compile(r"\bdose (of|to)\b", re.I),
]
ML_WORDS_RE = re.compile(r"\b(probability|risk estimate|screening estimate|machine learning|ML model)\b", re.I)
ML_TITLE_RE = re.compile(r"\bml\b", re.I)
# A sentence that STATES the ML result (risk level or probability); explanation sentences (SHAP, model notes) do not
RISK_RESULT_RE = re.compile(r"\b(low|moderate|high)\b[\w\s-]{0,15}\brisk\b|\brisk\b[\w\s:-]{0,15}\b(low|moderate|high)\b"
                            r"|\b(probability|risk level|risk category)\b", re.I)
# Words that tell the reader where a value is compared with its range
STATUS_WORDS = {
    "high": re.compile(nfc(r"\b(above|higher|high|raised|over)\b|ज़्यादा|ज्यादा|अधिक|ऊपर|बढ़ा हुआ|उच्च|ऊंचा|ऊँचा"), re.I),
    "low": re.compile(nfc(r"\b(below|lower|low|under)\b|कम|नीचे|घटा हुआ|निम्न"), re.I),
    "near": re.compile(r"\b(close|near)\b|पास|हद|करीब|क़रीब", re.I),
}
CAVEAT_RE = re.compile(r"not meaningful|already (a )?documented", re.I)
# Time spans ("3 years", "3 वर्षों"): they must appear in the cited facts with the same unit
DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)[\s-]*(years?|yrs?|months?|weeks?|days?|वर्षों|वर्ष|सालों|साल|महीनों|महीने|महीना|"
                         r"हफ़्तों|हफ्तों|हफ़्ते|हफ्ते|दिनों|दिन)(?![a-z])", re.I)
DURATION_UNIT = {"y": "year", "व": "year", "स": "year", "m": "month", "म": "month", "w": "week", "ह": "week",
                 "d": "day", "द": "day"}
DEVANAGARI_DIGITS_RE = re.compile(r"[०-९]")

# Hindi words for conditions -> English words that must appear in the cited facts
HINDI_CLAIMS = {  # normalised with nfc() below
    "प्रीडायबिटीज": ["prediabetes"], "मधुमेह": ["diabetes"], "डायबिटीज": ["diabetes"],
    "उच्च रक्तचाप": ["hypertension"], "हाइपरटेंशन": ["hypertension"],
    "एनीमिया": ["anaemia", "anemia"], "खून की कमी": ["anaemia", "anemia"],
    "किडनी की बीमारी": ["kidney disease", "renal"], "गुर्दे की बीमारी": ["kidney disease", "renal"],
    "किडनी रोग": ["kidney disease", "renal"], "गुर्दे का रोग": ["kidney disease", "renal"], "रीनल": ["renal"],
    "लिवर रोग": ["liver disease", "hepatitis", "cirrhosis", "fatty liver"], "रेटिनोपैथी": ["retinopathy"],
    "न्यूरोपैथी": ["neuropathy"],
    "लिवर की बीमारी": ["liver disease", "hepatitis", "cirrhosis", "fatty liver"],
    "हेपेटाइटिस": ["hepatitis"], "सिरोसिस": ["cirrhosis"], "कैंसर": ["cancer", "neoplasm", "carcinoma"],
    "स्ट्रोक": ["stroke"], "दिल की बीमारी": ["heart disease", "coronary", "heart failure"],
    "हृदय रोग": ["heart disease", "coronary", "heart failure"], "संक्रमण": ["infection"],
    "थायराइड": ["thyroid"], "मोटापा": ["obesity"],
}
HINDI_CLAIMS = {nfc(k): v for k, v in HINDI_CLAIMS.items()}
ENGLISH_IN_HINDI_RE = re.compile(r"\b(below|above)\b", re.I)

# "Within range" words, and words that say a value is outside its range (used to catch contradictions)
WITHIN_RE = re.compile(nfc(r"\b(within|normal)\b|सामान्य सीमा (में|के भीतर|के अंदर)|सीमा के भीतर|सीमा के अंदर"
                       r"|सामान्य (है|हैं|था|थी|थे|रहा|रही|रहे)"), re.I)
OUTSIDE_RE = re.compile(nfc(r"\b(above|below|higher|lower|outside|raised|high|low)\b|ऊपर|नीचे|ज़्यादा|ज्यादा|अधिक|बाहर"), re.I)

# Words that frighten patients; allowed only if they are part of a cited fact (e.g. a recorded condition name)
FRIGHTENING = ["dangerous", "alarming", "life-threatening", "fatal", "deadly", "scary", "critical", "severe",
               "emergency", "खतरनाक", "जानलेवा", "घातक", "भयानक", "गंभीर"]
FRIGHTENING = [nfc(w) for w in FRIGHTENING]


def _spelling(text):
    """British and American spellings count as the same word (haemoglobin = hemoglobin)."""
    return text.lower().replace("haem", "hem").replace("anaem", "anem")


def extract(text):
    """Split a text into dates (year, month, day), report ids and numbers."""
    dates = []
    for pattern, convert in DATE_PATTERNS:
        for match in pattern.finditer(text):
            dates.append(convert(match))
        text = pattern.sub(" ", text)
    reports = set(REPORT_RE.findall(text))
    text = REPORT_RE.sub(" ", text)
    text = UNIT_RE.sub(" ", text)
    numbers = [float(n) for n in NUMBER_RE.findall(text)]
    return dates, reports, numbers


def durations(text):
    """{(3.0, 'year'), ...} for every time span written in the text ('51-year-old' -> (51, 'year'))."""
    return {(float(m[1]), DURATION_UNIT[m[2][0].lower()]) for m in DURATION_RE.finditer(text)}


def _evidence(facts):
    dates, reports, numbers, text = set(), set(), set(), ""
    for f in facts:
        d, r, n = extract(f["text"])
        dates |= set(d)
        reports |= set(r) | set(f.get("report_ids", []))
        numbers |= set(n) | set(f.get("numbers", []))
        text += " " + f["text"].lower()
    return dates, reports, numbers, text


def _number_ok(x, allowed):
    for a in allowed:
        candidates = {a, round(a, 1), round(a, 2)}
        if 0 < a <= 1:                                  # a probability may be shown as a percentage
            candidates |= {round(a * 100, 1), round(a * 100)}
        if any(abs(x - c) < 1e-9 for c in candidates):
            return True
    return False


def _date_ok(date, allowed):
    year, month, day = date
    return any(y == year and (month is None or m == month) and (day is None or d == day)
               for (y, m, d) in allowed)


def foreign_letters(text):
    """Letters that are neither Latin nor Devanagari (e.g. Urdu/Arabic letters slipping into Hindi)."""
    return sorted({c for c in text if unicodedata.category(c).startswith("L")
                   and not unicodedata.name(c, "").startswith(("LATIN", "DEVANAGARI"))})


def check_sentence(sentence, facts_by_id, audience, section_title=""):
    problems = []
    text, cited = nfc(sentence.get("text", "")), sentence.get("fact_ids", [])
    foreign = foreign_letters(text)
    if foreign:
        scripts = sorted({unicodedata.name(c).split()[0] for c in foreign})
        problems.append(f"contains letters from another script ({', '.join(scripts)}): {''.join(foreign)}")
    if re.search("[\u0900-\u097F]", text) and ENGLISH_IN_HINDI_RE.search(text):
        problems.append("a Hindi sentence uses the English word 'below' or 'above'; write 'से कम' or 'से ज़्यादा'")
    unknown = [f for f in cited if f not in facts_by_id]
    if unknown:
        problems.append(f"cites unknown fact ids {unknown}")
    facts = [facts_by_id[f] for f in cited if f in facts_by_id]
    dates, reports, numbers = extract(text)
    lowered = text.lower()
    claims = [w for w in CLAIM_WORDS if w in lowered and not (w == "diabetes" and "prediabetes" in lowered and lowered.count("diabetes") == lowered.count("prediabetes"))]
    hindi = [w for w in HINDI_CLAIMS if w in text and not (w == "डायबिटीज" and text.count("डायबिटीज") == text.count("प्रीडायबिटीज"))]
    if DEVANAGARI_DIGITS_RE.search(text):
        problems.append("uses Devanagari digits; numbers must be written with 0-9 so they can be checked")

    if not facts:
        if numbers or dates or reports or claims or hindi:
            problems.append("states numbers, dates or conditions without citing any fact")
    else:
        ok_dates, ok_reports, ok_numbers, fact_text = _evidence(facts)
        problems += [f"number {n:g} is not in the cited facts" for n in numbers if not _number_ok(n, ok_numbers)]
        problems += [f"date {d} is not in the cited facts" for d in dates if not _date_ok(d, ok_dates)]
        problems += [f"report {r} is not in the cited facts" for r in reports if r not in ok_reports]
        allowed_spans = set().union(*(durations(f["text"]) for f in facts))
        problems += [f"states a time span ({n:g} {unit}s) that is not in the cited facts; trend facts cover "
                     f"the last few reports, not years" for n, unit in sorted(durations(text) - allowed_spans)]
        for word in claims:
            same = {"anaemia": "anemia", "anemia": "anaemia"}.get(word)
            if word not in fact_text and not (same and same in fact_text):
                problems.append(f"mentions '{word}', which is not in the cited facts")
        for word in hindi:
            if not any(e in fact_text for e in HINDI_CLAIMS[word]):
                problems.append(f"mentions '{word}', which is not in the cited facts")

    # A test named in the sentence must be one of the cited facts (or be named inside them)
    labs = {f["test_name"]: f for f in facts_by_id.values() if f["kind"] == "lab_observation"}
    cited_tests = {f.get("test_name") for f in facts}
    cited_text = _spelling(" ".join(f["text"] for f in facts))
    remaining = _spelling(text)
    for name in sorted(labs, key=len, reverse=True):              # longest first: "Total Cholesterol" before "Cholesterol"
        pattern = re.compile(rf"(?<![\w/]){re.escape(_spelling(name))}(?![\w/])")
        if pattern.search(remaining):
            remaining = pattern.sub(" ", remaining)
            if name not in cited_tests and _spelling(name) not in cited_text:
                problems.append(f"names {name} but does not cite its facts")

    # Calling results "within range" while citing results that are outside their range
    outside = sorted({f["test_name"] for f in facts if f["kind"] in ("lab_observation", "trend")
                      and labs.get(f.get("test_name"), {}).get("alert")
                      and labs[f["test_name"]].get("range_status") in ("high", "low")})
    if outside and WITHIN_RE.search(text) and not OUTSIDE_RE.search(text):
        problems.append(f"says results are within range or normal, but {', '.join(outside)} "
                        f"{'is' if len(outside) == 1 else 'are'} outside the reference range")

    if any(p.search(text) for p in ADVICE_RE):
        problems.append("gives medicine advice")
    kinds = {f["kind"] for f in facts}
    if audience == "patient" and (ML_WORDS_RE.search(text) or "ml_estimate" in kinds):
        problems.append("patient summary must not include ML risk estimates")
    if audience == "patient":
        fact_text = " ".join(f["text"].lower() for f in facts)
        problems += [f"uses the frightening word '{w}'" for w in FRIGHTENING if w in lowered and w not in fact_text]
    if audience == "doctor" and "ml_estimate" in kinds:
        ml_numbers = {n for f in facts if f["kind"] == "ml_estimate" for n in f.get("numbers", [])}
        states_result = RISK_RESULT_RE.search(text) or any(_number_ok(n, ml_numbers) for n in numbers)
        if states_result and "estimate" not in lowered:
            problems.append("states the ML result (risk level or probability) without calling it a screening estimate")
        if not ML_TITLE_RE.search(section_title):
            problems.append("ML estimate placed outside the ML section")
    return problems


def check_layout(summary, facts_by_id, layout):
    """layout: list of (section title, allowed fact kinds or None). Checks titles, order and which
    kinds of fact each section may cite (e.g. ML estimates only in the ML section)."""
    problems, expected = [], [title for title, _ in layout]
    titles = [s.get("title") for s in summary["sections"]]
    unknown = [t for t in titles if t not in expected]
    if unknown:
        problems.append({"section": None, "sentence": None, "problem": f"unexpected section titles {unknown}; use {expected}"})
    elif titles != [t for t in expected if t in titles]:
        problems.append({"section": None, "sentence": None, "problem": f"sections are not in the order {expected}"})
    allowed = dict(layout)
    for section in summary["sections"]:
        kinds = allowed.get(section.get("title"))
        if not kinds:
            continue
        for sentence in section.get("sentences", []):
            if not isinstance(sentence, dict):
                continue
            wrong = sorted({facts_by_id[f]["kind"] for f in sentence.get("fact_ids", [])
                            if f in facts_by_id and facts_by_id[f]["kind"] not in kinds})
            for kind in wrong:
                places = [title for title, ok in layout if ok and kind in ok]
                problems.append({"section": section["title"], "sentence": sentence.get("text"),
                                 "problem": f"cites a {kind} fact in section '{section['title']}'; "
                                            f"{kind} facts belong in: {'; '.join(places)}"})
    return problems


def check_ml_caveats(summary, facts):
    """When an ML estimate is about a condition that is already documented, the doctor summary must say
    the estimate is not meaningful (the documented diagnosis wins over the model)."""
    problems = []
    for f in facts:
        if f["kind"] != "ml_estimate" or not f.get("already_documented"):
            continue
        texts = [x["text"] for s in summary["sections"] for x in s.get("sentences", [])
                 if isinstance(x, dict) and f["id"] in x.get("fact_ids", [])]
        if texts and not any(CAVEAT_RE.search(x) for x in texts):
            problems.append({"section": None, "sentence": None,
                             "problem": f"does not say that the {f.get('disease')} estimate ({f['id']}) is not meaningful "
                                        f"because the condition is already documented"})
    return problems


def check_key_status(summary, facts, key_names):
    """Patient summaries must say, for each important result, that it is above / below / close to the usual range.
    (Run 3, patient C: every value was given, but never compared with its range.)"""
    problems = []
    for name in key_names:
        lab = next((f for f in facts if f["kind"] == "lab_observation" and f.get("test_name") == name), None)
        if not lab or lab.get("range_status") not in ("high", "low", "near_high", "near_low"):
            continue
        ids = {f["id"] for f in facts if f.get("test_name") == name and f["kind"] in ("lab_observation", "trend")}
        texts = [x["text"] for s in summary["sections"] for x in s.get("sentences", [])
                 if isinstance(x, dict) and ids & set(x.get("fact_ids", []))]
        status = lab["range_status"]
        word = STATUS_WORDS["near" if status.startswith("near") else status]
        if texts and not any(word.search(x) for x in texts):
            where = {"high": "above", "low": "below"}.get(status, "close to a limit of")
            hindi = {"high": "सामान्य सीमा से ज़्यादा", "low": "सामान्य सीमा से कम"}.get(status, "सीमा के पास")
            problems.append({"section": None, "sentence": None,
                             "problem": f"does not say that {name} is {where} the usual range: the sentence that "
                                        f"explains {name} must say so (English: '{where} the usual range', "
                                        f"Hindi: '{hindi}')"})
    return problems


def check_coverage(summary, facts, required_ids):
    """Important facts (e.g. high-severity alerts) that the summary never cites."""
    cited = {f for s in summary["sections"] for x in s.get("sentences", []) if isinstance(x, dict)
             for f in x.get("fact_ids", [])}
    by_id = {f["id"]: f for f in facts}
    return [{"section": None, "sentence": None,
             "problem": f"leaves out an important fact: {fid} ({by_id[fid]['text'][:80]})"}
            for fid in required_ids if fid not in cited and fid in by_id]


def check_summary(summary, facts, audience, layout=None, required_ids=(), key_names=()):
    """Check a whole summary. Returns {'passed', 'sentences', 'supported', 'problems'}."""
    if isinstance(summary, dict) and isinstance(summary.get("sections"), list):
        summary = {**summary, "sections": [
            {**s, "title": nfc(s.get("title")), "sentences": [
                {**x, "text": nfc(x.get("text", ""))} if isinstance(x, dict) else x for x in s.get("sentences", [])]}
            if isinstance(s, dict) else s for s in summary["sections"]]}
    facts = [{**f, "text": nfc(f["text"])} for f in facts]
    if not isinstance(summary, dict) or not isinstance(summary.get("sections"), list) or not summary["sections"]:
        return {"passed": False, "sentences": 0, "supported": 0,
                "problems": [{"sentence": None, "problem": "answer does not have the expected 'sections' structure"}]}
    facts_by_id = {f["id"]: f for f in facts}
    total, supported, problems = 0, 0, []
    for section in summary["sections"]:
        for sentence in section.get("sentences", []):
            if not isinstance(sentence, dict) or not isinstance(sentence.get("fact_ids", []), list):
                problems.append({"sentence": str(sentence)[:120], "problem": "sentence has the wrong structure"})
                total += 1
                continue
            total += 1
            found = check_sentence(sentence, facts_by_id, audience, section.get("title", ""))
            if found:
                problems += [{"section": section.get("title"), "sentence": sentence.get("text"), "problem": p}
                             for p in found]
            else:
                supported += 1
    if layout:
        problems += check_layout(summary, facts_by_id, layout)
    problems += check_coverage(summary, facts, required_ids)
    if audience == "doctor":
        problems += check_ml_caveats(summary, facts)
    else:
        problems += check_key_status(summary, facts, key_names)
    return {"passed": total > 0 and not problems, "sentences": total, "supported": supported, "problems": problems}

