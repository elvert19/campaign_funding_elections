"""
CivicTech Open Finance Tracker — Kenya
=======================================
Politician profile system with searchable directory, per-profile financial
records, community reviews, shadow voting, and AI anomaly detection.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import math, sqlite3, requests, uuid, time, io, csv, json, re

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CivicTech – Kenya Finance Tracker",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SECRETS / OPEN-CORE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def _s(key, default):
    try:    return st.secrets[key]
    except: return default

def _ss(section, key, default):
    try:    return st.secrets[section][key]
    except: return default

TEAM_PASSWORD      = _s("team_password", "CivicTest2026!")
MIN_REVIEW_LEN     = int(_ss("security", "min_review_length",    80))
TRUST_VOTE_GATE    = int(_ss("security", "trust_vote_gate",       5))
TRUST_UPLOAD_GATE  = int(_ss("security", "trust_upload_gate",    15))
TRUST_BONUS        = int(_ss("security", "trust_bonus_review",    1))
RATE_LIMIT_SECS    = float(_ss("security", "rate_limit_secs",   3.0))
NLP_RED_FLAGS      = _ss("security", "nlp_red_flags", [
    "legal defense", "luxury", "v8", "harambee", "miscellaneous",
    "unspecified", "cash advance", "private jet", "slush", "facilitation",
])
DB_PATH = "campaign_finance.db"

# Fraud-prevention thresholds (hidden in secrets)
MAX_SUBMISSIONS_PER_HOUR = int(_ss("security","max_submissions_per_hour", 50))
MAX_AMOUNT_KES           = float(_ss("security","max_single_amount_kes",  500_000_000))
MIN_AMOUNT_KES           = float(_ss("security","min_single_amount_kes",  1.0))
QUARANTINE_TRUST_GATE    = int(_ss("security","quarantine_trust_gate",    3))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PASSWORD BOUNCER
# ─────────────────────────────────────────────────────────────────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return
    st.markdown("## 🔒 CivicTech – Restricted Access")
    st.caption("This testing environment is restricted to authorised team members.")
    pwd = st.text_input("Team password", type="password", key="pwd_in")
    if st.button("Log in", key="pwd_btn"):
        if pwd == TEAM_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ Incorrect password.")
    st.stop()

check_password()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        # ── Politician profiles ───────────────────────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS politicians (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                slug          TEXT    UNIQUE NOT NULL,
                full_name     TEXT    NOT NULL,
                role          TEXT,
                party         TEXT,
                county        TEXT,
                constituency  TEXT,
                photo_url     TEXT,
                bio           TEXT,
                created_by    TEXT,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")

        # ── Append-only expense ledger (linked to politician slug) ────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id        TEXT    NOT NULL,
                politician_slug  TEXT    NOT NULL,
                original_amount  REAL    NOT NULL,
                currency         TEXT    NOT NULL DEFAULT 'KES',
                description      TEXT    NOT NULL,
                version          INTEGER NOT NULL DEFAULT 1,
                is_active        INTEGER NOT NULL DEFAULT 1,
                submitted_by     TEXT,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")

        # ── Reviews (linked to politician slug OR specific record_id) ─────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_slug  TEXT    NOT NULL,
                record_id        TEXT,
                anon_id          TEXT    NOT NULL,
                title            TEXT    NOT NULL DEFAULT 'Finding',
                findings         TEXT    NOT NULL,
                upvotes          INTEGER NOT NULL DEFAULT 0,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")

        # ── Anonymous citizen trust scores ────────────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS citizens (
                anon_id     TEXT    PRIMARY KEY,
                trust_score INTEGER NOT NULL DEFAULT 0,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")

        # ── Immutable audit log ───────────────────────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                action    TEXT    NOT NULL,
                actor_id  TEXT,
                detail    TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")

        # ── Quarantine — holds unverified submissions for moderator review ────
        c.execute("""
            CREATE TABLE IF NOT EXISTS quarantine (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_slug  TEXT    NOT NULL,
                original_amount  REAL    NOT NULL,
                currency         TEXT    NOT NULL DEFAULT 'KES',
                description      TEXT    NOT NULL,
                submitted_by     TEXT,
                reason           TEXT,
                status           TEXT    NOT NULL DEFAULT 'pending',
                reviewed_by      TEXT,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")

        # ── Submission rate tracking (per anon_id, rolling 1-hour window) ────
        c.execute("""
            CREATE TABLE IF NOT EXISTS submission_counts (
                anon_id    TEXT    NOT NULL,
                window_start DATETIME NOT NULL,
                count      INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (anon_id, window_start)
            )""")

        conn.commit()


def log(action, actor=None, detail=None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO audit_log (action,actor_id,detail) VALUES (?,?,?)",
            (action, actor, detail))
        conn.commit()

# ─────────────────────────────────────────────────────────────────────────────
# FRAUD PREVENTION LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _name_tokens(name: str) -> set:
    """Lower-case word tokens from a name, ignoring short words."""
    return {w for w in re.split(r"[\s\-]+", name.lower()) if len(w) > 2}

def name_similarity(a: str, b: str) -> float:
    """
    Simple token-overlap similarity 0.0–1.0.
    1.0 = identical tokens, 0.0 = no shared tokens.
    """
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))

def detect_wrong_politician(file_df: pd.DataFrame,
                             target_slug: str,
                             target_name: str) -> list[dict]:
    """
    Scans every row of an uploaded file.
    If a row contains a 'Politician' column whose value does NOT match the
    target profile, flag it as a potential cross-contamination attempt.

    Returns list of {row_index, found_name, similarity, risk_level}.
    """
    warnings = []
    if "Politician" not in file_df.columns:
        return warnings

    for i, row in file_df.iterrows():
        found = str(row.get("Politician","")).strip()
        if not found or found == "—":
            continue
        sim = name_similarity(found, target_name)
        # If the name in the file is clearly someone else (sim < 0.3)
        # and it is not blank/generic
        if sim < 0.3 and len(found) > 3:
            risk = "HIGH" if sim < 0.1 else "MEDIUM"
            warnings.append({
                "row":       i + 2,          # +2 for 1-index + header row
                "found":     found,
                "similarity": round(sim, 2),
                "risk":      risk,
            })
    return warnings

def check_duplicate_expense(politician_slug: str,
                             amount: float,
                             description: str,
                             currency: str) -> dict | None:
    """
    Checks if an identical (amount + description + currency) record already
    exists for this politician in the last 30 days.
    Returns the duplicate row as a dict, or None if clean.
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT record_id, timestamp FROM expenses
            WHERE politician_slug = ?
              AND original_amount  = ?
              AND lower(description) = lower(?)
              AND currency        = ?
              AND is_active       = 1
              AND timestamp >= datetime('now', '-30 days')
            LIMIT 1""",
            (politician_slug, amount, description, currency)).fetchone()
    if row:
        return {"record_id": row[0], "timestamp": row[1]}
    return None

def check_rate_limit(anon_id: str) -> tuple[bool, int]:
    """
    Sliding 1-hour window rate limiter per anon_id.
    Returns (allowed: bool, current_count: int).
    """
    if anon_id is None:
        return True, 0
    now_hour = time.strftime("%Y-%m-%d %H:00:00", time.gmtime())
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT count FROM submission_counts WHERE anon_id=? AND window_start=?",
            (anon_id, now_hour)).fetchone()
        current = row[0] if row else 0
        allowed = current < MAX_SUBMISSIONS_PER_HOUR
        if allowed:
            if row:
                conn.execute(
                    "UPDATE submission_counts SET count=count+1 "
                    "WHERE anon_id=? AND window_start=?",
                    (anon_id, now_hour))
            else:
                conn.execute(
                    "INSERT INTO submission_counts (anon_id,window_start,count) "
                    "VALUES (?,?,1)",
                    (anon_id, now_hour))
            conn.commit()
    return allowed, current

def validate_amount(amount_kes: float) -> tuple[bool, str]:
    """Hard bounds on KES amount — catches accidental or malicious nonsense."""
    if amount_kes < MIN_AMOUNT_KES:
        return False, f"Amount KES {amount_kes:,.2f} is below the minimum allowed (KES {MIN_AMOUNT_KES:,.0f})."
    if amount_kes > MAX_AMOUNT_KES:
        return False, f"Amount KES {amount_kes:,.0f} exceeds the maximum single-transaction limit (KES {MAX_AMOUNT_KES:,.0f}). Split into separate records if legitimate."
    return True, ""

def quarantine_expense(politician_slug, amount, currency,
                       description, submitted_by, reason):
    """Send a suspicious record to the quarantine queue instead of the live ledger."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO quarantine
              (politician_slug,original_amount,currency,description,submitted_by,reason)
            VALUES (?,?,?,?,?,?)""",
            (politician_slug, amount, currency, description, submitted_by, reason))
        conn.commit()
    log("QUARANTINE", submitted_by,
        f"slug={politician_slug} reason={reason} {amount}{currency}")

def get_quarantine() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM quarantine ORDER BY timestamp DESC", conn)

def approve_quarantine(q_id: int, moderator_id: str):
    """Pull a quarantined item into the live ledger."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT * FROM quarantine WHERE id=?", (q_id,)).fetchone()
        if not row:
            return False
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM quarantine WHERE id=?", (q_id,)).description]
        rec = dict(zip(cols, row))
    insert_expense(
        rec["politician_slug"], rec["original_amount"],
        rec["currency"], rec["description"],
        submitted_by=rec["submitted_by"],
    )
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE quarantine SET status='approved', reviewed_by=? WHERE id=?",
            (moderator_id, q_id))
        conn.commit()
    log("QUARANTINE_APPROVE", moderator_id, f"q_id={q_id}")
    return True

def reject_quarantine(q_id: int, moderator_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE quarantine SET status='rejected', reviewed_by=? WHERE id=?",
            (moderator_id, q_id))
        conn.commit()
    log("QUARANTINE_REJECT", moderator_id, f"q_id={q_id}")

def safe_insert_expense(politician_slug, target_name, amount_kes,
                        original_amount, currency, description,
                        submitted_by, force=False) -> dict:
    """
    Single entry-point for ALL expense insertions.
    Runs every fraud check before touching the ledger.

    Returns:
      {"status": "ok"}                    → inserted cleanly
      {"status": "quarantined", "reason"} → held for moderator review
      {"status": "rejected",    "reason"} → hard-blocked, not saved at all
      {"status": "duplicate",   "existing_record_id", "timestamp"}
    """
    # 1. Rate limit
    allowed, count = check_rate_limit(submitted_by)
    if not allowed:
        log("RATE_LIMIT_BLOCK", submitted_by,
            f"slug={politician_slug} count={count}")
        return {"status": "rejected",
                "reason": f"Submission rate limit reached ({MAX_SUBMISSIONS_PER_HOUR}/hour). "
                           "Please wait before submitting more records."}

    # 2. Amount bounds
    ok, err = validate_amount(amount_kes)
    if not ok:
        log("AMOUNT_BOUNDS_BLOCK", submitted_by, err)
        return {"status": "rejected", "reason": err}

    # 3. Duplicate check
    dup = check_duplicate_expense(politician_slug, original_amount,
                                  description, currency)
    if dup and not force:
        return {"status": "duplicate",
                "existing_record_id": dup["record_id"],
                "timestamp": dup["timestamp"]}

    # 4. Low-trust users → quarantine instead of live ledger
    trust = login_citizen(submitted_by) if submitted_by else None
    if trust is not None and trust < QUARANTINE_TRUST_GATE and not force:
        quarantine_expense(politician_slug, original_amount, currency,
                           description, submitted_by,
                           "Low trust score — pending moderator review")
        return {"status": "quarantined",
                "reason": f"Your Trust Score ({trust}) is below the threshold "
                           f"({QUARANTINE_TRUST_GATE}) for direct submission. "
                           "Your record has been queued for moderator review."}

    # 5. All checks passed → insert
    insert_expense(politician_slug, original_amount, currency,
                   description, submitted_by=submitted_by)
    return {"status": "ok"}

# ── Politician helpers ────────────────────────────────────────────────────────
def make_slug(name: str) -> str:
    """Convert 'John Kamau Njoroge' → 'john-kamau-njoroge'"""
    return re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")

def create_politician(full_name, role="", party="", county="",
                      constituency="", photo_url="", bio="", created_by=None):
    slug = make_slug(full_name)
    base_slug = slug
    # Guarantee uniqueness
    with sqlite3.connect(DB_PATH) as conn:
        i = 1
        while conn.execute("SELECT 1 FROM politicians WHERE slug=?", (slug,)).fetchone():
            slug = f"{base_slug}-{i}"; i += 1
        conn.execute("""
            INSERT INTO politicians
              (slug,full_name,role,party,county,constituency,photo_url,bio,created_by)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (slug, full_name, role, party, county, constituency,
             photo_url, bio, created_by))
        conn.commit()
    log("CREATE_PROFILE", created_by, full_name)
    return slug

def update_politician(slug, **kwargs):
    allowed = {"full_name","role","party","county","constituency","photo_url","bio"}
    fields  = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    with sqlite3.connect(DB_PATH) as conn:
        sets  = ", ".join(f"{k}=?" for k in fields)
        vals  = list(fields.values()) + [slug]
        conn.execute(f"UPDATE politicians SET {sets} WHERE slug=?", vals)
        conn.commit()

def get_all_politicians() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM politicians ORDER BY full_name ASC", conn)

def get_politician(slug: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT * FROM politicians WHERE slug=?", (slug,)).fetchone()
        if not row:
            return None
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM politicians WHERE slug=?", (slug,)).description]
    return dict(zip(cols, row))

def search_politicians(query: str) -> pd.DataFrame:
    q = f"%{query.lower()}%"
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("""
            SELECT * FROM politicians
            WHERE lower(full_name) LIKE ?
               OR lower(role)      LIKE ?
               OR lower(party)     LIKE ?
               OR lower(county)    LIKE ?
               OR lower(constituency) LIKE ?
            ORDER BY full_name ASC""",
            conn, params=(q,q,q,q,q))

# ── Expense helpers ───────────────────────────────────────────────────────────
def insert_expense(politician_slug, amount, currency, description,
                   record_id=None, version=1, submitted_by=None):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        if record_id is None:
            record_id = str(uuid.uuid4())
        else:
            c.execute(
                "UPDATE expenses SET is_active=0 WHERE record_id=?",
                (record_id,))
        c.execute("""
            INSERT INTO expenses
              (record_id,politician_slug,original_amount,currency,
               description,version,is_active,submitted_by)
            VALUES (?,?,?,?,?,?,1,?)""",
            (record_id, politician_slug, amount, currency,
             description, version, submitted_by))
        conn.commit()
    log("INSERT_EXPENSE", submitted_by,
        f"slug={politician_slug} {amount}{currency}")
    return record_id

def get_expenses(politician_slug=None) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        if politician_slug:
            return pd.read_sql_query(
                "SELECT * FROM expenses WHERE is_active=1 AND politician_slug=? "
                "ORDER BY timestamp DESC", conn, params=(politician_slug,))
        return pd.read_sql_query(
            "SELECT * FROM expenses WHERE is_active=1 ORDER BY timestamp DESC", conn)

def get_all_expenses_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM expenses ORDER BY timestamp DESC", conn)

# ── Review helpers ────────────────────────────────────────────────────────────
def insert_review(politician_slug, anon_id, findings, title="Finding", record_id=None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO reviews
              (politician_slug,record_id,anon_id,title,findings)
            VALUES (?,?,?,?,?)""",
            (politician_slug, record_id, anon_id, title, findings))
        conn.commit()
    log("PUBLISH_REVIEW", anon_id, f"slug={politician_slug}")

def get_reviews(politician_slug=None) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        if politician_slug:
            return pd.read_sql_query(
                "SELECT * FROM reviews WHERE politician_slug=? "
                "ORDER BY upvotes DESC, timestamp DESC",
                conn, params=(politician_slug,))
        return pd.read_sql_query(
            "SELECT * FROM reviews ORDER BY upvotes DESC, timestamp DESC", conn)

def cast_vote(review_id, vote_power):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE reviews SET upvotes=upvotes+? WHERE id=?",
            (vote_power, review_id))
        conn.commit()

# ── Citizen helpers ───────────────────────────────────────────────────────────
def register_citizen(anon_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO citizens (anon_id,trust_score) VALUES (?,0)",
            (anon_id,))
        conn.commit()

def login_citizen(anon_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT trust_score FROM citizens WHERE anon_id=?",
            (anon_id,)).fetchone()
    return row[0] if row else None

def award_trust(anon_id, pts=1):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE citizens SET trust_score=trust_score+? WHERE anon_id=?",
            (pts, anon_id))
        conn.commit()

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CURRENCY CONVERSION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_rates():
    try:
        r = requests.get("https://open.er-api.com/v6/latest/KES", timeout=5)
        d = r.json()
        if d.get("result") == "success":
            return d["rates"]
    except Exception:
        pass
    return {"KES":1.0,"USD":0.0074,"GBP":0.0058,"EUR":0.0068,
            "TZS":2.6,"UGX":28.0,"ETB":0.42}

def to_kes(amount, currency, rates):
    r = rates.get(str(currency).upper(), 1.0)
    return amount / r if r else amount

# ─────────────────────────────────────────────────────────────────────────────
# 5.  AI PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def benfords_score(amounts):
    s = pd.Series(amounts).dropna()
    if len(s) < 10:
        return None
    digits = []
    for a in s:
        x = str(abs(float(a))).replace(".","").lstrip("0")
        if x: digits.append(int(x[0]))
    obs = (pd.Series(digits).value_counts(normalize=True)
           .reindex(range(1,10), fill_value=0).values)
    exp = np.array([math.log10(1+1/d) for d in range(1,10)])
    return round(min(float(np.mean(np.abs(obs-exp)))*500, 100), 1), obs, exp

def run_pipeline(df, rates):
    if df.empty:
        return df
    df = df.copy()
    df["amount_kes"] = df.apply(
        lambda r: to_kes(r["original_amount"], r["currency"], rates), axis=1)
    df["flag_text"]   = df["description"].apply(
        lambda x: any(kw in str(x).lower() for kw in NLP_RED_FLAGS))
    df["flag_amount"] = False
    if len(df) >= 6:
        model = IsolationForest(contamination=0.1, random_state=42)
        df["flag_amount"] = model.fit_predict(df[["amount_kes"]]) == -1
    df["is_flagged"] = df["flag_text"] | df["flag_amount"]
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FILE PARSING  (universal uploader)
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "Politician": [
        "politician","candidate","name","full name","fullname","person",
        "mp","mca","senator","governor","minister","cs","cec","ward rep",
        "ward representative","councillor","councilor","aspirant","running mate",
        "petitioner","contestant","beneficiary","payee","recipient",
        "account name","account holder","submitted by","filed by","reported by",
        "politician name","candidate name","candidate_name","politician_name",
        "mgombea","mwanasiasa","jina","jina kamili",
    ],
    "Amount": [
        "amount","cost","sum","total","value","price","figure","quantity",
        "expenditure","expense","expenses","payment","payments","fee","fees",
        "charges","charge","debit","credit","disbursement","remittance",
        "outflow","spend","spending","spent","gross","net","paid",
        "invoice amount","billed amount","transaction amount","trans amount",
        "receipt amount","approved amount","actual amount","budget amount",
        "commitment amount","payment amount","cheque amount","voucher amount",
        "lpo amount","amount (kes)","amount_kes","amount kes","kes amount",
        "amount (usd)","amount (ksh)","ksh","kes","usd","gbp","eur","ugx","tzs",
        "kiasi","gharama","malipo",
    ],
    "Currency": [
        "currency","curr","ccy","fx","denomination","currency code","cur",
        "iso","iso code","transaction currency","payment currency",
        "currency type","currency_code","sarafu",
    ],
    "Description": [
        "description","desc","details","detail","item","items","line item",
        "particulars","narration","narrative","note","notes","purpose","reason",
        "explanation","remarks","remark","comment","comments","memo",
        "expense type","expense category","expenditure type","payment description",
        "payment narration","transaction description","transaction narration",
        "invoice description","goods/services","services rendered",
        "service description","nature of expense","nature of payment",
        "what","for","what for","used for","purpose of payment",
        "account description","vote head","programme","sub programme",
        "activity","project","project name","budget line","budget head",
        "cost centre","cost center","fund","category","type","kind","class",
        "maelezo","sababu","aina",
    ],
}

def fuzzy_match(df, unmatched):
    suggestions, lower_cols = {}, {c.lower().strip(): c for c in df.columns}
    for canonical in unmatched:
        best, score = None, 0
        for alias in COLUMN_ALIASES[canonical]:
            for cl, corig in lower_cols.items():
                if alias in cl or cl in alias:
                    if len(alias) > score:
                        score, best = len(alias), corig
        if best:
            suggestions[canonical] = best
    return suggestions

def normalise_cols(df):
    rename, lower_cols = {}, {c.lower().strip(): c for c in df.columns}
    for canon, aliases in COLUMN_ALIASES.items():
        if canon in df.columns: continue
        for alias in aliases:
            if alias.lower() in lower_cols:
                rename[lower_cols[alias.lower()]] = canon; break
    df = df.rename(columns=rename)
    still = [c for c in COLUMN_ALIASES if c not in df.columns]
    if still:
        fz = fuzzy_match(df, still)
        if fz: df = df.rename(columns={v:k for k,v in fz.items()})
    return df, [c for c in COLUMN_ALIASES if c not in df.columns]

def sniff_delim(raw):
    try:
        return csv.Sniffer().sniff(
            raw[:4096].decode("utf-8", errors="ignore"),
            delimiters=",\t|;").delimiter
    except csv.Error:
        return ","

def parse_file(uploaded):
    name = uploaded.name.lower()
    raw  = uploaded.read(); uploaded.seek(0)
    try:
        if name.endswith((".xlsx",".xls")):
            return pd.read_excel(io.BytesIO(raw)), ""
        if name.endswith(".ods"):
            return pd.read_excel(io.BytesIO(raw), engine="odf"), ""
        if name.endswith((".json",".jsonl")):
            txt = raw.decode("utf-8", errors="ignore")
            try:
                d = json.loads(txt)
                return pd.DataFrame(d if isinstance(d,list) else [d]), ""
            except json.JSONDecodeError:
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                return pd.DataFrame([json.loads(l) for l in lines]), ""
        if name.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(raw), sep="\t",
                               encoding="utf-8", on_bad_lines="skip"), ""
        # CSV / TXT / anything else
        delim = sniff_delim(raw)
        return pd.read_csv(io.BytesIO(raw), sep=delim,
                           encoding="utf-8", on_bad_lines="skip",
                           skipinitialspace=True), ""
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
init_db()
for k,v in {"anon_id":None,"trust_score":0,
            "last_action_time":0.0,"view":"directory",
            "active_slug":None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

rates = get_rates()

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SIDEBAR — Identity + navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔎 CivicTech Tracker")
    st.caption("Kenya Campaign Finance Watchdog")
    st.divider()

    # Identity
    st.markdown("### 🕵️ Citizen Identity")
    st.caption("No email required. Save your ID to keep your Trust Score.")

    if st.session_state["anon_id"] is None:
        t1, t2 = st.tabs(["New", "Returning"])
        with t1:
            if st.button("Generate Anonymous ID", use_container_width=True):
                nid = "Civic-" + str(uuid.uuid4())[:8].upper()
                register_citizen(nid)
                st.session_state.update({"anon_id": nid, "trust_score": 0})
                st.rerun()
        with t2:
            lid = st.text_input("Your Civic ID", placeholder="Civic-XXXXXXXX",
                                label_visibility="collapsed")
            if st.button("Access Profile", use_container_width=True):
                sc = login_citizen(lid.strip())
                if sc is not None:
                    st.session_state.update({"anon_id": lid.strip(),
                                             "trust_score": sc})
                    st.rerun()
                else:
                    st.error("ID not found.")
    else:
        st.success("✅ Identity active")
        st.code(st.session_state["anon_id"])
        st.metric("🏆 Trust Score", st.session_state["trust_score"])
        st.caption("Score ≥ 5 → votes count  |  ≥ 15 → file uploads unlocked")
        if st.button("Log out", use_container_width=True):
            st.session_state.update({"anon_id":None,"trust_score":0})
            st.rerun()

    st.divider()

    # Navigation
    st.markdown("### 🗺️ Navigation")
    if st.button("🏠 Politician Directory",   use_container_width=True):
        st.session_state.update({"view":"directory","active_slug":None})
        st.rerun()
    if st.button("➕ Add New Politician",      use_container_width=True):
        st.session_state.update({"view":"create","active_slug":None})
        st.rerun()
    if st.button("📊 National Overview",       use_container_width=True):
        st.session_state.update({"view":"overview","active_slug":None})
        st.rerun()
    if st.button("🔶 Moderator Queue",         use_container_width=True):
        st.session_state.update({"view":"modqueue","active_slug":None})
        st.rerun()
    if st.button("🔍 Audit Trail",             use_container_width=True):
        st.session_state.update({"view":"audit","active_slug":None})
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ AI flags are statistical anomalies for public review — "
        "not conclusions of wrongdoing."
    )

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — reusable UI components
# ─────────────────────────────────────────────────────────────────────────────
ROLES = [
    "Member of Parliament", "Senator", "Governor", "MCA",
    "Cabinet Secretary", "CEC Member", "Ward Representative",
    "President", "Deputy President", "Permanent Secretary", "Other",
]
COUNTIES = [
    "Nairobi","Mombasa","Kisumu","Nakuru","Uasin Gishu","Kiambu",
    "Machakos","Meru","Nyeri","Kilifi","Kakamega","Bungoma","Siaya",
    "Kisii","Migori","Homa Bay","Bomet","Kericho","Nandi","Trans Nzoia",
    "Elgeyo Marakwet","West Pokot","Turkana","Marsabit","Isiolo","Garissa",
    "Wajir","Mandera","Tana River","Lamu","Taita Taveta","Kwale","Kajiado",
    "Narok","Laikipia","Nyandarua","Murang'a","Kirinyaga","Embu","Tharaka Nithi",
    "Meru","Isiolo","Samburu","Baringo","Elgeyo Marakwet","Vihiga","Busia",
    "Other / National",
]

def render_benford_section(df_expenses):
    """Render Benford's Law analysis for a given expenses dataframe."""
    res = benfords_score(df_expenses["amount_kes"])
    if res is None:
        st.info("Need ≥ 10 expense records for Benford's Law analysis.")
        return
    score, obs, exp = res
    if score > 65:
        st.error(f"🔴 Benford Suspicion Score: **{score}/100** — High divergence.")
    elif score > 35:
        st.warning(f"🟡 Benford Suspicion Score: **{score}/100** — Moderate deviation.")
    else:
        st.success(f"🟢 Benford Suspicion Score: **{score}/100** — Looks natural.")

    chart_df = pd.DataFrame({
        "Expected (Benford %)": np.round(exp*100, 1),
        "Observed (%)":         np.round(obs*100, 1),
    }, index=range(1,10))
    chart_df.index.name = "Leading Digit"
    st.bar_chart(chart_df)

def render_upload_section(politician_slug, target_name=""):
    """
    Universal file uploader scoped to a specific politician.
    Includes cross-contamination detection, duplicate checking,
    rate limiting, amount bounds validation, and quarantine routing.
    """
    st.markdown(
        "Accepts: **Excel** (.xlsx .xls), **CSV**, **TSV**, **TXT**, "
        "**JSON** (.json .jsonl), **ODS**\n\n"
        "Columns can use any recognised name. "
        "The Politician column is **ignored** — all records are automatically "
        "locked to **this profile only**."
    )

    # Warn clearly that the Politician column is ignored
    st.info(
        f"🔒 **All records in this file will be attributed to: {target_name}**\n\n"
        "If your file contains records for multiple politicians, upload each "
        "politician's records from their own profile page."
    )

    tmpl = pd.DataFrame({
        "Amount":      [150000, 5000],
        "Currency":    ["KES",  "USD"],
        "Description": ["Rally PA System", "Consulting Fee"],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        tmpl.to_excel(w, index=False)
    st.download_button("⬇ Download template", buf.getvalue(),
                       "template.xlsx", "application/vnd.ms-excel")

    up = st.file_uploader(
        "Choose file",
        type=["xlsx","xls","csv","tsv","txt","json","jsonl","ods"],
        key=f"up_{politician_slug}",
    )
    if not up:
        return

    df_raw, err = parse_file(up)
    if err:
        st.error(f"Could not read file: {err}"); return

    st.caption(f"Columns found: `{'`, `'.join(df_raw.columns.tolist())}`")
    df_norm, missing = normalise_cols(df_raw)

    matched = [c for c in ["Amount","Currency","Description"] if c in df_norm.columns]
    if matched:
        st.success(f"✅ Auto-matched: {', '.join(matched)}")

    # ── Cross-contamination scan ──────────────────────────────────────────────
    # Check if the file contains names belonging to OTHER politicians
    cross_warnings = detect_wrong_politician(df_norm if "Politician" in df_norm.columns
                                             else df_raw, politician_slug, target_name)
    if cross_warnings:
        st.warning(
            f"⚠️ **Cross-contamination detected in {len(cross_warnings)} row(s)**\n\n"
            "Your file contains a Politician column with names that do not match "
            f"**{target_name}**. These rows will still be attributed to this profile "
            "as you are uploading from their page — but please verify your file is correct."
        )
        with st.expander("View flagged rows"):
            for w in cross_warnings[:10]:
                st.caption(
                    f"Row {w['row']}: found '{w['found']}' "
                    f"(similarity to {target_name}: {w['similarity']:.0%}) "
                    f"— Risk: **{w['risk']}**"
                )

    # Drop the Politician column entirely — we use politician_slug instead
    if "Politician" in df_norm.columns:
        df_norm = df_norm.drop(columns=["Politician"])

    # Currency defaults silently to KES
    if "Currency" in missing:
        df_norm["Currency"] = "KES"
        missing = [m for m in missing if m != "Currency"]
        st.info("No currency column — defaulting to **KES**.")
    if "Description" in missing:
        df_norm["Description"] = "—"
        missing = [m for m in missing if m != "Description"]

    # Manual mapping for anything still missing
    if missing:
        st.warning(f"Could not auto-map: **{', '.join(missing)}**")
        all_cols = ["— skip —"] + list(df_raw.columns)
        mapping  = {}
        for field in missing:
            hints = {"Amount": "cost / spend / total / payment / kiasi",
                     "Description": "narration / purpose / details / maelezo"}
            mapping[field] = st.selectbox(
                f"Which column is **{field}**?  *({hints.get(field,'')})*",
                all_cols, key=f"map_{politician_slug}_{field}")
        if st.button("Apply mapping", key=f"apply_{politician_slug}"):
            for field, col in mapping.items():
                if col != "— skip —":
                    df_norm[field] = df_raw[col]
            st.rerun()

    if "Amount" not in df_norm.columns:
        st.error("Amount column is required."); return

    # Clean amounts
    df_norm["Amount"] = (df_norm["Amount"].astype(str)
                         .str.replace(r"[^\d.]","",regex=True))
    df_norm["Amount"] = pd.to_numeric(df_norm["Amount"], errors="coerce")
    df_norm = df_norm.dropna(subset=["Amount"])

    preview_cols = [c for c in ["Amount","Currency","Description"] if c in df_norm.columns]
    st.markdown(f"**{len(df_norm)} valid rows ready** — preview:")
    st.dataframe(df_norm[preview_cols].head(), use_container_width=True, hide_index=True)

    if st.button("🔒 Lock all rows into ledger", key=f"lock_{politician_slug}",
                 use_container_width=True):
        results = {"ok":0, "quarantined":0, "duplicate":0, "rejected":0}
        dup_details, reject_details = [], []

        for _, row in df_norm.iterrows():
            amt_orig  = float(row["Amount"])
            curr      = str(row.get("Currency","KES")).strip().upper()
            desc      = str(row.get("Description","—")).strip()
            amt_kes   = to_kes(amt_orig, curr, rates)

            result = safe_insert_expense(
                politician_slug, target_name,
                amt_kes, amt_orig, curr, desc,
                submitted_by=st.session_state["anon_id"],
            )
            status = result["status"]
            results[status] = results.get(status, 0) + 1

            if status == "duplicate":
                dup_details.append(
                    f"KES {amt_orig:,.0f} — '{desc[:40]}' "
                    f"(existing: {result['existing_record_id'][:8]}…)")
            elif status == "rejected":
                reject_details.append(result["reason"])

        # Summary report
        if results["ok"]:
            st.success(f"✅ {results['ok']} records locked to **{target_name}**.")
        if results["quarantined"]:
            st.warning(
                f"🔶 {results['quarantined']} records sent to moderator review queue "
                "(Trust Score below threshold)."
            )
        if results["duplicate"]:
            st.warning(f"🔁 {results['duplicate']} duplicate(s) skipped:")
            for d in dup_details[:5]:
                st.caption(f"  → {d}")
        if results["rejected"]:
            st.error(f"🚫 {results['rejected']} record(s) blocked:")
            for r in set(reject_details[:3]):
                st.caption(f"  → {r}")
        st.rerun()

def render_reviews_section(politician_slug, df_expenses):
    """Community reviews + shadow voting for a politician profile."""
    st.markdown("---")
    st.subheader("🕵️ Community Investigations")
    st.caption(
        "Publish findings linked to this politician's profile. "
        "Reviews are ranked by community votes."
    )

    col_submit, col_feed = st.columns([1,1])

    with col_submit:
        st.markdown("#### Publish a Finding")
        if st.session_state["anon_id"] is None:
            st.warning("Generate a Citizen ID in the sidebar to publish reviews.")
        else:
            st.info(f"Your Trust Score: **{st.session_state['trust_score']}**  |  "
                    f"Min length: **{MIN_REVIEW_LEN} chars**")
            with st.form(f"review_{politician_slug}", clear_on_submit=True):
                r_title = st.text_input("Finding title",
                    placeholder="e.g. V8 rental traced to shell company")

                # Optionally link to a specific expense record
                rec_options = ["— General finding (no specific record) —"]
                if not df_expenses.empty:
                    for _, row in df_expenses.iterrows():
                        rec_options.append(
                            f"{row['record_id'][:8]}… | "
                            f"{row['currency']} {row['original_amount']:,.0f} | "
                            f"{str(row['description'])[:40]}")
                rec_sel = st.selectbox("Link to a specific expense (optional)",
                                       rec_options)
                linked_rid = None
                if rec_sel != rec_options[0] and not df_expenses.empty:
                    idx = rec_options.index(rec_sel) - 1
                    linked_rid = df_expenses.iloc[idx]["record_id"]

                findings = st.text_area("Your findings & evidence", height=140,
                    placeholder="Describe what you found, provide links to "
                                "documents, explain the irregularity…")

                if st.form_submit_button("Publish Finding", use_container_width=True):
                    if not r_title.strip():
                        st.error("Please add a title.")
                    elif len(findings) < MIN_REVIEW_LEN:
                        st.error(f"Too short ({len(findings)} chars). "
                                 f"Minimum is {MIN_REVIEW_LEN}.")
                    else:
                        insert_review(politician_slug,
                                      st.session_state["anon_id"],
                                      findings, r_title.strip(), linked_rid)
                        award_trust(st.session_state["anon_id"], TRUST_BONUS)
                        st.session_state["trust_score"] += TRUST_BONUS
                        st.success(f"✅ Published! +{TRUST_BONUS} Trust Point earned.")
                        st.rerun()

    with col_feed:
        st.markdown("#### Published Findings")
        df_rev = get_reviews(politician_slug)
        if df_rev.empty:
            st.info("No findings published yet for this politician.")
        else:
            for _, row in df_rev.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**")
                    st.caption(
                        f"By `{row['anon_id']}` · "
                        f"🏆 {row['upvotes']} votes · "
                        f"{str(row['timestamp'])[:10]}"
                    )
                    if row["record_id"]:
                        st.caption(f"🔗 Linked to expense `{str(row['record_id'])[:8]}…`")
                    st.write(row["findings"])

                    if st.button("⬆️ Upvote", key=f"vote_{row['id']}"):
                        now = time.time()
                        if now - st.session_state["last_action_time"] < RATE_LIMIT_SECS:
                            st.error(f"⏳ Please wait {RATE_LIMIT_SECS:.0f}s between votes.")
                        else:
                            st.session_state["last_action_time"] = now
                            # SHADOW VOTE — low trust = power 0, UI still says success
                            vp = 1 if st.session_state["trust_score"] >= TRUST_VOTE_GATE else 0
                            cast_vote(int(row["id"]), vp)
                            log("VOTE", st.session_state["anon_id"],
                                f"review_id={row['id']} power={vp}")
                            st.success("Vote recorded! Thank you.")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  VIEW ROUTER
# ─────────────────────────────────────────────────────────────────────────────
view = st.session_state["view"]

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: POLITICIAN DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════════
if view == "directory":
    st.title("🏛️ Politician Directory")
    st.caption("Search and browse all tracked politicians. Click a profile to view their records.")

    # Search bar
    query = st.text_input("🔍 Search by name, role, party, county or constituency",
                          placeholder="e.g. Nairobi, Governor, ODM…",
                          label_visibility="collapsed")

    df_pols = search_politicians(query) if query else get_all_politicians()

    if df_pols.empty:
        st.info("No politicians found. Use **➕ Add New Politician** in the sidebar.")
    else:
        # Summary stats bar
        all_expenses = get_expenses()
        all_analyzed = run_pipeline(all_expenses, rates) if not all_expenses.empty \
                       else pd.DataFrame()

        st.caption(f"Showing **{len(df_pols)}** profiles")
        st.divider()

        # Render as a grid of cards — 3 per row
        cols_per_row = 3
        rows = [df_pols.iloc[i:i+cols_per_row]
                for i in range(0, len(df_pols), cols_per_row)]

        for row_df in rows:
            cols = st.columns(cols_per_row)
            for col, (_, pol) in zip(cols, row_df.iterrows()):
                with col:
                    with st.container(border=True):
                        # Avatar initial circle
                        initials = "".join(w[0].upper()
                                          for w in pol["full_name"].split()[:2])

                        # Get quick stats for this politician
                        pol_exp = all_analyzed[
                            all_analyzed["politician_slug"] == pol["slug"]
                        ] if not all_analyzed.empty else pd.DataFrame()

                        n_exp    = len(pol_exp)
                        total_k  = pol_exp["amount_kes"].sum() if n_exp else 0
                        n_flag   = int(pol_exp["is_flagged"].sum()) if n_exp else 0
                        n_rev    = len(get_reviews(pol["slug"]))

                        st.markdown(
                            f"<div style='width:48px;height:48px;border-radius:50%;"
                            f"background:#1f4e79;color:white;display:flex;"
                            f"align-items:center;justify-content:center;"
                            f"font-size:18px;font-weight:600;margin-bottom:8px'>"
                            f"{initials}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**{pol['full_name']}**")
                        if pol["role"]:
                            st.caption(pol["role"])
                        if pol["party"]:
                            st.caption(f"🏛 {pol['party']}")
                        if pol["county"]:
                            st.caption(f"📍 {pol['county']}")

                        st.markdown("---")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Records",  n_exp)
                        m2.metric("Flagged",  n_flag)
                        m3.metric("Reviews",  n_rev)

                        if n_exp:
                            st.caption(f"KES {total_k:,.0f} total declared")

                        if st.button("View Profile →",
                                     key=f"view_{pol['slug']}",
                                     use_container_width=True):
                            st.session_state.update({
                                "view": "profile",
                                "active_slug": pol["slug"],
                            })
                            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: CREATE POLITICIAN PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "create":
    st.title("➕ Add New Politician Profile")
    st.caption(
        "Create a searchable profile. Financial records and community "
        "findings will be permanently linked to this profile."
    )

    with st.form("create_politician_form"):
        st.markdown("#### Basic Information")
        c1, c2 = st.columns(2)
        full_name    = c1.text_input("Full Name *", placeholder="e.g. Jane Akinyi Odhiambo")
        role         = c2.selectbox("Role / Position", [""] + ROLES)

        c3, c4 = st.columns(2)
        party        = c3.text_input("Party / Coalition", placeholder="e.g. ODM, UDA, Jubilee…")
        county       = c4.selectbox("County", [""] + COUNTIES)

        constituency = st.text_input("Constituency / Ward",
                                     placeholder="e.g. Westlands, Kibra, Langata…")

        st.markdown("#### Optional Details")
        bio          = st.text_area("Short Bio / Context",
                                    placeholder="Role in government, notable positions, "
                                                "years in office…", height=80)
        photo_url    = st.text_input("Photo URL (optional)",
                                     placeholder="https://… (publicly accessible image)")

        submitted = st.form_submit_button("Create Profile", use_container_width=True)

    if submitted:
        if not full_name.strip():
            st.error("Full name is required.")
        else:
            slug = create_politician(
                full_name.strip(), role, party, county,
                constituency, photo_url, bio,
                created_by=st.session_state["anon_id"],
            )
            st.success(f"✅ Profile created for **{full_name}**!")
            st.session_state.update({"view":"profile","active_slug":slug})
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: POLITICIAN PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "profile":
    slug = st.session_state["active_slug"]
    pol  = get_politician(slug)

    if pol is None:
        st.error("Profile not found.")
        st.stop()

    # ── Back button ──────────────────────────────────────────────────────────
    if st.button("← Back to Directory"):
        st.session_state.update({"view":"directory","active_slug":None})
        st.rerun()

    # ── Profile header ───────────────────────────────────────────────────────
    initials = "".join(w[0].upper() for w in pol["full_name"].split()[:2])
    hcol1, hcol2 = st.columns([1, 4])

    with hcol1:
        if pol.get("photo_url"):
            try:
                st.image(pol["photo_url"], width=120)
            except Exception:
                st.markdown(
                    f"<div style='width:100px;height:100px;border-radius:50%;"
                    f"background:#1f4e79;color:white;display:flex;"
                    f"align-items:center;justify-content:center;"
                    f"font-size:32px;font-weight:600'>{initials}</div>",
                    unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='width:100px;height:100px;border-radius:50%;"
                f"background:#1f4e79;color:white;display:flex;"
                f"align-items:center;justify-content:center;"
                f"font-size:32px;font-weight:600'>{initials}</div>",
                unsafe_allow_html=True)

    with hcol2:
        st.title(pol["full_name"])
        meta = []
        if pol.get("role"):         meta.append(f"**{pol['role']}**")
        if pol.get("party"):        meta.append(f"🏛 {pol['party']}")
        if pol.get("county"):       meta.append(f"📍 {pol['county']}")
        if pol.get("constituency"): meta.append(f"🗳 {pol['constituency']}")
        st.markdown("  ·  ".join(meta) if meta else "")
        if pol.get("bio"):
            st.caption(pol["bio"])

    st.divider()

    # ── Quick stats ──────────────────────────────────────────────────────────
    df_exp = get_expenses(slug)
    df_analyzed = run_pipeline(df_exp, rates) if not df_exp.empty \
                  else pd.DataFrame()

    n_exp   = len(df_analyzed)
    total_k = df_analyzed["amount_kes"].sum()  if n_exp else 0
    n_flag  = int(df_analyzed["is_flagged"].sum()) if n_exp else 0
    n_rev   = len(get_reviews(slug))

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Total Declared (KES)", f"KES {total_k:,.0f}")
    s2.metric("Expense Records",      n_exp)
    s3.metric("AI-Flagged Items",     n_flag,
              delta=f"{n_flag/n_exp*100:.0f}%" if n_exp else None,
              delta_color="inverse")
    s4.metric("Community Reviews",   n_rev)

    # ── Profile tabs ─────────────────────────────────────────────────────────
    (tab_expenses, tab_upload, tab_benford,
     tab_reviews, tab_modq, tab_edit) = st.tabs([
        "💰 Expenses", "📤 Upload Records",
        "📐 Benford's Law", "🕵️ Community Findings",
        "🔶 Moderator Queue", "✏️ Edit Profile",
    ])

    # ── TAB: EXPENSES ────────────────────────────────────────────────────────
    with tab_expenses:
        st.subheader(f"Financial Records — {pol['full_name']}")
        st.caption("All records are permanently locked. Corrections create a new version, not a deletion.")

        if df_analyzed.empty:
            st.info("No financial records yet. Use the **Upload Records** tab to add them.")
        else:
            # Filter controls
            fc1, fc2 = st.columns(2)
            ff = fc1.selectbox("Show", ["All","Flagged only","Clean only"],
                               key="exp_filter")
            sort_by = fc2.selectbox("Sort by",
                                    ["Newest first","Largest amount","Smallest amount"],
                                    key="exp_sort")

            ds = df_analyzed.copy()
            if ff == "Flagged only": ds = ds[ds["is_flagged"]]
            elif ff == "Clean only": ds = ds[~ds["is_flagged"]]

            if sort_by == "Largest amount":  ds = ds.sort_values("amount_kes", ascending=False)
            elif sort_by == "Smallest amount": ds = ds.sort_values("amount_kes")

            # Render as cards for readability
            for _, row in ds.iterrows():
                flag_icon = "⚠️" if row["is_flagged"] else "✅"
                with st.container(border=True):
                    ec1, ec2 = st.columns([3,1])
                    with ec1:
                        st.markdown(
                            f"{flag_icon} **{row['currency']} "
                            f"{row['original_amount']:,.2f}**"
                            f"  *(KES {row['amount_kes']:,.0f})*"
                        )
                        st.write(row["description"])
                        flag_labels = []
                        if row["flag_amount"]: flag_labels.append("🤖 Amount anomaly")
                        if row["flag_text"]:   flag_labels.append("📝 Description flag")
                        if flag_labels:
                            st.caption("  ·  ".join(flag_labels))
                    with ec2:
                        st.caption(f"v{row['version']}")
                        st.caption(str(row["timestamp"])[:10])
                        st.caption(f"`{str(row['record_id'])[:8]}…`")

            # Manual single-record add
            st.divider()
            st.markdown("##### Add a single record manually")
            with st.form(f"manual_{slug}", clear_on_submit=True):
                m1, m2, m3 = st.columns(3)
                m_amt  = m1.number_input("Amount", min_value=0.01,
                                          format="%.2f", value=1000.0)
                m_curr = m2.selectbox("Currency",
                                       ["KES","USD","GBP","EUR","TZS","UGX"])
                m_desc = m3.text_input("Description")
                if st.form_submit_button("🔒 Lock record", use_container_width=True):
                    if m_desc.strip():
                        result = safe_insert_expense(
                            slug, pol["full_name"],
                            to_kes(m_amt, m_curr, rates),
                            m_amt, m_curr, m_desc.strip(),
                            submitted_by=st.session_state["anon_id"],
                        )
                        if result["status"] == "ok":
                            st.success("Record locked.")
                        elif result["status"] == "duplicate":
                            st.warning(
                                f"Duplicate detected — this exact record already "
                                f"exists (ID: {result['existing_record_id'][:8]}…, "
                                f"added {result['timestamp'][:10]}). "
                                "Tick the checkbox below to force-submit anyway."
                            )
                        elif result["status"] == "quarantined":
                            st.warning(result["reason"])
                        elif result["status"] == "rejected":
                            st.error(result["reason"])
                        st.rerun()
                    else:
                        st.error("Description is required.")

            # Export
            buf = io.BytesIO()
            export_cols = ["record_id","original_amount","currency",
                           "amount_kes","description","version",
                           "flag_text","flag_amount","timestamp"]
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                ds[export_cols].to_excel(w, index=False, sheet_name="Expenses")
            st.download_button(
                "📥 Export records (.xlsx)", buf.getvalue(),
                f"{slug}_expenses.xlsx", "application/vnd.ms-excel",
            )

            # Correction form
            st.divider()
            st.markdown("##### Submit a Correction (creates new version, never deletes)")
            edit_rid = st.selectbox(
                "Select record to correct",
                df_analyzed["record_id"].tolist(),
                format_func=lambda x: (
                    f"{x[:8]}… — {df_analyzed[df_analyzed['record_id']==x].iloc[0]['description'][:40]}"
                ),
            )
            with st.form(f"correct_{slug}"):
                ec1, ec2, ec3 = st.columns(3)
                new_amt  = ec1.number_input("Corrected Amount",
                                            min_value=0.01, format="%.2f")
                new_curr = ec2.selectbox("Currency",
                                         ["KES","USD","GBP","EUR","TZS","UGX"])
                new_desc = ec3.text_input("Corrected Description")
                if st.form_submit_button("Submit Correction"):
                    old = df_analyzed[df_analyzed["record_id"]==edit_rid].iloc[0]
                    insert_expense(
                        slug, new_amt, new_curr, new_desc,
                        record_id=edit_rid,
                        version=int(old["version"])+1,
                        submitted_by=st.session_state["anon_id"],
                    )
                    st.success(f"Version {int(old['version'])+1} created. Original archived.")
                    st.rerun()

    # ── TAB: UPLOAD ──────────────────────────────────────────────────────────
    with tab_upload:
        st.subheader(f"Upload Records for {pol['full_name']}")
        render_upload_section(slug, target_name=pol["full_name"])

    # ── TAB: BENFORD'S LAW ───────────────────────────────────────────────────
    with tab_benford:
        st.subheader(f"Benford's Law — {pol['full_name']}")
        st.markdown(
            "Naturally occurring financial figures follow a predictable logarithmic "
            "digit distribution. Fabricated numbers break this curve. A high score "
            "flags the profile for human review — it is not proof of wrongdoing."
        )
        if df_analyzed.empty:
            st.info("No records yet.")
        else:
            render_benford_section(df_analyzed)

    # ── TAB: COMMUNITY FINDINGS ──────────────────────────────────────────────
    with tab_reviews:
        render_reviews_section(slug, df_analyzed if not df_analyzed.empty
                               else pd.DataFrame())

    # ── TAB: MODERATOR QUEUE ────────────────────────────────────────────────
    with tab_modq:
        st.subheader(f"Moderator Review Queue — {pol['full_name']}")
        st.caption(
            "Records submitted by low-trust citizens are held here before "
            "going live. Review each one and approve or reject."
        )
        if not st.session_state.get("authenticated"):
            st.warning("Moderator access required.")
        else:
            df_q = get_quarantine()
            df_q_pol = df_q[
                (df_q["politician_slug"] == slug) &
                (df_q["status"] == "pending")
            ]
            if df_q_pol.empty:
                st.success("✅ No pending items in the queue for this politician.")
            else:
                st.warning(f"**{len(df_q_pol)} record(s) pending review.**")
                for _, qrow in df_q_pol.iterrows():
                    with st.container(border=True):
                        qc1, qc2 = st.columns([3,1])
                        with qc1:
                            st.markdown(
                                f"**{qrow['currency']} {qrow['original_amount']:,.2f}**"
                            )
                            st.write(qrow["description"])
                            st.caption(
                                f"Submitted by: `{qrow['submitted_by']}` · "
                                f"{str(qrow['timestamp'])[:10]} · "
                                f"Reason held: *{qrow['reason']}*"
                            )
                        with qc2:
                            if st.button("✅ Approve",
                                         key=f"qapprove_{qrow['id']}",
                                         use_container_width=True):
                                approve_quarantine(
                                    int(qrow["id"]),
                                    st.session_state["anon_id"] or "moderator",
                                )
                                st.success("Approved and added to live ledger.")
                                st.rerun()
                            if st.button("❌ Reject",
                                         key=f"qreject_{qrow['id']}",
                                         use_container_width=True):
                                reject_quarantine(
                                    int(qrow["id"]),
                                    st.session_state["anon_id"] or "moderator",
                                )
                                st.info("Record rejected.")
                                st.rerun()

    # ── TAB: EDIT PROFILE ────────────────────────────────────────────────────
    with tab_edit:
        st.subheader("Edit Profile Details")
        st.caption("Financial records cannot be edited — use the Correction system. "
                   "Profile details (name, role, party, etc.) can be updated here.")

        with st.form(f"edit_profile_{slug}"):
            e1, e2 = st.columns(2)
            new_name  = e1.text_input("Full Name", value=pol["full_name"] or "")
            new_role  = e2.selectbox("Role", [""] + ROLES,
                                     index=([""] + ROLES).index(pol["role"])
                                     if pol.get("role") in ROLES else 0)
            e3, e4 = st.columns(2)
            new_party = e3.text_input("Party / Coalition", value=pol["party"] or "")
            new_county= e4.selectbox("County", [""] + COUNTIES,
                                     index=([""] + COUNTIES).index(pol["county"])
                                     if pol.get("county") in COUNTIES else 0)
            new_const = st.text_input("Constituency / Ward",
                                      value=pol["constituency"] or "")
            new_bio   = st.text_area("Bio", value=pol["bio"] or "", height=80)
            new_photo = st.text_input("Photo URL", value=pol["photo_url"] or "")

            if st.form_submit_button("Save Changes", use_container_width=True):
                update_politician(slug,
                                  full_name=new_name, role=new_role,
                                  party=new_party, county=new_county,
                                  constituency=new_const,
                                  bio=new_bio, photo_url=new_photo)
                log("EDIT_PROFILE", st.session_state["anon_id"], slug)
                st.success("Profile updated.")
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: NATIONAL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "overview":
    st.title("📊 National Spending Overview")
    st.caption(
        "Aggregated data across all tracked politicians. "
        "**AI flags are anomalies for review — not conclusions of wrongdoing.**"
    )

    all_exp = get_expenses()
    if all_exp.empty:
        st.info("No financial records yet.")
        st.stop()

    df_all = run_pipeline(all_exp, rates)
    df_pols = get_all_politicians()

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Declared (KES)", f"KES {df_all['amount_kes'].sum():,.0f}")
    k2.metric("Total Records",        f"{len(df_all):,}")
    k3.metric("Politicians Tracked",  f"{df_pols.shape[0]}")
    k4.metric("AI-Flagged Items",     int(df_all["is_flagged"].sum()),
              delta=f"{df_all['is_flagged'].mean()*100:.1f}% of all",
              delta_color="inverse")

    st.divider()
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("##### Total Declared Spending per Politician (KES)")
        spend = (df_all.merge(df_pols[["slug","full_name"]],
                              left_on="politician_slug", right_on="slug",
                              how="left")
                 .groupby("full_name")["amount_kes"].sum()
                 .sort_values(ascending=False))
        st.bar_chart(spend)

    with c2:
        st.markdown("##### AI-Flagged Anomalies per Politician")
        flags = (df_all.merge(df_pols[["slug","full_name"]],
                              left_on="politician_slug", right_on="slug",
                              how="left")
                 .groupby("full_name")["is_flagged"].sum()
                 .sort_values(ascending=False))
        st.bar_chart(flags)

    st.divider()
    st.markdown("##### Benford's Law — All Politicians")
    rows = []
    for _, pol in df_pols.iterrows():
        sub = df_all[df_all["politician_slug"]==pol["slug"]]["amount_kes"]
        res = benfords_score(sub)
        rows.append({
            "Politician":            pol["full_name"],
            "Role":                  pol.get("role",""),
            "County":                pol.get("county",""),
            "Records":               len(sub),
            "Benford Suspicion Score": res[0] if res else "Need ≥10",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Full export
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df_all.to_excel(w, index=False, sheet_name="All Expenses")
        df_pols.to_excel(w, index=False, sheet_name="Politicians")
    st.download_button(
        "📥 Download Full Dataset (.xlsx)",
        buf.getvalue(), "civictech_full_export.xlsx",
        "application/vnd.ms-excel",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: GLOBAL MODERATOR QUEUE
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "modqueue":
    st.title("🔶 Global Moderator Queue")
    st.caption(
        "All pending submissions from low-trust citizens across every politician. "
        "Approve to push to the live ledger. Reject to discard permanently."
    )

    if not st.session_state.get("authenticated"):
        st.error("Moderator access required.")
        st.stop()

    df_q_all = get_quarantine()
    df_q_pending = df_q_all[df_q_all["status"] == "pending"]

    # Summary metrics
    qm1, qm2, qm3 = st.columns(3)
    qm1.metric("Pending Review",  len(df_q_pending))
    qm2.metric("Approved",  len(df_q_all[df_q_all["status"]=="approved"]))
    qm3.metric("Rejected",  len(df_q_all[df_q_all["status"]=="rejected"]))

    st.divider()

    if df_q_pending.empty:
        st.success("✅ No pending items across any politician.")
    else:
        # Group by politician for easier scanning
        df_pols_lookup = get_all_politicians().set_index("slug")["full_name"].to_dict()

        for pol_slug, group in df_q_pending.groupby("politician_slug"):
            pol_name = df_pols_lookup.get(pol_slug, pol_slug)
            st.subheader(f"📌 {pol_name}  ({len(group)} pending)")

            for _, qrow in group.iterrows():
                with st.container(border=True):
                    qc1, qc2 = st.columns([4, 1])
                    with qc1:
                        st.markdown(
                            f"**{qrow['currency']} {qrow['original_amount']:,.2f}**"
                        )
                        st.write(qrow["description"])
                        st.caption(
                            f"Submitter: `{qrow['submitted_by']}` · "
                            f"Date: {str(qrow['timestamp'])[:10]} · "
                            f"Held because: *{qrow['reason']}*"
                        )
                    with qc2:
                        if st.button("✅ Approve", key=f"gqa_{qrow['id']}",
                                     use_container_width=True):
                            approve_quarantine(
                                int(qrow["id"]),
                                st.session_state["anon_id"] or "moderator",
                            )
                            st.success("Approved.")
                            st.rerun()
                        if st.button("❌ Reject", key=f"gqr_{qrow['id']}",
                                     use_container_width=True):
                            reject_quarantine(
                                int(qrow["id"]),
                                st.session_state["anon_id"] or "moderator",
                            )
                            st.info("Rejected.")
                            st.rerun()

    # Show recent decisions
    st.divider()
    st.subheader("Recent Decisions")
    df_decided = df_q_all[df_q_all["status"] != "pending"].sort_values(
        "timestamp", ascending=False).head(50)
    if df_decided.empty:
        st.info("No decisions recorded yet.")
    else:
        display = df_decided[["politician_slug","original_amount","currency",
                               "description","status","reviewed_by","timestamp"]].copy()
        display["politician"] = display["politician_slug"].map(df_pols_lookup)
        st.dataframe(
            display[["politician","original_amount","currency",
                      "description","status","reviewed_by","timestamp"]],
            hide_index=True, use_container_width=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════════
elif view == "audit":
    st.title("🔍 Immutable Audit Trail")
    st.caption("Every record, correction, and action is stored permanently. Nothing is deleted.")

    audit_t1, audit_t2, audit_t3, audit_t4 = st.tabs([
        "💾 Expense History", "📋 Action Log",
        "🔶 Quarantine Log", "🚦 Rate Limit Events",
    ])

    with audit_t1:
        st.markdown("#### All Expense Versions — Nothing is ever deleted")
        df_hist = get_all_expenses_history()
        df_hist["Status"] = df_hist["is_active"].apply(
            lambda x: "✅ Active" if x else "🗃 Archived")
        # Filter controls
        ath_pol = st.multiselect(
            "Filter by politician slug",
            options=sorted(df_hist["politician_slug"].unique()),
            key="ath_pol")
        if ath_pol:
            df_hist = df_hist[df_hist["politician_slug"].isin(ath_pol)]
        st.dataframe(
            df_hist[["record_id","politician_slug","original_amount",
                     "currency","description","version","Status","timestamp"]],
            hide_index=True, use_container_width=True)
        buf_h = io.BytesIO()
        with pd.ExcelWriter(buf_h, engine="openpyxl") as w:
            df_hist.to_excel(w, index=False, sheet_name="Expense History")
        st.download_button("📥 Export expense history", buf_h.getvalue(),
                           "expense_history.xlsx", "application/vnd.ms-excel")

    with audit_t2:
        st.markdown("#### System Action Log")
        with sqlite3.connect(DB_PATH) as conn:
            df_log = pd.read_sql_query(
                "SELECT action,actor_id,detail,timestamp "
                "FROM audit_log ORDER BY timestamp DESC LIMIT 500", conn)
        if df_log.empty:
            st.info("No actions logged yet.")
        else:
            # Highlight security events
            security_actions = {
                "RATE_LIMIT_BLOCK","AMOUNT_BOUNDS_BLOCK","QUARANTINE",
                "QUARANTINE_APPROVE","QUARANTINE_REJECT",
            }
            action_filter = st.multiselect(
                "Filter by action type",
                options=sorted(df_log["action"].unique()),
                key="log_filter")
            if action_filter:
                df_log = df_log[df_log["action"].isin(action_filter)]
            st.dataframe(df_log, hide_index=True, use_container_width=True)
            buf_l = io.BytesIO()
            with pd.ExcelWriter(buf_l, engine="openpyxl") as w:
                df_log.to_excel(w, index=False, sheet_name="Action Log")
            st.download_button("📥 Export action log", buf_l.getvalue(),
                               "action_log.xlsx", "application/vnd.ms-excel")

    with audit_t3:
        st.markdown("#### Quarantine Log — All Submissions Held for Review")
        df_qlog = get_quarantine()
        if df_qlog.empty:
            st.info("No quarantine entries.")
        else:
            status_filter = st.selectbox(
                "Show", ["All","pending","approved","rejected"],
                key="q_status_filter")
            if status_filter != "All":
                df_qlog = df_qlog[df_qlog["status"]==status_filter]
            st.dataframe(df_qlog, hide_index=True, use_container_width=True)

    with audit_t4:
        st.markdown("#### Rate Limit Events")
        st.caption(
            "Shows citizens who hit the submission rate limit. "
            "Repeated hits from the same ID may indicate automated abuse."
        )
        with sqlite3.connect(DB_PATH) as conn:
            df_rl = pd.read_sql_query(
                "SELECT actor_id, detail, timestamp FROM audit_log "
                "WHERE action='RATE_LIMIT_BLOCK' "
                "ORDER BY timestamp DESC LIMIT 200", conn)
        if df_rl.empty:
            st.info("No rate limit events recorded.")
        else:
            # Frequency table
            freq = df_rl["actor_id"].value_counts().reset_index()
            freq.columns = ["Citizen ID", "Times Blocked"]
            st.markdown("**Most blocked submitters:**")
            st.dataframe(freq.head(20), hide_index=True, use_container_width=True)
            st.divider()
            st.markdown("**Full event log:**")
            st.dataframe(df_rl, hide_index=True, use_container_width=True)