# 🔎 CivicTech Open Finance Tracker — Kenya

An immutable, AI-powered, open-source platform for tracking and analysing
political campaign spending in Kenya, with searchable politician profiles,
community investigations, and a multi-layer fraud-prevention system.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/civictech-finance-tracker.git
cd civictech-finance-tracker

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
> **Password:** You must set your own admin password in your `.streamlit/secrets.toml` file before running the app.

---

## Features

### Politician Profiles
- Searchable directory of politicians (name, role, party, county, constituency)
- Per-profile financial ledger, Benford analysis, and community findings
- Photo, bio, and role metadata per profile
- All 47 Kenyan counties and all major roles in dropdowns

### Financial Records
- Append-only immutable SQLite ledger — nothing is ever deleted
- Corrections create a new versioned row; the original is permanently archived
- Multi-format bulk upload: `.xlsx`, `.xls`, `.csv`, `.tsv`, `.txt`, `.json`, `.jsonl`, `.ods`
- Smart column matching: ~100 aliases per field including Swahili and IFMIS terms
- Live KES currency conversion (USD, GBP, EUR, UGX, TZS, ETB)

### AI / Statistical Analysis
- **Benford's Law** — digit distribution suspicion score 0–100 per politician
- **Isolation Forest** — ML outlier detection on KES-normalised amounts
- **NLP keyword flagging** — description scanner for suspicious terms (hidden in secrets)

### Fraud Prevention
| Layer | What it does |
|---|---|
| Cross-contamination scan | Detects names in uploaded files that don't match the target politician |
| Duplicate detection | Blocks identical amount+description+currency within 30 days |
| Amount bounds | Hard KES min/max per transaction (configurable in secrets) |
| Rate limiting | Max submissions per citizen per hour (sliding 1-hour window) |
| Quarantine queue | Low-trust citizens' submissions held for moderator review |
| Shadow voting | Bot votes register as 0 — UI shows "success" so bots never know |
| Append-only ledger | Nothing can be deleted or silently overwritten |

### Community Reviews
- Anonymous Civic IDs (no email, no phone)
- Proof-of-effort trust scoring — earn points by publishing detailed reviews
- Reviews can link to a specific expense record or the whole profile
- Upvote system with rate limiting and shadow voting

### Moderator Tools
- Global Moderator Queue — approve or reject all pending quarantine items
- Per-politician Moderator Queue tab on each profile
- Audit Trail with 4 tabs: expense history, action log, quarantine log, rate-limit events

---

## Deployment (Free — Streamlit Cloud)

1. Push `app.py`, `requirements.txt`, `README.md`, `CONTRIBUTING.md` to GitHub.
   **Do NOT push** `.streamlit/secrets.toml` or `campaign_finance.db`.

2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo → `app.py`.

3. In **Settings → Secrets**, paste the contents of `.streamlit/secrets.toml`
   with your own values.

4. Click **Deploy**.

---

## Security Model (Open Core)

The algorithm logic is fully public. The *parameters* that make the traps
effective (thresholds, keyword lists, rate limits) are stored in `st.secrets`
and never committed to the repo.

> "We provide the math. The citizens provide the verdict."

---

## ⚠️ Disclaimer

AI flags indicate **statistical anomalies**, not conclusions of wrongdoing.
All flagged items are labelled "Items for Review". Human analysts and
citizens make the final judgement.

---

## License

MIT — free to use, adapt, and deploy for civic tech purposes.
