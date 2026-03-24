# Contributing to CivicTech Open Finance Tracker

Thank you for your interest in helping build civic transparency tooling for Kenya.

---

## Ways to contribute

- **Bug reports** — open a GitHub Issue with steps to reproduce
- **Feature ideas** — open a GitHub Issue tagged `enhancement`
- **Code contributions** — fork → branch → Pull Request (PR)
- **NGO partnerships** — reach out via Issues tagged `partnership`
- **Data contributions** — use the Excel upload feature in the live app

---

## Pull Request process

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes. Keep commits small and descriptive.
3. Test locally with `streamlit run app.py`.
4. Open a Pull Request against `main` with a clear description of:
   - What the PR changes
   - Why the change is needed
   - How you tested it

The core team reviews all PRs before merging. We may ask for changes.

---

## What we will NOT merge

- Anything that adds security thresholds, NLP keyword lists, or Trust Score
  math directly into public code (these belong in `st.secrets`)
- Code that weakens the append-only constraint on the `expenses` table
- UI changes that label AI flags as "fraud" or "corrupt"

---

## Code style

- Python 3.11+
- Follow PEP 8
- Keep functions small and well-named
- Add docstrings to all new functions

---

## Reporting security vulnerabilities

Do **not** open a public GitHub Issue for security vulnerabilities.
Email the maintainers directly (listed in the repo profile).
