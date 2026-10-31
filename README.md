# Cynbit Inquiry Intelligence & Proposal Assistant — PoC

## What it demonstrates
- TF-IDF + Logistic Regression service-category classification
- Requirement tagging using transparent keyword rules
- Transparent heuristic priority score with visible reasons
- Clarification question suggestions
- Templated, editable initial proposal draft
- CSV and text exports

## Data and limitations
- The classifier is trained at app startup on a small, hand-written synthetic dataset embedded in `app.py`.
- `sample_inquiries.csv` contains fictional test inquiries.
- The model has not been trained or validated on Cynbit data, and its confidence is not calibrated.
- The priority score is a hand-designed heuristic, not a conversion probability.
- This is not connected to Cynbit's website, CRM, email, or internal systems.
- The proposal is not a commercial quote or approved commitment. Human review is required.

## Windows PowerShell
Requires Python 3.10 or 3.11.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If activation works in your environment, you can also use:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Demo steps
1. Use the fictional default inquiry and click Analyze inquiry.
2. Try a cloud, CRM, analytics, or marketing example.
3. Change budget/timeline statuses and observe the heuristic score explanation.
4. Review the proposal draft, then download the TXT and CSV files.

## Files
- `app.py`: Streamlit application and synthetic model training examples
- `requirements.txt`: dependencies
- `sample_inquiries.csv`: fictional test examples
- `README.md`: setup and limitations
