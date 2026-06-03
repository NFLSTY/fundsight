# FundSight — AI-Powered Personal Finance Assistant

> Built with Python + Flask + Google Gemini AI. No database required.

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Your Gemini API Key
Create your own `.env` file and write this inside:
```python
GEMINI_API_KEY = 'your_actual_gemini_api_key_here'
```
Or set as environment variable:
```bash
# Windows
set GEMINI_API_KEY=your_key_here

# Mac/Linux
export GEMINI_API_KEY=your_key_here
```

### 3. Run the App
```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 📁 Project Structure
```
FundSight/
├── app.py                   # Flask backend + all API routes
├── requirements.txt         # Python dependencies
├── mock_expenses.csv        # Test file for CSV import
└── templates/
    ├── base.html            # Base layout with sidebar nav
    ├── index.html           # Landing page
    ├── dashboard.html       # Main dashboard
    ├── budget.html          # Budget planner + CSV import
    ├── savings.html         # Savings goal planner
    ├── investment.html      # Investment guidance
    ├── chat.html            # AI chatbot interface
    └── reports.html         # Report generation + PDF download
```

## 🧠 AI Features (Gemini-Powered)
| Feature | Description |
|---|---|
| Budget Analysis | Health score, 50/30/20 rule check, overspending alerts |
| Savings Plan | Monthly milestones, feasibility check, strategies |
| Investment Guidance | Risk-based Indonesian market recommendations |
| AI Chatbot | Context-aware assistant with your full financial profile |
| Quick Insights | 4-point summary: positive, warning, opportunity, next step |
| Report Narrative | Full AI-written financial health summary |

## 📊 Charts (Matplotlib)
- **Pie chart** — expense distribution
- **Bar chart** — category comparison  
- **50/30/20 allocation** — budget rule visualization
- **Savings projection** — line graph with goal milestone

## 📄 Export Options
- **PDF Report** — formatted with tables via ReportLab
- **Text file** — plain text summary
- **Print** — browser print dialog

## 💡 Tips
- Use `mock_expenses.csv` to test CSV import in the Budget page
- The AI chatbot remembers the last 20 messages per session
- All data is in-memory — restarting the server resets data
- Risk profile persists within a session

## 🔧 Tech Stack
- **Backend**: Python 3.12+, Flask 3.1.0
- **AI**: Google Gemini 3.1 Flash Lite
- **Charts**: Matplotlib (server-side rendering)
- **PDF**: ReportLab
- **Frontend**: Vanilla JS + CSS custom properties (no framework)
- **Storage**: In-memory Python dict (no database)
