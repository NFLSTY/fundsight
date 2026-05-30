from flask import Flask, render_template, request, jsonify, send_file, session
import json
import os
import csv
import io
import base64
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
    

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from google import genai
from dotenv import load_dotenv
from flask_session import Session

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)

# Configure Gemini
load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.5-flash"

def get_finance_data():
    if "finance_data" not in session:
        session["finance_data"] = {
            "income": 0,
            "expenses": {},
            "savings_goal": {"name": "", "amount": 0, "timeline_months": 0},
            "risk_level": "Medium",
            "chat_history": [],
            "historical_expenses": [3000000, 3100000, 2950000, 3300000, 3200000, 3400000] # dummy data
        }
    return session["finance_data"]

@app.after_request
def after_request(response):
    if "finance_data" in session:
        session.modified = True
    return response

def save_finance_data(data):
    session["finance_data"] = data
    session.modified = True

def get_ai_response(prompt, system_context="", stream=False):
    try:
        full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        return f"AI service error: {str(e)}. Please check your Gemini API key."

def generate_chart(chart_type, data, title):
    """Generate charts and return as base64 string."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Dark theme
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    # Expanded color palette to handle many expenses
    base_colors = ['#00d4aa', '#7c3aed', '#f59e0b', '#ef4444', '#3b82f6', '#10b981', '#f97316', '#ec4899', 
                   '#8b5cf6', '#06b6d4', '#14b8a6', '#84cc16', '#eab308', '#f43f5e', '#a855f7', '#38bdf8']
    
    # Repeat colors if data length exceeds palette length
    colors_palette = (base_colors * ((len(data) // len(base_colors)) + 1))[:len(data)]
    
    if chart_type == "pie" and data:
        labels = list(data.keys())
        values = list(data.values())
        wedges, texts, autotexts = ax.pie(
            values, labels=None, autopct='%1.1f%%',
            colors=colors_palette[:len(values)],
            startangle=90, pctdistance=0.85,
            wedgeprops=dict(width=0.6, edgecolor='#0d1117', linewidth=2)
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5),
                          frameon=False, labelcolor='#c9d1d9', fontsize=9)
        ax.set_title(title, color='#e6edf3', fontsize=13, fontweight='bold', pad=15)

    elif chart_type == "bar" and data:
        labels = list(data.keys())
        values = list(data.values())
        bars = ax.bar(labels, values, color=colors_palette[:len(values)],
                     edgecolor='#0d1117', linewidth=1.5, width=0.6)
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=9)
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, color='#e6edf3', fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel('Amount (Rp)', color='#8b949e', fontsize=10)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.01,
                   f'Rp {val:,.0f}', ha='center', va='bottom', color='#c9d1d9', fontsize=8)
        plt.xticks(rotation=30, ha='right')

    elif chart_type == "savings_projection":
        months = data.get("months", 12)
        monthly_saving = data.get("monthly_saving", 0)
        goal = data.get("goal", 0)
        x = list(range(months + 1))
        y = [i * monthly_saving for i in x]
        ax.fill_between(x, y, alpha=0.3, color='#00d4aa')
        ax.plot(x, y, color='#00d4aa', linewidth=2.5, marker='o', markersize=4)
        ax.axhline(y=goal, color='#f59e0b', linestyle='--', linewidth=1.5, label=f'Goal: Rp {goal:,.0f}')
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=9)
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, color='#e6edf3', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Months', color='#8b949e')
        ax.set_ylabel('Savings (Rp)', color='#8b949e')
        ax.legend(frameon=False, labelcolor='#c9d1d9')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', data=get_finance_data())

@app.route('/budget')
def budget():
    return render_template('budget.html', data=get_finance_data())

@app.route('/savings')
def savings():
    return render_template('savings.html', data=get_finance_data())

@app.route('/investment')
def investment():
    return render_template('investment.html', data=get_finance_data())

@app.route('/chat')
def chat():
    return render_template('chat.html', data=get_finance_data())

@app.route('/reports')
def reports():
    return render_template('reports.html', data=get_finance_data())

# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.route('/api/status')
def api_status():
    if not GEMINI_API_KEY:
        return jsonify({"status": "error", "message": "API Key missing"})
    try:
        # Just check if the client can be initialized and we have an API key
        # Removing the actual ping request that can cause issues or unnecessary quota usage
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/update_income', methods=['POST'])
def update_income():
    finance_data = get_finance_data()
    data = request.json
    finance_data = get_finance_data()
    finance_data['income'] = float(data.get('income', 0))
    save_finance_data(finance_data)
    return jsonify({"success": True, "income": finance_data['income']})

@app.route('/api/add_expense', methods=['POST'])
def add_expense():
    finance_data = get_finance_data()
    data = request.json
    category = data.get('category', '').strip()
    amount = float(data.get('amount', 0))
    finance_data = get_finance_data()
    if category and amount > 0:
        finance_data['expenses'][category] = finance_data['expenses'].get(category, 0) + amount
    save_finance_data(finance_data)
    return jsonify({"success": True, "expenses": finance_data['expenses']})

@app.route('/api/remove_expense', methods=['POST'])
def remove_expense():
    finance_data = get_finance_data()
    data = request.json
    category = data.get('category', '')
    finance_data = get_finance_data()
    if category in finance_data['expenses']:
        del finance_data['expenses'][category]
    save_finance_data(finance_data)
    return jsonify({"success": True, "expenses": finance_data['expenses']})

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    finance_data = get_finance_data()
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        reader = csv.DictReader(stream)
        added = 0
        finance_data = get_finance_data()
        
        # Save preview of parsed rows
        preview_data = []
        
        for row in reader:
            cat = row.get('category', row.get('Category', ''))
            amt = row.get('amount', row.get('Amount', 0))
            if cat and amt:
                finance_data['expenses'][cat] = finance_data['expenses'].get(cat, 0) + float(amt)
                added += 1
                if len(preview_data) < 5:
                    preview_data.append({"category": cat, "amount": amt})
        save_finance_data(finance_data)
        return jsonify({
            "success": True, 
            "added": added, 
            "expenses": finance_data['expenses'],
            "preview": preview_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/update_savings_goal', methods=['POST'])
def update_savings_goal():
    finance_data = get_finance_data()
    data = request.json
    finance_data = get_finance_data()
    
    # Optional logic directly requested: Add savings goal as an ongoing monthly expense
    new_goal = {
        "name": data.get('name', ''),
        "amount": float(data.get('amount', 0)),
        "timeline_months": int(data.get('timeline_months', 12))
    }
    
    finance_data['savings_goal'] = new_goal
    
    # Calculate monthly expense for savings and add to expenses list
    if new_goal["amount"] > 0 and new_goal["timeline_months"] > 0:
        monthly_saving = new_goal["amount"] / new_goal["timeline_months"]
        expense_name = f"Savings Goal: {new_goal['name']}" if new_goal['name'] else "Savings Goal"
        finance_data['expenses'][expense_name] = monthly_saving
        
    save_finance_data(finance_data)
    return jsonify({"success": True, "savings_goal": finance_data['savings_goal']})

@app.route('/api/update_risk', methods=['POST'])
def update_risk():
    finance_data = get_finance_data()
    data = request.json
    finance_data = get_finance_data()
    finance_data['risk_level'] = data.get('risk_level', 'Medium')
    save_finance_data(finance_data)
    return jsonify({"success": True})

@app.route('/api/get_data')
def get_data():
    finance_data = get_finance_data()
    finance_data = get_finance_data()
    total_expenses = sum(finance_data['expenses'].values())
    remaining = finance_data['income'] - total_expenses
    savings_goal = finance_data['savings_goal']
    monthly_needed = (savings_goal['amount'] / savings_goal['timeline_months']
                      if savings_goal['timeline_months'] > 0 else 0)
    return jsonify({
        **finance_data,
        "total_expenses": total_expenses,
        "remaining": remaining,
        "monthly_savings_needed": monthly_needed
    })

@app.route('/api/chart/<chart_type>')
def get_chart(chart_type):
    finance_data = get_finance_data()
    finance_data = get_finance_data()
    if chart_type == "expense_pie":
        if not finance_data['expenses']:
            return jsonify({"error": "No expense data"}), 400
        img = generate_chart("pie", finance_data['expenses'], "Expense Distribution")
        return jsonify({"image": img})

    elif chart_type == "expense_bar":
        if not finance_data['expenses']:
            return jsonify({"error": "No expense data"}), 400
        img = generate_chart("bar", finance_data['expenses'], "Expenses by Category")
        return jsonify({"image": img})

    elif chart_type == "budget_allocation":
        income = finance_data['income']
        if income <= 0:
            return jsonify({"error": "Set income first"}), 400
        allocation = {"Needs (50%)": income * 0.5, "Savings (20%)": income * 0.2, "Wants (30%)": income * 0.3}
        img = generate_chart("pie", allocation, "50/20/30 Budget Allocation")
        return jsonify({"image": img})

    elif chart_type == "savings_projection":
        sg = finance_data['savings_goal']
        if sg['timeline_months'] <= 0 or sg['amount'] <= 0:
            return jsonify({"error": "Set savings goal first"}), 400
        monthly_saving = sg['amount'] / sg['timeline_months']
        img = generate_chart("savings_projection", {
            "months": sg['timeline_months'],
            "monthly_saving": monthly_saving,
            "goal": sg['amount']
        }, f"Savings Projection — {sg['name']}")
        return jsonify({"image": img})

    return jsonify({"error": "Unknown chart type"}), 400

@app.route('/api/ai_budget', methods=['POST'])
def ai_budget():
    finance_data = get_finance_data()
    data = request.json
    income = data.get('income', finance_data['income'])
    expenses = data.get('expenses', finance_data['expenses'])
    total_exp = sum(expenses.values())
    
    prompt = f"""You are a professional financial advisor. Analyze this user's finances and provide structured budget advice.

User Financial Data:
- Monthly Income: Rp {income:,.0f}
- Total Expenses: Rp {total_exp:,.0f}
- Remaining: Rp {income - total_exp:,.0f}
- Expense Breakdown: {json.dumps(expenses, indent=2)}

Provide:
1. **Budget Health Score** (0-100) with brief explanation
2. **50/30/20 Rule Analysis** — how well they follow it
3. **Top 3 Overspending Areas** (if any)
4. **3 Actionable Recommendations** to optimize their budget
5. **Savings Opportunity** — how much more they could save

Example Output Formatting (Do not copy the numbers, use as style guide):
**Budget Health Score: 75/100**
Your budget is mostly healthy, but there's room to grow.

**50/30/20 Rule Analysis**
Your Needs are at 60% (above the 50% target). You are on track for Wants.

**Top Overspending Areas**
- Food & Dining: High for your income bracket
- Entertainment: Can be optimized

**Actionable Recommendations**
- Cook at home at least 4 nights a week to save food costs.
- Cancel one unused streaming subscription.
- Allocate an extra 5% to savings automatically on payday.

Keep it concise, practical, and motivating. Format clearly with headers."""
    
    response = get_ai_response(prompt)
    return jsonify({"advice": response})

@app.route('/api/ai_savings', methods=['POST'])
def ai_savings():
    finance_data = get_finance_data()
    data = request.json
    goal = data.get('goal', finance_data['savings_goal'])
    income = data.get('income', finance_data['income'])
    total_exp = sum(finance_data['expenses'].values())
    available = income - total_exp
    
    prompt = f"""You are a savings strategist. Create a personalized monthly savings plan.

Goal: {goal.get('name', 'Savings Goal')}
Target Amount: Rp {goal.get('amount', 0):,.0f}
Timeline: {goal.get('timeline_months', 12)} months
Monthly Income: Rp {income:,.0f}
Current Available After Expenses: Rp {available:,.0f}
Monthly Savings Needed: Rp {goal.get('amount', 0) / max(goal.get('timeline_months', 1), 1):,.0f}

Provide:
1. **Feasibility Assessment** — is this goal realistic?
2. **Monthly Savings Breakdown** — week-by-week or monthly milestones
3. **3 Specific Strategies** to reach the goal faster
4. **Risk Buffer** — emergency fund recommendation
5. **Motivation Milestone** — celebrate at what checkpoints?

Example formatting style to follow:
**Feasibility Assessment: Highly Realistic**
Based on your current available income, you can achieve this within 10 months if you stay disciplined.

**Monthly Savings Breakdown**
- Week 1: Rp 500,000
- Week 2: Rp 500,000...

Be encouraging and specific with Rupiah amounts."""
    
    response = get_ai_response(prompt)
    return jsonify({"plan": response})

@app.route('/api/ai_investment', methods=['POST'])
def ai_investment():
    finance_data = get_finance_data()
    data = request.json
    risk = data.get('risk_level', finance_data['risk_level'])
    income = data.get('income', finance_data['income'])
    available = income - sum(finance_data['expenses'].values())
    
    prompt = f"""You are a certified financial planner specializing in Indonesian investments.

User Profile:
- Risk Tolerance: {risk}
- Monthly Income: Rp {income:,.0f}
- Monthly Investable Amount: Rp {max(available * 0.5, 0):,.0f} (estimated)

Provide investment guidance for an Indonesian investor:
1. **Investment Strategy Overview** for {risk} risk profile
2. **Asset Allocation** (specific percentages)
3. **Recommended Indonesian Investment Products** (e.g., Reksa Dana, ORI, Saham IDX, Deposito, P2P Lending)
4. **Starter Portfolio** — concrete first steps with Rupiah amounts
5. **Key Risks to Watch** for this profile
6. **Timeline for Results** — realistic expectations

Be specific to Indonesian market context. Include product names where relevant."""
    
    response = get_ai_response(prompt)
    return jsonify({"guidance": response})

@app.route('/api/chat_stream', methods=['POST'])
def chat_stream():
    finance_data = get_finance_data()
    data = request.json
    user_message = data.get('message', '')
    
    finance_context = f"""Current user financial context:
- Monthly Income: Rp {finance_data['income']:,.0f}
- Total Expenses: Rp {sum(finance_data['expenses'].values()):,.0f}
- Expense Categories: {json.dumps(finance_data['expenses'])}
- Savings Goal: {finance_data['savings_goal'].get('name', 'Not set')} — Rp {finance_data['savings_goal'].get('amount', 0):,.0f}
- Risk Tolerance: {finance_data['risk_level']}
"""
    
    history_context = ""
    if finance_data['chat_history']:
        recent = finance_data['chat_history'][-6:]
        history_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])
    
    system_prompt = f"""You are Fundy, a friendly and knowledgeable personal finance assistant focused on the Indonesian market.
You help users manage budgets, plan savings, understand investments, and make smart financial decisions.
Always give practical, actionable advice. Use Rupiah (Rp) for currency. Be warm but professional.
Whenever users ask something outside the topic, answer honestly that your purposed is only for finance needs.
CRITICAL LANGUAGE RULE: You MUST reply in the EXACT SAME language that the user uses in their last message. If the user asks in English, reply ONLY in English. If the user asks in Indonesian, reply ONLY in Indonesian. Do not mix languages.

Do not proactively summarize the user's financial status in every response unless relevant to their question.

{finance_context}

Recent conversation:
{history_context}"""
    
    finance_data['chat_history'].append({"role": "user", "content": user_message})
    save_finance_data(finance_data)
    
    def generate():
        full_prompt = f"{system_prompt}\n\n{user_message}"
        response = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=full_prompt
        )
        full_response = ""
        for chunk in response:
            text = chunk.text
            full_response += text
            yield text
        
        fd = session.get('finance_data', {})
        fd['chat_history'].append({"role": "assistant", "content": full_response})
        if len(fd['chat_history']) > 20:
            fd['chat_history'] = fd['chat_history'][-20:]
        session['finance_data'] = fd
        session.modified = True

    from flask import Response, stream_with_context
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    finance_data = get_finance_data()
    data = request.json
    user_message = data.get('message', '')
    
    finance_data = get_finance_data()
    finance_context = f"""Current user financial context:
- Monthly Income: Rp {finance_data['income']:,.0f}
- Total Expenses: Rp {sum(finance_data['expenses'].values()):,.0f}
- Expense Categories: {json.dumps(finance_data['expenses'])}
- Savings Goal: {finance_data['savings_goal'].get('name', 'Not set')} — Rp {finance_data['savings_goal'].get('amount', 0):,.0f}
- Risk Tolerance: {finance_data['risk_level']}
"""
    
    history_context = ""
    if finance_data['chat_history']:
        recent = finance_data['chat_history'][-6:]
        history_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])
    
    system_prompt = f"""You are Fundy, a friendly and knowledgeable personal finance assistant focused on the Indonesian market.
You help users manage budgets, plan savings, understand investments, and make smart financial decisions.
Always give practical, actionable advice. Use Rupiah (Rp) for currency. Be warm but professional.
Whenever users ask something outside the topic, answer honestly that your purposed is only for finance needs.
CRITICAL LANGUAGE RULE: You MUST reply in the EXACT SAME language that the user uses in their last message. If the user asks in English, reply ONLY in English. If the user asks in Indonesian, reply ONLY in Indonesian. Do not mix languages.

Do not proactively summarize the user's financial status in every response unless relevant to their question.

{finance_context}

Recent conversation:
{history_context}"""
    
    response = get_ai_response(user_message, system_prompt)
    
    finance_data['chat_history'].append({"role": "user", "content": user_message})
    finance_data['chat_history'].append({"role": "assistant", "content": response})
    
    # Keep only last 20 messages
    if len(finance_data['chat_history']) > 20:
        finance_data['chat_history'] = finance_data['chat_history'][-20:]
    
    save_finance_data(finance_data)
    return jsonify({"response": response})

@app.route('/api/ai_insights')
def ai_insights():
    finance_data = get_finance_data()
    income = finance_data['income']
    expenses = finance_data['expenses']
    total_exp = sum(expenses.values())
    
    if income <= 0:
        return jsonify({"insights": "Please set your monthly income to get personalized insights."})
    
    prompt = f"""Analyze this financial data and provide key insights in a concise format:

Income: Rp {income:,.0f}
Expenses: {json.dumps(expenses)}
Total Spent: Rp {total_exp:,.0f}
Savings Rate: {((income - total_exp) / income * 100):.1f}%

Give exactly 4 insights in this format:
🟢 POSITIVE: [one thing they're doing well]
🔴 WARNING: [biggest risk or overspending area]  
💡 OPPORTUNITY: [best savings opportunity]
📈 NEXT STEP: [single most impactful action to take this month]

Keep each insight to 1-2 sentences max."""
    
    response = get_ai_response(prompt)
    return jsonify({"insights": response})

@app.route('/api/download_report')
def download_report():
    finance_data = get_finance_data()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=24, textColor=colors.HexColor('#00d4aa'), spaceAfter=6)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                    fontSize=14, textColor=colors.HexColor('#7c3aed'), spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=3)

    story.append(Paragraph("FundSight Financial Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}", body_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Financial Summary", heading_style))
    income = finance_data['income']
    total_exp = sum(finance_data['expenses'].values())
    summary_data = [
        ['Metric', 'Amount'],
        ['Monthly Income', f"Rp {income:,.0f}"],
        ['Total Expenses', f"Rp {total_exp:,.0f}"],
        ['Net Savings', f"Rp {income - total_exp:,.0f}"],
        ['Savings Rate', f"{((income - total_exp) / income * 100):.1f}%" if income > 0 else "N/A"],
    ]
    t = Table(summary_data, colWidths=[3*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00d4aa')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    if finance_data['expenses']:
        story.append(Paragraph("Expense Breakdown", heading_style))
        exp_data = [['Category', 'Amount', '% of Income']]
        for cat, amt in sorted(finance_data['expenses'].items(), key=lambda x: -x[1]):
            pct = f"{(amt/income*100):.1f}%" if income > 0 else "N/A"
            exp_data.append([cat, f"Rp {amt:,.0f}", pct])
        t2 = Table(exp_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#7c3aed')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
            ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.2*inch))

    sg = finance_data['savings_goal']
    if sg['amount'] > 0:
        story.append(Paragraph("Savings Goal", heading_style))
        story.append(Paragraph(f"Goal: {sg['name']}", body_style))
        story.append(Paragraph(f"Target: Rp {sg['amount']:,.0f}", body_style))
        story.append(Paragraph(f"Timeline: {sg['timeline_months']} months", body_style))
        monthly = sg['amount'] / sg['timeline_months'] if sg['timeline_months'] > 0 else 0
        story.append(Paragraph(f"Required Monthly Savings: Rp {monthly:,.0f}", body_style))

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name=f"FundSight_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                     mimetype='application/pdf')

@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    finance_data = get_finance_data()
    finance_data = get_finance_data()
    finance_data['income'] = 0
    finance_data['expenses'] = {}
    finance_data['savings_goal'] = {"name": "", "amount": 0, "timeline_months": 0}
    finance_data['risk_level'] = "Medium"
    finance_data['chat_history'] = []
    save_finance_data(finance_data)
    return jsonify({"success": True})

@app.route('/api/forecast_expense')
def forecast_expense():
    finance_data = get_finance_data()
    hist = finance_data.get('historical_expenses', [])
    current_total = sum(finance_data['expenses'].values())
    if current_total > 0:
        hist = hist + [current_total]
    
    if len(hist) < 2:
        return jsonify({"forecast": current_total})
        
    X = np.array(range(len(hist))).reshape(-1, 1)
    y = np.array(hist)
    
    reg = LinearRegression().fit(X, y)
    next_month_idx = np.array([[len(hist)]])
    prediction = reg.predict(next_month_idx)[0]
    
    return jsonify({
        "forecast": max(0, float(prediction)),
        "history": hist
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)