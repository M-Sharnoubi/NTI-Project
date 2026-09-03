"""
Shared design system for the Streamlit app — modern white + blue theme,
RTL-native for Arabic.

Import CUSTOM_CSS and inject once in main.py with st.markdown(..., unsafe_allow_html=True).
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800;900&display=swap');

:root {
    --brand-primary: #2563EB;
    --brand-primary-hover: #1D4ED8;
    --brand-dark: #0F172A;
    --brand-gradient: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
    --bg-surface: #F8FAFC;
    --card-bg: #FFFFFF;
    --border-subtle: #E2E8F0;
    --border-accent: #BFDBFE;
    --text-heading: #0F172A;
    --text-body: #334155;
    --text-muted: #64748B;
    
    /* Semantic Colors */
    --danger: #EF4444;
    --danger-soft: #FEF2F2;
    --warning: #F59E0B;
    --warning-soft: #FFFBEB;
    --success: #10B981;
    --success-soft: #ECFDF5;
    --neutral: #64748B;
    --neutral-soft: #F1F5F9;

    /* Shadows & Radii */
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-soft: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    --shadow-hover: 0 10px 25px -5px rgba(37, 99, 235, 0.1);
}

/* ---------- Base RTL & Typography ---------- */
html, body, [class*="css"] {
    font-family: 'Tajawal', sans-serif !important;
    color: var(--text-body);
}

.stApp {
    background-color: var(--bg-surface);
    direction: rtl;
}

.block-container {
    max-width: 1000px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Clean Header Section */
.app-header {
    background: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-soft);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.app-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--text-heading);
    margin: 0;
    background: var(--brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.app-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Hide Streamlit Chrome */
section[data-testid="stSidebar"], #MainMenu, footer, header {
    visibility: hidden;
}

/* ---------- Tabs Styling ---------- */
div[data-baseweb="tab-list"] {
    background-color: #E2E8F0;
    padding: 4px;
    border-radius: var(--radius-md);
    gap: 4px;
    margin-bottom: 1.5rem;
}

button[data-baseweb="tab"] {
    font-family: 'Tajawal', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--card-bg) !important;
    color: var(--brand-primary) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
}

div[data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ---------- Chat Area Improvements ---------- */
div[data-testid="stChatMessage"] {
    background-color: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-soft);
    direction: rtl;
    text-align: right;
    transition: transform 0.15s ease;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background-color: #EFF6FF;
    border-color: var(--border-accent);
}

div[data-testid="stChatMessage"] p {
    font-size: 1rem;
    line-height: 1.7;
    color: var(--text-heading);
}

/* Chat Input Bar */
div[data-testid="stChatInput"] {
    border-radius: var(--radius-md);
}

div[data-testid="stChatInput"] textarea {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}

div[data-testid="stChatInput"]:focus-within textarea {
    border-color: var(--brand-primary) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

/* ---------- Dashboard Metrics & Cards ---------- */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    box-shadow: var(--shadow-soft);
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--text-heading);
}

.metric-label {
    font-size: 0.875rem;
    color: var(--text-muted);
}

/* Priority Cards */
.ticket-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    background-color: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-right: 5px solid var(--neutral);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-soft);
    transition: all 0.2s ease;
}

.ticket-card:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}

.ticket-card.high   { border-right-color: var(--danger); }
.ticket-card.medium { border-right-color: var(--warning); }
.ticket-card.low    { border-right-color: var(--neutral); }

.priority-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
}

.priority-badge.high { background: var(--danger-soft); color: var(--danger); }
.priority-badge.medium { background: var(--warning-soft); color: var(--warning); }
.priority-badge.low { background: var(--neutral-soft); color: var(--neutral); }

/* Escalation Warning Alert */
.escalation-box {
    background: var(--danger-soft);
    border: 1px solid #FCA5A5;
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
    color: var(--danger);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
"""