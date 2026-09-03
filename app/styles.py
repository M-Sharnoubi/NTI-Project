"""
Shared design system for the Streamlit app — Modern Minimal Light Theme.
RTL-native for Arabic with clean hierarchy and subtle focus states.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Tajawal:wght@400;500;700&display=swap');

:root {
    /* Neutral Light Palette */
    --bg-surface: #F8FAFC;
    --card-bg: #FFFFFF;
    
    /* Subtle Accent (Restrained Indigo) */
    --accent-primary: #3B82F6;
    --accent-light: #F0F6FF;
    
    /* Border & Separators */
    --border-subtle: #E2E8F0;
    --border-strong: #CBD5E1;
    
    /* Text Hierarchy (Dark Charcoal) */
    --text-heading: #0F172A;
    --text-body: #334155;
    --text-muted: #64748B;
    
    /* Functional Colors */
    --danger: #DC2626;
    --danger-soft: #FEF2F2;
    --danger-border: #FECACA;
    --warning: #D97706;
    --warning-soft: #FFFBEB;
    --success: #16A34A;
    --success-soft: #F0FDF4;

    /* Elevation & Layout */
    --radius-sm: 6px;
    --radius-md: 8px;
    --shadow-subtle: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02);
}

/* ---------- Global Reset & Typography ---------- */
html, body, [class*="css"] {
    font-family: 'Tajawal', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-body);
}

.stApp {
    background-color: var(--bg-surface);
    direction: rtl;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hide Streamlit Chrome */
section[data-testid="stSidebar"], #MainMenu, footer, header {
    visibility: hidden;
}

/* ---------- App Header Bar ---------- */
.app-header {
    background: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.app-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0;
    color: var(--text-heading);
}

.app-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 2px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
}

.status-dot {
    width: 6px;
    height: 6px;
    background-color: var(--success);
    border-radius: 50%;
}

/* ---------- Clean Tabs Navigation ---------- */
div[data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 1px solid var(--border-subtle);
    padding: 0;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}

button[data-baseweb="tab"] {
    font-family: 'Tajawal', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: var(--text-muted) !important;
    border-radius: 0 !important;
    padding: 8px 0 !important;
    border: none !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-heading) !important;
    border-bottom-color: var(--text-heading) !important;
}

div[data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ---------- Chat Messages ---------- */
div[data-testid="stChatMessage"] {
    background-color: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--shadow-subtle);
    direction: rtl;
    text-align: right;
}

/* User Message Minimal Tint */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background-color: var(--bg-surface);
    border-color: var(--border-subtle);
}

div[data-testid="stChatMessage"] p {
    font-size: 0.925rem;
    line-height: 1.6;
    color: var(--text-body);
}

/* Chat Input Bar */
div[data-testid="stChatInput"] textarea {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-heading) !important;
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1rem !important;
    box-shadow: var(--shadow-subtle) !important;
}

div[data-testid="stChatInput"]:focus-within textarea {
    border-color: var(--text-muted) !important;
}

/* ---------- Dashboard Component Styling ---------- */
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
    padding: 1rem;
    box-shadow: var(--shadow-subtle);
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-heading);
    line-height: 1.2;
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-top: 4px;
}

/* Ticket Items */
.ticket-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
    background-color: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-subtle);
}

.priority-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.priority-badge.high { 
    background: var(--danger-soft); 
    color: var(--danger); 
    border: 1px solid var(--danger-border);
}

.priority-badge.medium { 
    background: var(--warning-soft); 
    color: var(--warning); 
    border: 1px solid #FDE68A;
}

.priority-badge.low { 
    background: var(--bg-surface); 
    color: var(--text-muted); 
    border: 1px solid var(--border-subtle);
}

/* Escalation Warning Box */
.escalation-box {
    background: var(--danger-soft);
    border: 1px solid var(--danger-border);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    color: var(--danger);
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Empty State */
.welcome-container {
    text-align: right;
    padding: 2rem 1.5rem;
    background: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    margin: 1rem 0 1.5rem;
    box-shadow: var(--shadow-subtle);
}

.welcome-title {
    color: var(--text-heading);
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.25rem;
}

.welcome-desc {
    color: var(--text-muted);
    font-size: 0.875rem;
}
/* Add or update these rules in styles.py */

/* Create bottom padding on the main chat container so messages don't get covered by the fixed bar */
div[data-testid="stMainBlockContainer"] {
    padding-bottom: 120px !important;
}

/* Pin the chat input container to the bottom of the viewport */
div[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 20px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 100% !important;
    max-width: 900px !important; /* Matches your block-container width */
    z-index: 999 !important;
    background-color: var(--bg-surface) !important;
    padding: 10px 0 !important;
}
</style>
"""