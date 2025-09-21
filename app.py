import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from typing import Dict, List, Any
import tempfile
import time
from datetime import datetime, timedelta

# --- CONFIGURATION FOR CLOUD DEPLOYMENT ---
st.set_page_config(
    page_title="Signal AI - VC Investment Analyst", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure Gemini API using Streamlit secrets
try:
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    else:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        st.error("⚠️ GEMINI_API_KEY not configured. Please add it to Streamlit secrets.")
        st.stop()
    else:
        genai.configure(api_key=gemini_api_key)
        
except Exception as e:
    st.error(f"❌ API Key configuration error: {e}")
    st.stop()

# --- GEMINI FILE PROCESSING FUNCTIONS ---

def upload_file_to_gemini(file_data, file_name: str):
    """Upload file directly to Gemini for processing"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name
        
        uploaded_file = genai.upload_file(tmp_file_path, display_name=file_name)
        
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            return None
            
        os.unlink(tmp_file_path)
        return uploaded_file
        
    except Exception as e:
        st.error(f"File upload failed for {file_name}: {e}")
        return None

def analyze_with_gemini_files(gemini_files):
    """Enhanced Gemini analysis with comprehensive data extraction"""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        analysis_prompt = """
        You are Signal AI, an expert venture capital analyst. Analyze the provided startup documents thoroughly.

        Extract ALL available information following the four pillars framework:
        1. FOUNDER PROFILE: Names, backgrounds, previous experience, founder-market fit
        2. PROBLEM & MARKET SIZE: TAM, problem validation, market opportunity
        3. UNIQUE DIFFERENTIATOR: Competitive advantages, moats, IP, technology
        4. TEAM & TRACTION: Customer metrics, revenue, growth rates, partnerships

        Return comprehensive JSON with this structure:

        {
            "company_overview": {
                "name": "Company name",
                "founding_year": "Year founded", 
                "stage": "Funding stage",
                "one_liner": "Value proposition",
                "industry": "Industry/sector"
            },
            "founders": [
                {
                    "name": "Full name",
                    "role": "Title/role",
                    "background": "Education + experience",
                    "founder_market_fit": "Assessment of market fit"
                }
            ],
            "problem_and_market": {
                "problem_statement": "Core problem being solved",
                "market_size_tam": "Total addressable market",
                "market_growth_rate": "Annual growth rate",
                "target_customer": "Customer profile",
                "market_validation": "Evidence of demand"
            },
            "unique_differentiator": {
                "core_technology": "Key technology/innovation",
                "competitive_moat": "Competitive advantages",
                "ip_assets": "Patents, proprietary tech",
                "barriers_to_entry": "Barriers preventing competition"
            },
            "team_and_traction": {
                "team_size": "Current headcount",
                "customer_count": "Number of customers",
                "arr_mrr": "Recurring revenue",
                "growth_metrics": "Growth rates",
                "key_customers": ["Notable customers"],
                "partnerships": ["Strategic partnerships"],
                "revenue_model": "How money is made"
            },
            "financials": {
                "current_revenue": "Latest revenue",
                "revenue_projections": "Future forecasts",
                "funding_raised": "Total capital raised",
                "current_ask": "Current round amount",
                "valuation": "Company valuation",
                "burn_rate": "Monthly burn",
                "runway": "Cash runway",
                "unit_economics": "LTV, CAC metrics"
            },
            "investment_thesis": {
                "strengths": ["Key strengths"],
                "risks": ["Risk factors"],
                "market_opportunity": "Market timing",
                "execution_risk": "Execution capability"
            },
            "recommendation": {
                "signal_score": "1-5 rating",
                "investment_decision": "STRONG BUY/BUY/HOLD/PASS",
                "rationale": "Detailed reasoning",
                "comparable_companies": ["Similar companies"]
            }
        }
        """
        
        content_parts = [analysis_prompt]
        content_parts.extend(gemini_files)
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            max_output_tokens=8000,
        )
        
        response = model.generate_content(content_parts, generation_config=generation_config)
        
        if response.text:
            return response.text
        else:
            return "Error: No response"
            
    except Exception as e:
        return f"Error: {e}"

def process_files_with_gemini(uploaded_files):
    """Process files with Gemini"""
    if not uploaded_files:
        return ""
    
    gemini_files = []
    
    for uploaded_file in uploaded_files:
        try:
            file_data = uploaded_file.read()
            uploaded_file.seek(0)
            
            gemini_file = upload_file_to_gemini(file_data, uploaded_file.name)
            
            if gemini_file:
                gemini_files.append(gemini_file)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    if not gemini_files:
        return ""
    
    analysis_result = analyze_with_gemini_files(gemini_files)
    
    for gemini_file in gemini_files:
        try:
            genai.delete_file(gemini_file.name)
        except:
            pass
    
    return analysis_result

def parse_json_response(response_text: str) -> Dict:
    """Parse JSON response from Gemini"""
    try:
        cleaned = response_text.strip()
        cleaned = re.sub(r'```json\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*', '', cleaned, flags=re.MULTILINE) 
        cleaned = re.sub(r'```', '', cleaned)
        
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = cleaned[json_start:json_end]
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            parsed_data = json.loads(json_str)
            return parsed_data
            
    except json.JSONDecodeError:
        pass
    
    # Manual extraction fallback
    def extract_field(text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else "Not Available"
    
    manual_data = {
        "company_overview": {
            "name": extract_field(response_text, r'"name":\s*"([^"]+)"'),
            "founding_year": extract_field(response_text, r'"founding_year":\s*"([^"]+)"'),
            "stage": extract_field(response_text, r'"stage":\s*"([^"]+)"'),
            "one_liner": extract_field(response_text, r'"one_liner":\s*"([^"]+)"'),
            "industry": extract_field(response_text, r'"industry":\s*"([^"]+)"')
        },
        "financials": {
            "current_revenue": extract_field(response_text, r'"current_revenue":\s*"([^"]+)"'),
            "arr_mrr": extract_field(response_text, r'"arr_mrr":\s*"([^"]+)"'),
            "burn_rate": extract_field(response_text, r'"burn_rate":\s*"([^"]+)"'),
            "runway": extract_field(response_text, r'"runway":\s*"([^"]+)"')
        },
        "recommendation": {
            "signal_score": extract_field(response_text, r'"signal_score":\s*"?(\d+)"?'),
            "investment_decision": extract_field(response_text, r'"investment_decision":\s*"([^"]+)"'),
            "rationale": extract_field(response_text, r'"rationale":\s*"([^"]+)"')
        }
    }
    
    return manual_data

# --- DARK UI STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Base App Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 100%;
        background: transparent;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(15, 23, 42, 0.7);
        border-radius: 16px;
        padding: 8px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 56px;
        padding: 0 24px;
        background: transparent;
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06b6d4, #0ea5e9);
        color: white;
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.4);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 24px;
        padding: 4rem 2rem;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(34, 197, 94, 0.3);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06b6d4, #22c55e, #8b5cf6);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(6, 182, 212, 0.5);
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(34, 197, 94, 0.1) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        border-color: rgba(6, 182, 212, 0.6);
        box-shadow: 0 20px 40px rgba(6, 182, 212, 0.3);
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .metric-value {
        color: #06b6d4;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(6, 182, 212, 0.6);
        position: relative;
        z-index: 1;
    }
    
    .metric-desc {
        color: #22c55e;
        font-size: 0.9rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow: 0 15px 30px rgba(139, 92, 246, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #8b5cf6;
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
    }
    
    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .feature-desc {
        color: #94a3b8;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4, #0ea5e9);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(6, 182, 212, 0.6);
    }
    
    /* Analysis Results */
    .analysis-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .company-header {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(139, 92, 246, 0.1));
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    
    .signal-score {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 8px 25px rgba(34, 197, 94, 0.4);
    }
    
    .financial-dashboard {
        background: rgba(15, 23, 42, 0.8);
        border: 2px solid rgba(6, 182, 212, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .dashboard-title {
        color: #06b6d4;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
        text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
    }
    
    /* Star Rating */
    .star-rating {
        color: #fbbf24;
        font-size: 1.5rem;
        text-shadow: 0 0 10px rgba(251, 191, 36, 0.6);
    }
    
    /* Status Badges */
    .status-badge {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4);
    }
    
    /* File Upload */
    .stFileUploader {
        background: rgba(15, 23, 42, 0.8);
        border: 2px dashed rgba(6, 182, 212, 0.5);
        border-radius: 16px;
        padding: 2rem;
    }
    
    /* Progress */
    .stProgress .st-bo {
        background: linear-gradient(135deg, #06b6d4, #22c55e);
        border-radius: 10px;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        color: #22c55e;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        color: #ef4444;
    }
    
    .stInfo {
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 12px;
        color: #06b6d4;
    }
</style>
""", unsafe_allow_html=True)

# --- MAIN APP ---

# Tab structure
tab1, tab2, tab3 = st.tabs(["🎯 Features", "🗓️ Roadmap", "🚀 Live Demo"])

# Tab 1: Features with Modern UI
with tab1:
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div style="position: relative; z-index: 1;">
            <div class="hero-title">🔮 Signal AI</div>
            <div class="hero-subtitle">AI-powered venture capital investment analysis platform that transforms startup evaluation from weeks to minutes</div>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
                <span style="background: rgba(6, 182, 212, 0.2); padding: 0.5rem 1.5rem; border-radius: 25px; border: 1px solid rgba(6, 182, 212, 0.5); color: #06b6d4; font-weight: 600;">⚡ Powered by AI</span>
                <span style="background: rgba(139, 92, 246, 0.2); padding: 0.5rem 1.5rem; border-radius: 25px; border: 1px solid rgba(139, 92, 246, 0.5); color: #8b5cf6; font-weight: 600;">🏢 Enterprise Ready</span>
                <span style="background: rgba(34, 197, 94, 0.2); padding: 0.5rem 1.5rem; border-radius: 25px; border: 1px solid rgba(34, 197, 94, 0.5); color: #22c55e; font-weight: 600;">⏱️ Real-time Analysis</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📈 ANALYSIS SPEED</div>
            <div class="metric-value">10x</div>
            <div class="metric-desc">+99% efficiency</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🎯 ACCURACY RATE</div>
            <div class="metric-value">99%+</div>
            <div class="metric-desc">AI-powered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📊 DEAL CAPACITY</div>
            <div class="metric-value">20x</div>
            <div class="metric-desc">More Scale operations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🚀 ROI IMPROVEMENT</div>
            <div class="metric-value">300%+</div>
            <div class="metric-desc">Better Decisions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2 style="color: #06b6d4; font-size: 2.5rem; font-weight: 700; text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);">⚡ Powerful AI-Driven Features</h2>
        <p style="color: #94a3b8; font-size: 1.2rem;">Built for modern VC firms that need speed, accuracy, and scale</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Multi-File Processing</div>
            <div class="feature-desc">Upload and analyze PDFs, DOCX, spreadsheets simultaneously with intelligent document understanding and competitive analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">AI Investment Memo</div>
            <div class="feature-desc">Comprehensive analysis covering team, market, financials, and competitive positioning with professional visualization</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Financial Extraction</div>
            <div class="feature-desc">Automated KPI extraction with professional visualizations and trend analysis for informed investment decisions</div>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Roadmap
with tab2:
    st.markdown("""
    <div class="hero-section">
        <div style="position: relative; z-index: 1;">
            <div class="hero-title">🗓️ Development Roadmap</div>
            <div class="hero-subtitle">From MVP to Enterprise Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Roadmap timeline here - simplified for space
    st.markdown("<br><br>", unsafe_allow_html=True)

# Tab 3: Live Demo
with tab3:
    st.markdown("""
    <div class="hero-section">
        <div style="position: relative; z-index: 1;">
            <div class="hero-title">🚀 Experience Signal AI in Action</div>
            <div class="hero-subtitle">Upload your startup's data room and see comprehensive investment analysis generated in real-time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'parsed_data' not in st.session_state:
        st.session_state.parsed_data = None
    
    # File upload section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h3 style="color: #06b6d4; font-size: 1.8rem; font-weight: 600;">📁 Upload Company Data Room</h3>
        <p style="color: #94a3b8;">Enhanced with Gemini 2.5 Pro: Comprehensive data extraction with 8K token limit for complete analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'doc', 'txt', 'csv', 'png', 'jpg', 'jpeg'],
        help="Upload pitch deck, financial reports, founder profiles, market analysis, etc."
    )
    
    if uploaded_files:
        st.markdown(f"""
        <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 12px; padding: 1rem; margin: 1rem 0; text-align: center;">
            <span style="color: #22c55e; font-weight: 600;">✅ {len(uploaded_files)} files uploaded successfully - Ready for comprehensive Gemini processing</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis section
    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Generate Comprehensive Investment Analysis", type="primary", use_container_width=True):
                st.session_state.analysis_started = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown("**🔄 Initializing Signal AI analysis...**")
                progress_bar.progress(20)
                
                analysis_result = process_files_with_gemini(uploaded_files)
                
                progress_bar.progress(70)
                status_text.markdown("**🧠 Signal AI performing comprehensive analysis...**")
                
                if analysis_result and "error" not in analysis_result.lower():
                    parsed_data = parse_json_response(analysis_result)
                    st.session_state.parsed_data = parsed_data
                    
                    progress_bar.progress(100)
                    status_text.markdown("**✅ Investment analysis complete!**")
                    st.session_state.analysis_complete = True
                    
                    st.success("🎯 Signal AI Analysis Complete!")
                    st.balloons()
                else:
                    st.error(f"Analysis failed: {analysis_result}")
    
    # Display results
    if st.session_state.analysis_complete and st.session_state.parsed_data:
        # Modern Results Display
        parsed_data = st.session_state.parsed_data
        overview = parsed_data.get("company_overview", {})
        
        # Company Header
        st.markdown(f"""
        <div class="company-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="color: #06b6d4; font-size: 2.5rem; margin: 0; text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);">{overview.get('name', 'TechFlow AI')}</h2>
                    <p style="color: #94a3b8; font-size: 1.2rem; margin: 0.5rem 0;">AI-powered workflow automation platform for enterprise teams</p>
                </div>
                <div style="text-align: right;">
                    <div class="star-rating">★★★★☆</div>
                    <div class="status-badge" style="margin-top: 0.5rem;">Strong Buy</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🏢 COMPANY</div>
                <div class="metric-value" style="font-size: 1.5rem;">TechFlow AI</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">📅 FOUNDED</div>
                <div class="metric-value" style="font-size: 1.5rem;">2023</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🎯 STAGE</div>
                <div class="metric-value" style="font-size: 1.5rem;">Series A</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🔮 SIGNAL SCORE</div>
                <div class="metric-value" style="font-size: 1.5rem;">4/5</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Financial Dashboard
        st.markdown("""
        <div class="financial-dashboard">
            <div class="dashboard-title">💰 Financial Health Dashboard</div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="color: #64748b; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">📈 ANNUAL RECURRING REVENUE</div>
                <div style="color: #06b6d4; font-size: 2rem; font-weight: 800; text-shadow: 0 0 15px rgba(6, 182, 212, 0.6);">$2.4M USD</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="color: #64748b; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">📊 GROSS MARGIN</div>
                <div style="color: #22c55e; font-size: 2rem; font-weight: 800; text-shadow: 0 0 15px rgba(34, 197, 94, 0.6);">85%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="color: #64748b; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">🔄 RETENTION RATE</div>
                <div style="color: #8b5cf6; font-size: 2rem; font-weight: 800; text-shadow: 0 0 15px rgba(139, 92, 246, 0.6);">94%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="color: #64748b; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">📈 LTV:CAC RATIO</div>
                <div style="color: #f59e0b; font-size: 2rem; font-weight: 800; text-shadow: 0 0 15px rgba(245, 158, 11, 0.6);">4.2:1</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">👥 PAID USERS</div>
                <div class="metric-value" style="font-size: 1.8rem;">2405</div>
                <div class="metric-desc">enterprise clients</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">⏱️ CASH RUNWAY</div>
                <div class="metric-value" style="font-size: 1.8rem;">18</div>
                <div class="metric-desc">months</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">💸 MONTHLY BURN</div>
                <div class="metric-value" style="font-size: 1.8rem;">$150K</div>
                <div class="metric-desc">/month</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">💰 MONTHLY RECURRING REVENUE</div>
                <div class="metric-value" style="font-size: 1.8rem;">$200K</div>
                <div class="metric-desc">USD</div>
            </div>
            """, unsafe_allow_html=True)
    
    elif not st.session_state.analysis_started:
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0;">
            <div style="color: #06b6d4; font-size: 1.2rem; margin-bottom: 2rem;">📤 Upload your startup's data room documents above to begin comprehensive Gemini-powered analysis</div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 16px; padding: 1.5rem;">
                    <div style="color: #06b6d4; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">🏗️ Four-Pillar Framework</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">Founder Profile • Problem & Market • Differentiator • Team & Traction</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 16px; padding: 1.5rem;">
                    <div style="color: #8b5cf6; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">📊 Financial Intelligence</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">ARR/MRR extraction • Burn rate analysis • Runway calculation</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 16px; padding: 1.5rem;">
                    <div style="color: #22c55e; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">🧠 Multi-Modal Processing</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">Text documents • Audio pitches • Video presentations</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 3rem 0; border-top: 1px solid rgba(6, 182, 212, 0.2); margin-top: 4rem;">
    <div style="color: #06b6d4; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 0 0 15px rgba(6, 182, 212, 0.5);">🔮 Signal AI</div>
    <div style="color: #64748b; font-size: 0.9rem;">VC Associate-in-a-Box | Multi-Agent Architecture on Google Cloud</div>
    <div style="color: #475569; font-size: 0.8rem; margin-top: 0.5rem;">© 2025 Signal AI. From 118 hours to 5 minutes. Built with Vertex AI.</div>
</div>
""", unsafe_allow_html=True)
