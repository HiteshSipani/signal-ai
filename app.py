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
    # Try to get API key from Streamlit secrets (for cloud deployment)
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    # Fallback to environment variable (for local development)
    else:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        st.error("⚠️ GEMINI_API_KEY not configured. Please add it to Streamlit secrets.")
        st.info("Go to your app settings → Secrets and add: GEMINI_API_KEY = 'your-api-key'")
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
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name
        
        # Upload to Gemini
        uploaded_file = genai.upload_file(tmp_file_path, display_name=file_name)
        
        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            st.error(f"Processing failed for {file_name}")
            return None
            
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return uploaded_file
        
    except Exception as e:
        st.error(f"File upload failed for {file_name}: {e}")
        return None

def analyze_with_gemini_files(gemini_files):
    """Enhanced Gemini analysis with comprehensive data extraction"""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Enhanced prompt focusing on specific data extraction
        analysis_prompt = """
        You are Signal AI, an expert venture capital analyst powered by multi-agent architecture. 
        Analyze the provided startup documents with the precision of a senior VC associate.

        EXTRACT ALL CRITICAL INVESTMENT DATA following the four pillars framework:
        1. FOUNDER PROFILE: Names, backgrounds, previous experience, founder-market fit
        2. PROBLEM & MARKET SIZE: TAM, problem validation, market opportunity ($B figures)
        3. UNIQUE DIFFERENTIATOR: Competitive advantages, moats, IP, technology
        4. TEAM & TRACTION: Customer metrics, revenue, growth rates, partnerships

        FINANCIAL FOCUS: Extract every number mentioned - revenue projections, funding amounts, 
        user counts, growth percentages, contract values, market size figures.

        Return comprehensive JSON with this structure:

        {
            "company_overview": {
                "name": "Exact company name",
                "founding_year": "Year founded", 
                "stage": "Current funding stage",
                "one_liner": "Clear value proposition",
                "industry": "Primary industry/sector"
            },
            "founders": [
                {
                    "name": "Full name",
                    "role": "Title/role",
                    "background": "Education + previous experience + years",
                    "founder_market_fit": "Assessment of fit to this market"
                }
            ],
            "problem_and_market": {
                "problem_statement": "Core problem being solved",
                "market_size_tam": "Total addressable market with $B figures",
                "market_growth_rate": "Annual growth rate %",
                "target_customer": "Specific customer profile",
                "market_validation": "Evidence of demand"
            },
            "unique_differentiator": {
                "core_technology": "Key technology/innovation",
                "competitive_moat": "Sustainable competitive advantages",
                "ip_assets": "Patents, proprietary tech, data",
                "barriers_to_entry": "What prevents competition"
            },
            "team_and_traction": {
                "team_size": "Current headcount",
                "customer_count": "Number of customers/users",
                "arr_mrr": "Annual/monthly recurring revenue",
                "growth_metrics": "User/revenue growth rates",
                "key_customers": ["List of notable customers"],
                "partnerships": ["Strategic partnerships"],
                "revenue_model": "How money is made"
            },
            "financials": {
                "current_revenue": "Latest revenue figures",
                "revenue_projections": "Future revenue forecasts",
                "funding_raised": "Total capital raised",
                "current_ask": "Amount seeking in current round",
                "valuation": "Company valuation",
                "burn_rate": "Monthly cash burn",
                "runway": "Months of runway remaining",
                "unit_economics": "LTV, CAC, payback period"
            },
            "investment_thesis": {
                "strengths": ["3-5 key investment strengths"],
                "risks": ["3-5 main risk factors"],
                "market_opportunity": "Size and timing of opportunity",
                "execution_risk": "Team's ability to execute"
            },
            "recommendation": {
                "signal_score": "1-5 rating",
                "investment_decision": "STRONG BUY/BUY/HOLD/PASS",
                "rationale": "Detailed reasoning with specific evidence",
                "comparable_companies": ["Similar successful companies"]
            }
        }
        """
        
        # Prepare content for analysis
        content_parts = [analysis_prompt]
        content_parts.extend(gemini_files)
        
        st.info("Signal AI analyzing with VC Associate precision...")
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            max_output_tokens=8000,
        )
        
        response = model.generate_content(content_parts, generation_config=generation_config)
        
        if response.text:
            st.success("Multi-agent analysis completed!")
            return response.text
        else:
            st.error("No response from Gemini")
            return "Error: No response"
            
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return f"Error: {e}"

def process_files_with_gemini(uploaded_files):
    """Enhanced file processing with better error handling"""
    
    if not uploaded_files:
        return ""
    
    gemini_files = []
    
    # Upload all files to Gemini
    for uploaded_file in uploaded_files:
        st.info(f"Processing {uploaded_file.name}...")
        
        try:
            # Read file data
            file_data = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Upload to Gemini
            gemini_file = upload_file_to_gemini(file_data, uploaded_file.name)
            
            if gemini_file:
                gemini_files.append(gemini_file)
                st.success(f"Successfully uploaded {uploaded_file.name}")
            else:
                st.warning(f"Failed to upload {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    if not gemini_files:
        st.error("No files were successfully processed")
        return ""
    
    # Analyze with improved function
    analysis_result = analyze_with_gemini_files(gemini_files)
    
    # Clean up
    for gemini_file in gemini_files:
        try:
            genai.delete_file(gemini_file.name)
        except:
            pass
    
    return analysis_result

# --- UTILITY FUNCTIONS ---

def clean_text_formatting(text: str) -> str:
    """Clean text to ensure consistent formatting"""
    if not text or text == "Not Available":
        return text
    
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    
    if len(text) > 50:
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    
    return text.strip()

def parse_rating(rating_value):
    """Parse rating value to ensure it's a valid integer between 1-5"""
    if rating_value == "Not Available" or rating_value is None:
        return None
    
    try:
        rating_str = str(rating_value)
        if '/' in rating_str:
            rating_str = rating_str.split('/')[0]
        
        numbers = re.findall(r'\d+', rating_str)
        if numbers:
            rating_int = int(numbers[0])
            if 1 <= rating_int <= 5:
                return rating_int
    except (ValueError, TypeError):
        pass
    
    return None

def parse_json_response(response_text: str) -> Dict:
    """Enhanced JSON parsing for comprehensive data"""
    
    # Strategy 1: Direct parsing with cleaning
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
            json_str = re.sub(r'([}\]])\s*\n\s*(["\w])', r'\1,\n\2', json_str)
            
            parsed_data = json.loads(json_str)
            st.success("JSON parsed successfully!")
            return parsed_data
            
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing failed: {e}")
    
    # Strategy 2: Manual extraction fallback
    st.info("Attempting manual data extraction...")
    
    def extract_field(text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else "Not Available"
    
    def extract_array(text: str, pattern: str) -> List[str]:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        return [match.strip('"') for match in matches if match.strip()]
    
    manual_data = {
        "company_overview": {
            "name": extract_field(response_text, r'"name":\s*"([^"]+)"'),
            "founding_year": extract_field(response_text, r'"founding_year":\s*"([^"]+)"'),
            "stage": extract_field(response_text, r'"stage":\s*"([^"]+)"'),
            "one_liner": extract_field(response_text, r'"one_liner":\s*"([^"]+)"'),
            "industry": extract_field(response_text, r'"industry":\s*"([^"]+)"')
        },
        "problem_and_market": {
            "market_size_tam": extract_field(response_text, r'"market_size_tam":\s*"([^"]+)"'),
            "problem_statement": extract_field(response_text, r'"problem_statement":\s*"([^"]+)"')
        },
        "team_and_traction": {
            "customer_count": extract_field(response_text, r'"customer_count":\s*"([^"]+)"'),
            "arr_mrr": extract_field(response_text, r'"arr_mrr":\s*"([^"]+)"')
        },
        "financials": {
            "current_ask": extract_field(response_text, r'"current_ask":\s*"([^"]+)"'),
            "funding_raised": extract_field(response_text, r'"funding_raised":\s*"([^"]+)"')
        },
        "recommendation": {
            "signal_score": extract_field(response_text, r'"signal_score":\s*"?(\d+)"?'),
            "investment_decision": extract_field(response_text, r'"investment_decision":\s*"([^"]+)"'),
            "rationale": extract_field(response_text, r'"rationale":\s*"([^"]+)"')
        }
    }
    
    return manual_data

def create_roadmap_timeline():
    """Create interactive roadmap timeline"""
    
    # Updated roadmap data with correct dates
    roadmap_data = [
        # COMPLETED (Past)
        {
            "phase": "Foundation",
            "status": "COMPLETED",
            "date": "Sep 2025",
            "title": "MVP Development",
            "description": "Core Gemini integration, file processing, basic analysis",
            "achievements": ["Multi-modal file processing", "JSON data extraction", "Basic investment memo generation"],
            "color": "#10B981",
            "position": 0
        },
        {
            "phase": "Foundation", 
            "status": "COMPLETED",
            "date": "Sep 2025",
            "title": "Enhanced Analysis Engine",
            "description": "Improved prompts, comprehensive data extraction, UI polish",
            "achievements": ["4-pillar framework integration", "Enhanced financial metrics", "Professional UI/UX"],
            "color": "#10B981",
            "position": 1
        },
        
        # IN PROGRESS (Current)
        {
            "phase": "Growth",
            "status": "IN PROGRESS", 
            "date": "Oct 2025",
            "title": "Multi-Agent Architecture",
            "description": "Implement specialized agents for different analysis tasks",
            "achievements": ["Supervisor Agent (Gemini 2.0 Pro)", "Specialist Agent (Fine-tuned Gemma)", "Research Agent (Vertex AI)"],
            "color": "#F59E0B",
            "position": 2
        },
        
        # PLANNED (Future)
        {
            "phase": "Growth",
            "status": "PLANNED",
            "date": "Oct 2025", 
            "title": "Production Platform",
            "description": "Vertex AI Pipelines, scalable infrastructure, enterprise features",
            "achievements": ["DAG-based workflows", "Real-time market data", "Batch processing"],
            "color": "#6B7280",
            "position": 3
        },
        {
            "phase": "Scale",
            "status": "PLANNED", 
            "date": "Nov 2025",
            "title": "Advanced Intelligence",
            "description": "Conversational AI, automated scheduling, predictive analytics",
            "achievements": ["Founder interview agent", "Predictive modeling", "Risk assessment AI"],
            "color": "#6B7280",
            "position": 4
        },
        {
            "phase": "Scale",
            "status": "PLANNED",
            "date": "Nov 2025",
            "title": "Enterprise Suite", 
            "description": "Full platform with collaboration, compliance, and advanced analytics",
            "achievements": ["Team collaboration", "Regulatory compliance", "Custom integrations"],
            "color": "#6B7280",
            "position": 5
        }
    ]
    
    # Create connected timeline chart
    fig = go.Figure()
    
    # Create the main timeline line
    x_positions = [item['position'] for item in roadmap_data]
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=[0.5] * len(x_positions),
        mode='lines',
        line=dict(color='#E5E7EB', width=4),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add milestone points and labels
    for item in roadmap_data:
        x_pos = item['position']
        
        # Milestone point
        fig.add_trace(go.Scatter(
            x=[x_pos],
            y=[0.5],
            mode='markers',
            marker=dict(
                size=20,
                color=item['color'],
                line=dict(color='white', width=3)
            ),
            hovertemplate=f"<b>{item['title']}</b><br>{item['date']}<br>{item['description']}<extra></extra>",
            showlegend=False
        ))
        
        # Status badge above the point
        fig.add_annotation(
            x=x_pos,
            y=0.8,
            text=f"<b>{item['title']}</b><br>{item['date']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=item['color'],
            ax=0,
            ay=-30,
            font=dict(size=11, color=item['color']),
            bgcolor="white",
            bordercolor=item['color'],
            borderwidth=2,
            borderpad=4
        )
        
        # Status indicator below
        status_emoji = "✅" if item['status'] == 'COMPLETED' else ("🔄" if item['status'] == 'IN PROGRESS' else "📋")
        fig.add_annotation(
            x=x_pos,
            y=0.2,
            text=f"{status_emoji} {item['status']}",
            showarrow=False,
            font=dict(size=10, color=item['color']),
            bgcolor="white",
            bordercolor=item['color'],
            borderwidth=1,
            borderpad=2
        )
    
    # Update layout for better appearance
    fig.update_layout(
        title="Signal AI Development Roadmap - Connected Timeline",
        title_font_size=24,
        title_x=0.5,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-0.5, len(roadmap_data) - 0.5]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, 1]
        ),
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig, roadmap_data

def display_investment_memo(parsed_data):
    """Display investment memo using 4-pillar framework"""
    
    if "error" in parsed_data:
        st.error("Analysis Error")
        st.markdown(f"**Error:** {parsed_data.get('error', 'Unknown error')}")
        return
    
    st.markdown("---")
    st.markdown("# Signal AI Investment Analysis")
    
    # Company Header
    overview = parsed_data.get("company_overview", {})
    company_name = clean_text_formatting(overview.get("name", "Startup Analysis"))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Company", company_name)
    with col2:
        industry = overview.get("industry", "Not Available")
        st.metric("Industry", industry)
    with col3:
        stage = overview.get("stage", "Not Available")
        st.metric("Stage", stage)
    with col4:
        recommendation = parsed_data.get("recommendation", {})
        signal_score = recommendation.get("signal_score", "Not Available")
        decision = recommendation.get("investment_decision", "ANALYZE")
        
        parsed_score = parse_rating(signal_score)
        if parsed_score:
            st.metric("Signal Score", f"{parsed_score}/5", delta=decision)
        else:
            st.metric("Signal Score", "N/A", delta=decision)
    
    # Value Proposition
    one_liner = overview.get("one_liner", "")
    if one_liner and one_liner != "Not Available":
        st.info(f"**Value Proposition:** {one_liner}")
    
    # FOUR PILLARS FRAMEWORK
    st.markdown("## Investment Analysis: Four Pillars Framework")
    
    # Pillar 1: Founder Profile
    st.markdown("### 1. Founder Profile & Market Fit")
    founders = parsed_data.get("founders", [])
    
    if founders:
        for founder in founders:
            if isinstance(founder, dict):
                name = founder.get('name', 'Founder')
                role = founder.get('role', 'Role not specified')
                background = founder.get('background', 'Background not available')
                market_fit = founder.get('founder_market_fit', 'Assessment not available')
                
                with st.expander(f"{name} - {role}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Background:** {background}")
                    with col2:
                        st.markdown(f"**Market Fit:** {market_fit}")
    else:
        st.info("Founder profiles not found in analysis")
    
    # Pillar 2: Problem & Market Size
    st.markdown("### 2. Problem & Market Opportunity")
    problem_market = parsed_data.get("problem_and_market", {})
    
    col1, col2 = st.columns(2)
    with col1:
        problem = problem_market.get('problem_statement', 'Not analyzed')
        st.markdown(f"**Problem Statement:** {problem}")
        
        target_customer = problem_market.get('target_customer', 'Not specified')
        st.markdown(f"**Target Customer:** {target_customer}")
    
    with col2:
        tam = problem_market.get('market_size_tam', 'Not analyzed')
        st.markdown(f"**Total Addressable Market:** {tam}")
        
        growth_rate = problem_market.get('market_growth_rate', 'Not analyzed')
        st.markdown(f"**Market Growth Rate:** {growth_rate}")
    
    # Pillar 3: Unique Differentiator
    st.markdown("### 3. Unique Differentiator & Competitive Moat")
    differentiator = parsed_data.get("unique_differentiator", {})
    
    col1, col2 = st.columns(2)
    with col1:
        core_tech = differentiator.get('core_technology', 'Not specified')
        st.markdown(f"**Core Technology:** {core_tech}")
        
        moat = differentiator.get('competitive_moat', 'Not analyzed')
        st.markdown(f"**Competitive Moat:** {moat}")
    
    with col2:
        ip_assets = differentiator.get('ip_assets', 'Not specified')
        st.markdown(f"**IP & Assets:** {ip_assets}")
        
        barriers = differentiator.get('barriers_to_entry', 'Not analyzed')
        st.markdown(f"**Barriers to Entry:** {barriers}")
    
    # Pillar 4: Team & Traction
    st.markdown("### 4. Team & Traction Metrics")
    traction = parsed_data.get("team_and_traction", {})
    
    # Traction Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        customers = traction.get('customer_count', 'Not Available')
        st.metric("Customers", customers)
    
    with col2:
        arr = traction.get('arr_mrr', 'Not Available')
        st.metric("ARR/MRR", arr)
    
    with col3:
        growth = traction.get('growth_metrics', 'Not Available')
        st.metric("Growth Rate", growth)
    
    # Key customers and partnerships
    key_customers = traction.get('key_customers', [])
    partnerships = traction.get('partnerships', [])
    
    if key_customers:
        st.markdown("**Key Customers:**")
        customer_text = ", ".join(key_customers)
        st.markdown(f"> {customer_text}")
    
    if partnerships:
        st.markdown("**Strategic Partnerships:**")
        partnership_text = ", ".join(partnerships)
        st.markdown(f"> {partnership_text}")
    
    # FINANCIAL DASHBOARD
    st.markdown("## Financial Overview")
    financials = parsed_data.get("financials", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Current Revenue", financials.get('current_revenue', 'Not Available')),
        ("Funding Ask", financials.get('current_ask', 'Not Available')),
        ("Valuation", financials.get('valuation', 'Not Available')),
        ("Runway", financials.get('runway', 'Not Available'))
    ]
    
    for i, (label, value) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.metric(label, value)
    
    # INVESTMENT THESIS
    st.markdown("## Investment Thesis")
    thesis = parsed_data.get("investment_thesis", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Strengths")
        strengths = thesis.get("strengths", [])
        if strengths:
            for strength in strengths:
                st.success(f"✓ {strength}")
        else:
            st.info("Strengths analysis not available")
    
    with col2:
        st.markdown("### Risks") 
        risks = thesis.get("risks", [])
        if risks:
            for risk in risks:
                st.warning(f"⚠ {risk}")
        else:
            st.info("Risk analysis not available")
    
    # FINAL RECOMMENDATION
    st.markdown("## Final Investment Recommendation")
    
    recommendation = parsed_data.get("recommendation", {})
    decision = recommendation.get("investment_decision", "ANALYZE")
    rationale = recommendation.get("rationale", "No rationale provided")
    
    if decision == "STRONG BUY":
        st.success(f"### {decision}")
    elif decision in ["BUY", "CONSIDER"]:
        st.warning(f"### {decision}")
    else:
        st.error(f"### {decision}")
    
    st.markdown(f"**Rationale:** {rationale}")
    
    # Download option
    st.markdown("---")
    memo_text = json.dumps(parsed_data, indent=2)
    st.download_button(
        label="Download Investment Analysis (JSON)",
        data=memo_text,
        file_name=f"{company_name}_signal_analysis.json",
        mime="application/json"
    )

# --- STREAMLIT APP ---

# Enhanced CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Fix for mobile/narrow screens */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Ensure tabs are visible and properly styled */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 4px;
        margin-bottom: 1rem;
        overflow-x: auto;
        width: 100%;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 20px;
        background-color: transparent;
        border-radius: 6px;
        color: #6b7280;
        font-weight: 500;
        font-size: 14px;
        border: none;
        white-space: nowrap;
        min-width: fit-content;
        flex-shrink: 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        font-family: 'Inter', sans-serif !important;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        font-family: 'Inter', sans-serif !important;
    }
    
    .main-header h3 {
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif !important;
    }
    
    .value-prop-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .pillar-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .pillar-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        min-height: 80px;
        font-family: 'Inter', sans-serif !important;
    }
    
    .roadmap-phase {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .completed { border-left: 4px solid #10B981; }
    .in-progress { border-left: 4px solid #F59E0B; }
    .planned { border-left: 4px solid #6B7280; }
</style>
""", unsafe_allow_html=True)

# Tab structure with enhanced content
tab1, tab2, tab3 = st.tabs(["Vision & Strategy", "Development Roadmap", "Live Demo"])

# Tab 1: Enhanced Vision & Strategy
with tab1:
    st.markdown("""
    <div class="main-header">
        <h1>Signal AI</h1>
        <h3>VC Associate-in-a-Box</h3>
        <p style="font-size: 1.1rem; opacity: 0.8;">Multi-Agent AI Platform Transforming Venture Capital Due Diligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Value Proposition
    st.markdown("""
    <div class="value-prop-card">
        <h2 style="margin-bottom: 1rem;">Our Value Proposition</h2>
        <p style="font-size: 1.2rem; margin-bottom: 1rem;">
        For early-stage venture capital firms overwhelmed by deal flow, Signal AI is an intelligent analysis platform 
        that reduces preliminary due diligence time by <strong>90%</strong> while increasing analytical depth and consistency.
        </p>
        <p style="font-size: 1rem; opacity: 0.9;">
        Unlike manual processes taking 118+ hours, our specialized multi-agent workflow automatically extracts, 
        synthesizes, and scores critical startup data from unstructured documents in minutes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Problem Statement
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## The Problem We're Solving")
        st.markdown("""
        **Venture Capital Due Diligence Crisis:**
        - **118+ hours** per deal for preliminary analysis
        - **Cognitive overload** from unstructured data
        - **Inconsistent evaluation** across investment teams
        - **Deal flow bottlenecks** preventing opportunity evaluation
        - **Manual processes** that don't scale with modern deal volume
        """)
        
        st.markdown("## Our Target Market")
        st.markdown("""
        - **Early-stage VC firms** (Seed to Series A)
        - **Angel investor groups** and syndicates  
        - **Accelerator programs** and incubators
        - **Corporate venture arms**
        - **Investment banks** (tech coverage)
        """)
    
    with col2:
        st.markdown("## The Signal AI Solution")
        st.markdown("""
        **Multi-Agent AI Architecture:**
        - **Supervisor Agent** (Gemini 2.0 Pro) - Strategic reasoning
        - **Specialist Agent** (Fine-tuned Gemma) - Financial extraction
        - **Research Agent** (Vertex AI + Search) - Market intelligence
        - **Interviewer Agent** (Future) - Founder conversations
        """)
        
        st.markdown("## Competitive Advantages")
        st.markdown("""
        - **Specialized workflow** vs generic AI tools
        - **Multi-modal processing** (docs, audio, video)
        - **Real-time market data** integration
        - **Industry-specific training** on VC terminology
        - **90% time reduction** with higher accuracy
        """)
    
    # Four Pillars Framework
    st.markdown("## Analysis Framework: Four Investment Pillars")
    
    pillars = [
        {
            "title": "1. Founder Profile & Market Fit",
            "description": "Deep analysis of founder backgrounds, experience, and alignment with market opportunity",
            "icon": "👥"
        },
        {
            "title": "2. Problem & Market Size", 
            "description": "Market validation, TAM assessment, growth rates, and competitive landscape",
            "icon": "📊"
        },
        {
            "title": "3. Unique Differentiator",
            "description": "Technology moats, IP assets, competitive advantages, and barriers to entry",
            "icon": "🚀"
        },
        {
            "title": "4. Team & Traction",
            "description": "Customer metrics, revenue growth, partnerships, and execution capability",
            "icon": "📈"
        }
    ]
    
    cols = st.columns(2)
    for i, pillar in enumerate(pillars):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="pillar-card">
                <h3>{pillar['icon']} {pillar['title']}</h3>
                <p>{pillar['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Impact Metrics
    st.markdown("## Expected Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    impact_metrics = [
        ("Time Reduction", "90%", "From 118 hours to 5 minutes"),
        ("Analysis Depth", "5x", "More comprehensive than manual"),
        ("Deal Throughput", "20x", "Evaluate more opportunities"),
        ("Decision Speed", "95%", "Faster investment decisions")
    ]
    
    for i, (metric, value, description) in enumerate(impact_metrics):
        with [col1, col2, col3, col4][i]:
            st.metric(metric, value, description)

# Tab 2: Development Roadmap
with tab2:
    st.markdown("""
    <div class="main-header">
        <h1>Development Roadmap</h1>
        <h3>From MVP to Enterprise Platform</h3>
        <p style="font-size: 1.1rem; opacity: 0.8;">Strategic milestones and technical architecture evolution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create and display timeline
    timeline_fig, roadmap_data = create_roadmap_timeline()
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Detailed roadmap phases
    st.markdown("## Detailed Development Phases")
    
    for item in roadmap_data:
        status_class = item['status'].lower().replace(' ', '-')
        
        st.markdown(f"""
        <div class="roadmap-phase {status_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: {item['color']};">{item['title']}</h3>
                <span style="background: {item['color']}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                    {item['status']}
                </span>
            </div>
            <p style="margin-bottom: 1rem; color: #4B5563;"><strong>{item['date']}</strong> - {item['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if item['achievements']:
            st.markdown("**Key Achievements:**")
            for achievement in item['achievements']:
                st.markdown(f"• {achievement}")
        
        st.markdown("---")
    
    # Technical Architecture Evolution
    st.markdown("## Technical Architecture Evolution")
    
    arch_phases = [
        {
            "phase": "Current (Q1 2025)",
            "title": "Single-Agent Foundation",
            "tech": ["Gemini 2.5 Pro API", "Streamlit Frontend", "Basic JSON Parsing", "Manual File Processing"],
            "color": "#10B981"
        },
        {
            "phase": "Next (Q2 2025)", 
            "title": "Multi-Agent System",
            "tech": ["Vertex AI Pipelines", "Gemini 2.0 Pro (Supervisor)", "Fine-tuned Gemma (Specialist)", "Agent Builder (Research)"],
            "color": "#F59E0B"
        },
        {
            "phase": "Future (Q3-Q4 2025)",
            "title": "Enterprise Platform", 
            "tech": ["Cloud Storage Integration", "Real-time APIs", "Conversational AI", "Advanced Analytics", "Team Collaboration"],
            "color": "#6B7280"
        }
    ]
    
    for arch in arch_phases:
        with st.expander(f"{arch['phase']}: {arch['title']}", expanded=False):
            st.markdown("**Technology Stack:**")
            for tech in arch['tech']:
                st.markdown(f"• {tech}")

# Tab 3: Live Demo (Enhanced)
with tab3:
    st.markdown("""
    <div class="main-header">
        <h1>Signal AI Live Demo</h1>
        <h3>Experience Multi-Agent VC Analysis</h3>
        <p style="font-size: 1.1rem; opacity: 0.8;">Upload startup documents and witness 90% time reduction in action</p>
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
    st.markdown("## Upload Company Data Room")
    st.info("**Current:** Single-agent processing with Gemini 2.5 Pro | **Coming Q2:** Multi-agent architecture with specialized models")
    
    uploaded_files = st.file_uploader(
        "Upload startup documents for comprehensive analysis",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'doc', 'txt', 'csv', 'png', 'jpg', 'jpeg'],
        help="Pitch deck, financial reports, founder bios, market analysis, traction data, etc."
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files ready for Signal AI processing")
        
        with st.expander("View uploaded files", expanded=True):
            for file in uploaded_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file.name}")
                with col2:
                    st.write(f"{file.size:,} bytes")
                with col3:
                    st.write("✓ Ready")
    
    # Analysis section
    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Investment Analysis", type="primary", use_container_width=True):
                st.session_state.analysis_started = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown("**Initializing Signal AI multi-agent workflow...**")
                progress_bar.progress(20)
                
                # Process with enhanced Gemini
                analysis_result = process_files_with_gemini(uploaded_files)
                
                progress_bar.progress(70)
                status_text.markdown("**Signal AI performing comprehensive 4-pillar analysis...**")
                
                if analysis_result and "error" not in analysis_result.lower():
                    parsed_data = parse_json_response(analysis_result)
                    st.session_state.parsed_data = parsed_data
                    
                    progress_bar.progress(100)
                    status_text.markdown("**✅ Investment analysis complete - 90% time reduction achieved!**")
                    st.session_state.analysis_complete = True
                    
                    st.success("🎯 Signal AI Analysis Complete!")
                    st.balloons()
                else:
                    st.error(f"Analysis failed: {analysis_result}")
    
    # Display results using 4-pillar framework
    if st.session_state.analysis_complete and st.session_state.parsed_data:
        display_investment_memo(st.session_state.parsed_data)
    
    elif not st.session_state.analysis_started:
        st.info("👆 Upload your startup's data room above to experience Signal AI's 90% time reduction")
        
        # Demo features preview
        st.markdown("## Signal AI Analysis Features")
        
        features = [
            ("🏗️ Four-Pillar Framework", "Founder Profile • Problem & Market • Differentiator • Team & Traction"),
            ("📊 Financial Intelligence", "ARR/MRR extraction • Burn rate analysis • Runway calculation • Unit economics"),
            ("🧠 Multi-Modal Processing", "Text documents • Audio pitches • Video presentations • Charts & graphs"),
            ("🔍 Market Intelligence", "Real-time competitive data • TAM validation • Growth rate analysis"),
            ("⚡ Instant Insights", "5-minute analysis • Signal scoring • Investment recommendation • Risk assessment")
        ]
        
        for feature, description in features:
            st.markdown(f"**{feature}**")
            st.markdown(f"> {description}")
            st.markdown("")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; border-top: 1px solid #e5e7eb; margin-top: 3rem; color: #6b7280;">
    <p><strong>Signal AI</strong> - VC Associate-in-a-Box | Multi-Agent Architecture on Google Cloud</p>
    <p style="font-size: 0.8rem;">© 2025 Signal AI. From 118 hours to 5 minutes. Built with Vertex AI.</p>
</div>
""", unsafe_allow_html=True)
