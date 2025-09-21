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
        You are Signal AI, an expert venture capital analyst powered by multi-agent architecture. 
        Analyze the provided startup documents with the precision of a senior VC associate.

        EXTRACT ALL CRITICAL INVESTMENT DATA following the four pillars framework:
        1. FOUNDER PROFILE: Names, backgrounds, previous experience, founder-market fit
        2. PROBLEM & MARKET SIZE: TAM, problem validation, market opportunity ($B figures)
        3. UNIQUE DIFFERENTIATOR: Competitive advantages, moats, IP, technology
        4. TEAM & TRACTION: Customer metrics, revenue, growth rates, partnerships

        CRITICAL FINANCIAL METRICS TO EXTRACT (if mentioned):
        - Annual Recurring Revenue (ARR) and Monthly Recurring Revenue (MRR)
        - Customer Acquisition Cost (CAC) and Lifetime Value (LTV)
        - Gross margin, net margin, and unit economics
        - Monthly/annual burn rate and cash runway
        - Customer retention rate and churn rate
        - Revenue growth rate (monthly/yearly)
        - Funding amounts, valuation, and use of funds
        - Customer count, average deal size, and contract values
        - EBITDA, gross revenue, and profitability metrics

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
                "arr_mrr": "Annual/monthly recurring revenue with specific amounts",
                "growth_metrics": "User/revenue growth rates with percentages",
                "key_customers": ["List of notable customers"],
                "partnerships": ["Strategic partnerships"],
                "revenue_model": "How money is made"
            },
            "financials": {
                "current_revenue": "Latest revenue figures (ARR/MRR)",
                "revenue_projections": "Future revenue forecasts",
                "funding_raised": "Total capital raised",
                "current_ask": "Amount seeking in current round",
                "valuation": "Company valuation",
                "burn_rate": "Monthly cash burn rate",
                "runway": "Months of runway remaining",
                "unit_economics": "LTV, CAC, payback period, gross margin %",
                "retention_rate": "Customer retention rate %",
                "growth_rate": "Revenue growth rate % (monthly/yearly)",
                "cac_ltv_ratio": "LTV:CAC ratio (e.g., 3:1)",
                "gross_margin": "Gross margin percentage",
                "churn_rate": "Customer churn rate %"
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
    
    for uploaded_file in uploaded_files:
        st.info(f"Processing {uploaded_file.name}...")
        
        try:
            file_data = uploaded_file.read()
            uploaded_file.seek(0)
            
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
    
    analysis_result = analyze_with_gemini_files(gemini_files)
    
    for gemini_file in gemini_files:
        try:
            genai.delete_file(gemini_file.name)
        except:
            pass
    
    return analysis_result

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
            st.success("JSON parsed successfully!")
            return parsed_data
            
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing failed: {e}")
    
    # Manual extraction fallback
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
    """Display investment memo using 4-pillar framework - ALL DATA FROM UPLOADED FILES"""
    
    if "error" in parsed_data:
        st.error("Analysis Error")
        st.markdown(f"**Error:** {parsed_data.get('error', 'Unknown error')}")
        return
    
    st.markdown("---")
    st.markdown("# Signal AI Investment Analysis")
    
    # Company Header - EXTRACTED DATA ONLY
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
    
    # Value Proposition - EXTRACTED DATA ONLY
    one_liner = overview.get("one_liner", "")
    if one_liner and one_liner != "Not Available":
        st.info(f"**Value Proposition:** {one_liner}")
    
    # FINANCIAL DASHBOARD - ALL EXTRACTED DATA
    st.markdown("## Financial Health Dashboard")
    financials = parsed_data.get("financials", {})
    
    # Create metric cards with EXTRACTED data only and explanations
    financial_metrics = [
        ("📈 Current Revenue", financials.get('current_revenue', 'Not Available'), "Total revenue generated by the company in the most recent period"),
        ("💰 ARR/MRR", financials.get('arr_mrr', 'Not Available'), "Annual Recurring Revenue / Monthly Recurring Revenue - predictable revenue from subscriptions or contracts"),
        ("🔥 Burn Rate", financials.get('burn_rate', 'Not Available'), "Monthly cash consumption rate - how much money the company spends per month"),
        ("⏱️ Cash Runway", financials.get('runway', 'Not Available'), "Number of months the company can operate with current cash before running out"),
        ("📊 Gross Margin", financials.get('gross_margin', 'Not Available'), "Percentage of revenue remaining after subtracting cost of goods sold - indicates pricing power"),
        ("🔄 Retention Rate", financials.get('retention_rate', 'Not Available'), "Percentage of customers who continue using the service over time - indicates product stickiness"),
        ("📈 Growth Rate", financials.get('growth_rate', 'Not Available'), "Rate at which revenue or users are increasing - typically measured monthly or yearly"),
        ("🎯 LTV:CAC Ratio", financials.get('cac_ltv_ratio', 'Not Available'), "Lifetime Value to Customer Acquisition Cost ratio - how much value each customer generates vs cost to acquire")
    ]
    
    # Display financial metrics in rows of 4 with info tooltips
    for i in range(0, len(financial_metrics), 4):
        cols = st.columns(4)
        for j, (label, value, explanation) in enumerate(financial_metrics[i:i+4]):
            if j < len(cols):
                with cols[j]:
                    if value != "Not Available":
                        st.metric(label, value, help=explanation)
                    else:
                        st.metric(label, "N/A", help=explanation)
    
    # FOUR PILLARS FRAMEWORK - ALL EXTRACTED DATA
    st.markdown("## Investment Analysis: Four Pillars Framework")
    
    # Pillar 1: Founder Profile - EXTRACTED DATA ONLY
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
        st.info("Founder profiles not found in uploaded documents")
    
    # Pillar 2: Problem & Market Size - EXTRACTED DATA ONLY
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
    
    # Pillar 3: Unique Differentiator - EXTRACTED DATA ONLY
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
    
    # Pillar 4: Team & Traction - EXTRACTED DATA ONLY
    st.markdown("### 4. Team & Traction Metrics")
    traction = parsed_data.get("team_and_traction", {})
    
    # Traction Metrics - EXTRACTED DATA ONLY
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
    
    # Key customers and partnerships - EXTRACTED DATA ONLY
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
    
    # ADDITIONAL FINANCIAL OVERVIEW - EXTRACTED DATA ONLY
    st.markdown("## Funding & Valuation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    funding_metrics = [
        ("Current Ask", financials.get('current_ask', 'Not Available'), "Amount of funding the company is currently seeking"),
        ("Total Raised", financials.get('funding_raised', 'Not Available'), "Total capital raised to date from all previous funding rounds"),
        ("Valuation", financials.get('valuation', 'Not Available'), "Current estimated worth of the company - pre-money or post-money valuation"),
        ("Unit Economics", financials.get('unit_economics', 'Not Available'), "Financial metrics per unit/customer - includes CAC, LTV, payback period")
    ]
    
    for i, (label, value, explanation) in enumerate(funding_metrics):
        with [col1, col2, col3, col4][i]:
            st.metric(label, value, help=explanation)
    
    # INVESTMENT THESIS - EXTRACTED DATA ONLY
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
            st.info("Strengths analysis not available in uploaded documents")
    
    with col2:
        st.markdown("### Risks") 
        risks = thesis.get("risks", [])
        if risks:
            for risk in risks:
                st.warning(f"⚠ {risk}")
        else:
            st.info("Risk analysis not available in uploaded documents")
    
    # FINAL RECOMMENDATION - EXTRACTED DATA ONLY
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
    
    # Comparable companies - EXTRACTED DATA ONLY
    comparable_companies = recommendation.get("comparable_companies", [])
    if comparable_companies:
        st.markdown("**Comparable Companies:**")
        comparable_text = ", ".join(comparable_companies)
        st.markdown(f"> {comparable_text}")
    
    # Download option
    st.markdown("---")
    memo_text = json.dumps(parsed_data, indent=2)
    st.download_button(
        label="Download Investment Analysis (JSON)",
        data=memo_text,
        file_name=f"{company_name}_signal_analysis.json",
        mime="application/json"
    )

# --- MODERN DARK UI STYLING ---
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
        padding: 2rem;
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
        overflow-x: auto;
        width: 100%;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 56px;
        padding: 0 20px;
        background: transparent;
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 14px;
        border: none;
        white-space: nowrap;
        min-width: fit-content;
        flex-shrink: 0;
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
    
    /* Value Proposition Card */
    .value-prop-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(6, 182, 212, 0.1));
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    /* Pillar Cards */
    .pillar-card {
        background: rgba(15, 23, 42, 0.8);
        border: 2px solid rgba(6, 182, 212, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .pillar-card:hover {
        border-color: rgba(6, 182, 212, 0.6);
        box-shadow: 0 15px 30px rgba(6, 182, 212, 0.2);
        transform: translateY(-5px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
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
    
    /* File Upload Styling */
    .stFileUploader {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 2px dashed rgba(6, 182, 212, 0.5) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }
    
    .stFileUploader > div {
        background: transparent !important;
        color: #e2e8f0 !important;
    }
    
    .stFileUploader label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 2px dashed rgba(6, 182, 212, 0.4) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(6, 182, 212, 0.8) !important;
        background: rgba(6, 182, 212, 0.1) !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] div {
        color: #e2e8f0 !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] span {
        color: #94a3b8 !important;
    }
    
    .stFileUploader button {
        background: linear-gradient(135deg, #06b6d4, #0ea5e9) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* File list styling */
    .stFileUploader [data-testid="stFileUploaderFile"] {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        margin: 0.5rem 0 !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderFile"] div {
        color: #e2e8f0 !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderFile"] button {
        background: rgba(239, 68, 68, 0.8) !important;
        border-radius: 6px !important;
    }
    
    /* Fix Streamlit Components for Dark Theme */
    
    /* Enhanced Expander/Collapsible styling with states */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(6, 182, 212, 0.1) !important;
        border-color: rgba(6, 182, 212, 0.6) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.2) !important;
    }
    
    .streamlit-expanderHeader[aria-expanded="false"] {
        border-radius: 12px !important;
        background: rgba(15, 23, 42, 0.6) !important;
    }
    
    .streamlit-expanderHeader[aria-expanded="true"] {
        border-radius: 12px 12px 0 0 !important;
        background: rgba(6, 182, 212, 0.1) !important;
        border-color: rgba(6, 182, 212, 0.5) !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        color: #e2e8f0 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Fix expander text */
    .streamlit-expanderContent p, 
    .streamlit-expanderContent div,
    .streamlit-expanderContent span,
    .streamlit-expanderContent [data-testid] {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Specific fix for file list in expanders */
    .streamlit-expanderContent [data-testid="column"] {
        background: transparent !important;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderContent [data-testid="column"] > div {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Force all text in expanders to be visible */
    details[open] div,
    details[open] span,
    details[open] p {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Additional fix for founder profile markdown content */
    .streamlit-expanderContent .stMarkdown {
        background: transparent !important;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderContent .stMarkdown p,
    .streamlit-expanderContent .stMarkdown strong,
    .streamlit-expanderContent .stMarkdown b {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Aggressive fix for all expander content */
    .streamlit-expanderContent * {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Specific targeting for Streamlit components in expanders */
    .streamlit-expanderContent .stText,
    .streamlit-expanderContent .stMarkdown,
    .streamlit-expanderContent [data-testid="stText"],
    .streamlit-expanderContent [data-testid="stMarkdown"] {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Target all possible text containers */
    .streamlit-expanderContent .element-container,
    .streamlit-expanderContent .stColumn,
    .streamlit-expanderContent [data-testid="column"] {
        background: transparent !important;
    }
    
    .streamlit-expanderContent .element-container *,
    .streamlit-expanderContent .stColumn *,
    .streamlit-expanderContent [data-testid="column"] * {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Specific fix for founder profile content */
    .streamlit-expanderContent [data-testid="column"] div,
    .streamlit-expanderContent [data-testid="column"] p,
    .streamlit-expanderContent [data-testid="column"] span,
    .streamlit-expanderContent [data-testid="column"] strong,
    .streamlit-expanderContent [data-testid="column"] b {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Force override any inherited styles */
    [data-testid="stExpander"] div[data-testid="column"] {
        background: transparent !important;
    }
    
    [data-testid="stExpander"] div[data-testid="column"] * {
        color: #e2e8f0 !important;
        background: transparent !important;
        border: none !important;
    }
    
    /* Target markdown content specifically */
    [data-testid="stExpander"] .stMarkdown p,
    [data-testid="stExpander"] .stMarkdown div,
    [data-testid="stExpander"] .stMarkdown strong {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    /* Metric components */
    [data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="metric-container"] > div {
        color: #e2e8f0 !important;
    }
    
    [data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #06b6d4 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 15px rgba(6, 182, 212, 0.5);
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #22c55e !important;
        font-weight: 500 !important;
    }
    
    /* Fix all text elements */
    .stMarkdown, .stText, p, div, span, li {
        color: #e2e8f0 !important;
    }
    
    /* Fix headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }
    
    /* Code blocks */
    .stCode {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* Dataframes and tables */
    .stDataFrame {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 12px !important;
    }
    
    .stDataFrame table {
        background: transparent !important;
        color: #e2e8f0 !important;
    }
    
    .stDataFrame th {
        background: rgba(6, 182, 212, 0.2) !important;
        color: #06b6d4 !important;
        font-weight: 600 !important;
    }
    
    .stDataFrame td {
        background: rgba(15, 23, 42, 0.5) !important;
        color: #e2e8f0 !important;
        border-color: rgba(6, 182, 212, 0.2) !important;
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 25px rgba(34, 197, 94, 0.4) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(34, 197, 94, 0.6) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 12px !important;
        color: #22c55e !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 12px !important;
        color: #ef4444 !important;
    }
    
    .stInfo {
        background: rgba(6, 182, 212, 0.1) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-radius: 12px !important;
        color: #06b6d4 !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 12px !important;
        color: #f59e0b !important;
    }
    
    /* Progress bar container */
    .stProgress > div > div {
        background: rgba(15, 23, 42, 0.8) !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar (if used) */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(6, 182, 212, 0.3) !important;
    }
    
    /* Plotly charts dark theme */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Force text color for markdown content */
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span,
    .stMarkdown div {
        color: #e2e8f0 !important;
    }
    
    /* Force header colors */
    .stMarkdown h1 { color: #06b6d4 !important; }
    .stMarkdown h2 { color: #06b6d4 !important; }
    .stMarkdown h3 { color: #e2e8f0 !important; }
    .stMarkdown h4 { color: #e2e8f0 !important; }
    
    /* Links */
    a {
        color: #06b6d4 !important;
    }
    
    a:hover {
        color: #22c55e !important;
    }
    
    /* Checkbox and radio */
    .stCheckbox > label {
        color: #e2e8f0 !important;
    }
    
    .stRadio > label {
        color: #e2e8f0 !important;
    }
    
    /* Slider */
    .stSlider > label {
        color: #e2e8f0 !important;
    }
    
    /* Roadmap Styling */
    .roadmap-phase {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .completed { border-left: 4px solid #10B981; }
    .in-progress { border-left: 4px solid #F59E0B; }
    .planned { border-left: 4px solid #6B7280; }
</style>
""", unsafe_allow_html=True)

# --- MAIN APP WITH ORIGINAL STRUCTURE ---

# Tab structure with original names
tab1, tab2, tab3 = st.tabs(["Vision & Strategy", "Development Roadmap", "Live Demo"])

# Tab 1: Enhanced Vision & Strategy (keeping original content)
with tab1:
    st.markdown("""
    <div class="hero-section">
        <div style="position: relative; z-index: 1;">
            <div class="hero-title">Signal AI</div>
            <div class="hero-subtitle">VC Associate-in-a-Box</div>
            <div style="color: #94a3b8; font-size: 1.1rem; opacity: 0.8;">Multi-Agent AI Platform Transforming Venture Capital Due Diligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Value Proposition
    st.markdown("""
    <div class="value-prop-card">
        <h2 style="margin-bottom: 1rem; color: #06b6d4;">Our Value Proposition</h2>
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
    
    # Problem Statement and Solution (original content)
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
    
    # Four Pillars Framework (original content)
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
    
    # Impact Metrics (original content)
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

# Tab 2: Development Roadmap (original structure)
with tab2:
    st.markdown("""
    <div class="hero-section">
        <div style="position: relative; z-index: 1;">
            <div class="hero-title">Development Roadmap</div>
            <div class="hero-subtitle">From MVP to Enterprise Platform</div>
            <div style="color: #94a3b8; font-size: 1.1rem; opacity: 0.8;">Strategic milestones and technical architecture evolution</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create and display timeline
    timeline_fig, roadmap_data = create_roadmap_timeline()
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Detailed roadmap phases (original content)
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
            <p style="margin-bottom: 1rem; color: #94a3b8;"><strong>{item['date']}</strong> - {item['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if item['achievements']:
            st.markdown("**Key Achievements:**")
            for achievement in item['achievements']:
                st.markdown(f"• {achievement}")
        
        st.markdown("---")
    
    # Technical Architecture Evolution (original content)
    st.markdown("## Technical Architecture Evolution")
    
    arch_phases = [
        {
            "phase": "Current (Sep 2025)",
            "title": "Single-Agent Foundation",
            "tech": ["Gemini 2.5 Pro API", "Streamlit Frontend", "Enhanced JSON Parsing", "Multi-modal Processing"],
            "color": "#10B981"
        },
        {
            "phase": "Next (Oct 2025)", 
            "title": "Multi-Agent System",
            "tech": ["Vertex AI Pipelines", "Gemini 2.0 Pro (Supervisor)", "Fine-tuned Gemma (Specialist)", "Agent Builder (Research)"],
            "color": "#F59E0B"
        },
        {
            "phase": "Future (Nov 2025)",
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

# Tab 3: Live Demo (original functionality with modern styling)
with tab3:
    st.markdown("""
    <div class="hero-section">
        <div style="position: relative; z-index: 1;">
            <div class="hero-title">Signal AI Live Demo</div>
            <div class="hero-subtitle">Experience Multi-Agent VC Analysis</div>
            <div style="color: #94a3b8; font-size: 1.1rem; opacity: 0.8;">Upload startup documents and witness 90% time reduction in action</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state (original)
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'parsed_data' not in st.session_state:
        st.session_state.parsed_data = None
    
    # File upload section (original functionality)
    st.markdown("## Upload Company Data Room")
    st.info("**Current:** Single-agent processing with Gemini 2.5 Pro | **Coming Oct:** Multi-agent architecture with specialized models")
    
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
    
    # Analysis section (original functionality)
    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Investment Analysis", type="primary", use_container_width=True):
                st.session_state.analysis_started = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown("**Initializing Signal AI multi-agent workflow...**")
                progress_bar.progress(20)
                
                # Process with enhanced Gemini (original function)
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
                else:
                    st.error(f"Analysis failed: {analysis_result}")
    
    # Display results using original 4-pillar framework
    if st.session_state.analysis_complete and st.session_state.parsed_data:
        display_investment_memo(st.session_state.parsed_data)
    
    elif not st.session_state.analysis_started:
        st.info("👆 Upload your startup's data room above to experience Signal AI's 90% time reduction")
        
        # Demo features preview (original content)
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

# Footer (original)
st.markdown("""
<div style="text-align: center; padding: 3rem 0; border-top: 1px solid rgba(6, 182, 212, 0.2); margin-top: 4rem;">
    <div style="color: #06b6d4; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 0 0 15px rgba(6, 182, 212, 0.5);">Signal AI</div>
    <div style="color: #64748b; font-size: 0.9rem;">VC Associate-in-a-Box | Multi-Agent Architecture on Google Cloud</div>
    <div style="color: #475569; font-size: 0.8rem; margin-top: 0.5rem;">© 2025 Signal AI. From 118 hours to 5 minutes. Built with Vertex AI.</div>
</div>
""", unsafe_allow_html=True)
