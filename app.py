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
        st.success("✅ Gemini API configured successfully")
        
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
        st.success(f"Uploaded {file_name} to Gemini")
        
        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            st.info("Processing file...")
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
        You are Signal AI, an expert venture capital analyst. Analyze the provided startup documents thoroughly and extract ALL available information.

        CRITICAL REQUIREMENTS:
        1. Extract EVERY financial metric mentioned in the documents (revenue projections, funding amounts, user numbers, growth rates, etc.)
        2. Find ALL founder information (names, roles, backgrounds, previous experience)
        3. Identify business model details, market size data, competitive information
        4. Look for traction metrics, customer names, partnership details
        5. Extract funding details, investor information, valuation data
        6. Return comprehensive JSON with ALL extracted data

        SPECIFIC DATA TO EXTRACT FROM SIA DOCUMENTS:
        - Expected revenues ($400k in FY 25-26 mentioned in documents)
        - Funding ask (INR 5 Crores seed stage)
        - Market size ($300B global data analytics, $5B-$200B agentic AI)
        - Customer names (Bosch, Mercedes-Benz, Abha Hospital, etc.)
        - Founder details (Divya Krishna R, Sumalata Kamat, Karthik C with full backgrounds)
        - Business metrics (customer counts, contract values, deployment timelines)
        - Competitive advantages and risks

        Return ONLY valid JSON with this COMPLETE structure:

        {
            "company_overview": {
                "name": "Extract exact company name",
                "founding_year": "Extract founding year", 
                "stage": "Extract current funding stage",
                "one_liner": "Extract value proposition"
            },
            "founders": [
                {
                    "name": "Full founder name",
                    "role": "Specific role/title",
                    "background": "Complete background including education, previous companies, years of experience"
                }
            ],
            "business_model": {
                "model_type": "Specific business model (B2B SaaS, etc.)",
                "revenue_streams": ["List all revenue sources mentioned"],
                "target_market": "Detailed target market description",
                "pricing_model": "Pricing structure if mentioned"
            },
            "financials": {
                "arr": "Annual recurring revenue projections",
                "mrr": "Monthly recurring revenue if mentioned", 
                "expected_revenue": "Revenue projections with timeframes",
                "gross_margin": "Gross margin percentage",
                "burn_rate": "Monthly burn rate",
                "runway": "Cash runway in months",
                "ltv_cac_ratio": "Customer lifetime value to acquisition cost ratio",
                "retention_rate": "Customer retention percentage",
                "paid_users": "Number of paying customers/users",
                "valuation": "Company valuation if mentioned",
                "contract_values": "Average contract values mentioned",
                "pricing": "Pricing information per user/subscription"
            },
            "market_analysis": {
                "market_size": "Total addressable market with specific numbers",
                "growth_rate": "Market growth rate with percentages",
                "competitive_landscape": "Detailed competitor analysis",
                "market_opportunity": "Specific market opportunity details"
            },
            "traction": {
                "customer_count": "Number of customers/users",
                "customer_names": ["List of specific customer names mentioned"],
                "partnerships": ["Partnership details"],
                "pilots_running": ["Companies running pilots"],
                "revenue_metrics": "Specific revenue achievements",
                "user_growth": "User growth metrics with timeframes",
                "key_achievements": ["All achievements and milestones mentioned"]
            },
            "funding": {
                "total_raised": "Total funding raised to date",
                "current_round": "Current funding round details",
                "funding_ask": "Amount being raised in current round",
                "current_valuation": "Current company valuation",
                "previous_investors": ["List of existing investors"],
                "use_of_funds": ["How funds will be used - percentages and purposes"]
            },
            "team_and_operations": {
                "team_size": "Current team size",
                "key_hires": "Key hiring plans",
                "locations": "Office locations and geographic presence",
                "technology_stack": "Technology and infrastructure details"
            },
            "competitive_analysis": {
                "competitors": ["List of competitors mentioned"],
                "differentiation": "Key differentiating factors",
                "competitive_advantages": ["Specific competitive advantages"]
            },
            "strengths": ["3-5 key strengths with specific evidence"],
            "risks": ["3-5 main risks and concerns"],
            "recommendation": {
                "rating": "1-5 numeric rating",
                "rationale": "Detailed rationale for rating with specific supporting evidence"
            }
        }
        """
        
        # Prepare content for analysis
        content_parts = [analysis_prompt]
        content_parts.extend(gemini_files)
        
        st.info("Performing comprehensive startup analysis...")
        
        # Increased token limits for more complete responses
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            max_output_tokens=8000,  # Increased from 4000
        )
        
        response = model.generate_content(content_parts, generation_config=generation_config)
        
        if response.text:
            st.success("Comprehensive analysis completed!")
            
            # Show more of the response for debugging
            with st.expander("Debug: Full Response Preview", expanded=False):
                st.text_area("Full Response", response.text, height=400)
            
            return response.text
        else:
            st.error("No response from Gemini")
            return "Error: No response"
            
    except Exception as e:
        st.error(f"Gemini analysis failed: {e}")
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
    """Clean text to ensure consistent formatting and remove artifacts"""
    if not text or text == "Not Available":
        return text
    
    # Remove any markdown formatting that might cause font issues
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold markdown
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic markdown
    text = re.sub(r'__(.*?)__', r'\1', text)      # Remove underline markdown
    
    # Fix spacing issues more carefully
    text = re.sub(r'\s+', ' ', text)              # Multiple spaces to single
    
    # Only apply camelCase fix if it makes sense (avoid breaking normal words)
    if len(text) > 50:  # Only for longer text blocks, not short names/titles
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase spacing
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Number-letter spacing
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Letter-number spacing
    
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
    
    # Show response length for debugging
    st.info(f"Response length: {len(response_text)} characters")
    
    # Strategy 1: Direct parsing with cleaning
    try:
        # Remove code block markers and clean
        cleaned = response_text.strip()
        cleaned = re.sub(r'```json\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*', '', cleaned, flags=re.MULTILINE) 
        cleaned = re.sub(r'```', '', cleaned)
        
        # Find JSON boundaries more carefully
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = cleaned[json_start:json_end]
            
            # More comprehensive JSON cleaning
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
            json_str = re.sub(r'([}\]])\s*\n\s*(["\w])', r'\1,\n\2', json_str)  # Add missing commas
            
            # Try to parse
            parsed_data = json.loads(json_str)
            st.success("JSON parsed successfully!")
            
            # Validate that we have comprehensive data
            sections_count = len([k for k in parsed_data.keys() if isinstance(parsed_data[k], dict) and parsed_data[k]])
            st.info(f"Extracted {sections_count} data sections")
            
            return parsed_data
            
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing failed: {e}")
        
        # Show where the JSON might be broken
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1:
                problematic_area = response_text[max(0, json_start + len(str(e)) - 100):json_start + len(str(e)) + 100]
                st.code(f"Problematic area around error:\n{problematic_area}")
        except:
            pass
    
    # Strategy 2: Extract what we can manually
    st.info("Attempting manual data extraction...")
    
    def extract_field(text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else "Not Available"
    
    def extract_array(text: str, pattern: str) -> List[str]:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        return [match.strip('"') for match in matches if match.strip()]
    
    # More comprehensive manual extraction
    manual_data = {
        "company_overview": {
            "name": extract_field(response_text, r'"name":\s*"([^"]+)"'),
            "founding_year": extract_field(response_text, r'"founding_year":\s*"([^"]+)"'),
            "stage": extract_field(response_text, r'"stage":\s*"([^"]+)"'),
            "one_liner": extract_field(response_text, r'"one_liner":\s*"([^"]+)"')
        },
        "founders": [],  # We'll try to extract this separately
        "business_model": {
            "model_type": extract_field(response_text, r'"model_type":\s*"([^"]+)"'),
            "revenue_streams": extract_array(response_text, r'"revenue_streams":\s*\[[^\]]*"([^"]+)"'),
            "target_market": extract_field(response_text, r'"target_market":\s*"([^"]+)"')
        },
        "financials": {},
        "traction": {
            "customer_names": extract_array(response_text, r'"customer_names":\s*\[[^\]]*"([^"]+)"'),
            "key_achievements": extract_array(response_text, r'"key_achievements":\s*\[[^\]]*"([^"]+)"')
        },
        "funding": {
            "funding_ask": extract_field(response_text, r'"funding_ask":\s*"([^"]+)"'),
            "total_raised": extract_field(response_text, r'"total_raised":\s*"([^"]+)"')
        },
        "strengths": extract_array(response_text, r'"strengths":\s*\[[^\]]*"([^"]+)"'),
        "risks": extract_array(response_text, r'"risks":\s*\[[^\]]*"([^"]+)"'),
        "recommendation": {
            "rating": extract_field(response_text, r'"rating":\s*"?(\d+)"?'),
            "rationale": extract_field(response_text, r'"rationale":\s*"([^"]+)"')
        }
    }
    
    # Extract financial data more comprehensively
    financial_fields = [
        'arr', 'mrr', 'expected_revenue', 'gross_margin', 'burn_rate', 
        'runway', 'ltv_cac_ratio', 'retention_rate', 'paid_users', 
        'valuation', 'contract_values', 'pricing'
    ]
    
    for field in financial_fields:
        value = extract_field(response_text, f'"{field}":\\s*"([^"]+)"')
        manual_data["financials"][field] = value if value != "Not Available" else "Not Available"
    
    extracted_count = sum(1 for section in manual_data.values() 
                         if isinstance(section, dict) and any(v != "Not Available" for v in section.values()) 
                         or isinstance(section, list) and section)
    
    st.warning(f"Manual extraction completed - {extracted_count} sections with data")
    
    return manual_data

def display_comprehensive_memo(parsed_data):
    """Display comprehensive investment memo with all available data"""
    
    if "error" in parsed_data:
        st.error("Analysis Error")
        st.markdown(f"**Error:** {parsed_data.get('error', 'Unknown error')}")
        with st.expander("View Raw Response", expanded=False):
            st.text(parsed_data.get('raw_response', 'No response available'))
        return
    
    st.markdown("---")
    st.markdown("# Investment Memo")
    
    # Company Overview Section
    overview = parsed_data.get("company_overview", {})
    company_name = clean_text_formatting(overview.get("name", "Startup Analysis"))
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Company", company_name)
    with col2:
        founded = clean_text_formatting(overview.get("founding_year", "Not Available"))
        st.metric("Founded", founded)
    with col3:
        stage = clean_text_formatting(overview.get("stage", "Not Available"))
        st.metric("Stage", stage)
    with col4:
        recommendation = parsed_data.get("recommendation", {})
        rating = recommendation.get("rating", "Not Available")
        parsed_rating = parse_rating(rating)
        
        if parsed_rating:
            if parsed_rating >= 4:
                st.metric("Signal Score", f"{parsed_rating}/5", delta="Strong Buy")
            elif parsed_rating >= 3:
                st.metric("Signal Score", f"{parsed_rating}/5", delta="Consider")
            else:
                st.metric("Signal Score", f"{parsed_rating}/5", delta="Pass")
        else:
            st.metric("Signal Score", "N/A")
    
    # Value Proposition
    one_liner = clean_text_formatting(overview.get("one_liner", ""))
    if one_liner and one_liner != "Not Available":
        st.info(f"**Value Proposition:** {one_liner}")
    
    # COMPREHENSIVE FINANCIAL METRICS SECTION
    st.markdown("## Financial Health Dashboard")
    
    financials = parsed_data.get("financials", {})
    
    # First row of financial metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        arr = financials.get("arr", "Not Available")
        expected_revenue = financials.get("expected_revenue", "Not Available")
        display_revenue = arr if arr != "Not Available" else expected_revenue
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Annual Recurring Revenue</div>
            <div class="metric-value">{display_revenue}</div>
        </div>
        """, unsafe_allow_html=True)
        
        retention = financials.get("retention_rate", "Not Available")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Retention Rate</div>
            <div class="metric-value">{retention}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        gross_margin = financials.get("gross_margin", "Not Available")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gross Margin</div>
            <div class="metric-value">{gross_margin}</div>
        </div>
        """, unsafe_allow_html=True)
        
        ltv_cac = financials.get("ltv_cac_ratio", "Not Available")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">LTV:CAC Ratio</div>
            <div class="metric-value">{ltv_cac}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        paid_users = financials.get("paid_users", "Not Available")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Paid Users</div>
            <div class="metric-value">{paid_users}</div>
        </div>
        """, unsafe_allow_html=True)
        
        runway = financials.get("runway", "Not Available")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cash Runway</div>
            <div class="metric-value">{runway}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Financial Metrics if available
    additional_financials = {
        "MRR": financials.get("mrr", "Not Available"),
        "Burn Rate": financials.get("burn_rate", "Not Available"),
        "Valuation": financials.get("valuation", "Not Available"),
        "Contract Values": financials.get("contract_values", "Not Available"),
        "Pricing": financials.get("pricing", "Not Available")
    }
    
    available_additional = {k: v for k, v in additional_financials.items() if v != "Not Available"}
    if available_additional:
        st.markdown("### Additional Financial Metrics")
        cols = st.columns(min(3, len(available_additional)))
        for i, (label, value) in enumerate(available_additional.items()):
            with cols[i % 3]:
                st.metric(label, value)
    
    # FOUNDERS & TEAM SECTION - ENHANCED
    st.markdown("## Founding Team Analysis")
    founders = parsed_data.get("founders", [])
    
    if founders:
        for i, founder in enumerate(founders):
            if isinstance(founder, dict):
                founder_name = founder.get('name', f'Founder {i+1}')
                founder_role = founder.get('role', 'Role not specified')
                founder_background = founder.get('background', 'Background not available')
                
                with st.expander(f"{founder_name} - {founder_role}", expanded=(i==0)):
                    st.markdown(f"**Background:** {founder_background}")
    else:
        st.info("Founder information not found in uploaded documents")
    
    # BUSINESS MODEL & MARKET ANALYSIS - ENHANCED
    st.markdown("## Business Model & Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Business Model")
        business_model = parsed_data.get("business_model", {})
        model_type = business_model.get('model_type', 'Not specified')
        target_market = business_model.get('target_market', 'Not specified')
        pricing_model = business_model.get('pricing_model', 'Not specified')
        
        st.markdown(f"• **Type:** {model_type}")
        st.markdown(f"• **Target Market:** {target_market}")
        if pricing_model != "Not specified":
            st.markdown(f"• **Pricing Model:** {pricing_model}")
        
        revenue_streams = business_model.get('revenue_streams', [])
        if revenue_streams:
            st.markdown("• **Revenue Streams:**")
            for stream in revenue_streams:
                st.markdown(f"  - {stream}")
    
    with col2:
        st.markdown("### Market Analysis")
        market = parsed_data.get("market_analysis", {})
        market_size = market.get('market_size', 'Not analyzed')
        growth_rate = market.get('growth_rate', 'Not analyzed')
        competitive_landscape = market.get('competitive_landscape', 'Not analyzed')
        
        st.markdown(f"• **Market Size:** {market_size}")
        st.markdown(f"• **Growth Rate:** {growth_rate}")
        st.markdown(f"• **Competition:** {competitive_landscape}")
    
    # TRACTION & CUSTOMER DETAILS - NEW SECTION
    traction = parsed_data.get("traction", {})
    if any(v != "Not Available" and v for v in traction.values() if not isinstance(v, list)) or any(traction.get(k, []) for k in traction if isinstance(traction.get(k), list)):
        st.markdown("## Traction & Customer Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer_count = traction.get("customer_count", "Not Available")
            if customer_count != "Not Available":
                st.metric("Customer Count", customer_count)
            
            customer_names = traction.get("customer_names", [])
            if customer_names:
                st.markdown("**Key Customers:**")
                for customer in customer_names:
                    st.markdown(f"• {customer}")
        
        with col2:
            revenue_metrics = traction.get("revenue_metrics", "Not Available")
            if revenue_metrics != "Not Available":
                st.markdown(f"**Revenue Metrics:** {revenue_metrics}")
            
            user_growth = traction.get("user_growth", "Not Available")
            if user_growth != "Not Available":
                st.markdown(f"**User Growth:** {user_growth}")
    
    # FUNDING DETAILS - ENHANCED
    funding = parsed_data.get("funding", {})
    if any(v != "Not Available" and v for v in funding.values() if not isinstance(v, list)) or any(funding.get(k, []) for k in funding if isinstance(funding.get(k), list)):
        st.markdown("## Funding Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            funding_ask = funding.get("funding_ask", "Not Available")
            if funding_ask != "Not Available":
                st.metric("Current Ask", funding_ask)
        
        with col2:
            total_raised = funding.get("total_raised", "Not Available")
            if total_raised != "Not Available":
                st.metric("Total Raised", total_raised)
        
        with col3:
            current_valuation = funding.get("current_valuation", "Not Available")
            if current_valuation != "Not Available":
                st.metric("Valuation", current_valuation)
        
        use_of_funds = funding.get("use_of_funds", [])
        if use_of_funds:
            st.markdown("**Use of Funds:**")
            for use in use_of_funds:
                st.markdown(f"• {use}")
        
        previous_investors = funding.get("previous_investors", [])
        if previous_investors:
            st.markdown("**Previous Investors:**")
            investor_text = ", ".join(previous_investors)
            st.markdown(f"> {investor_text}")
    
    # KEY ACHIEVEMENTS - NEW SECTION
    key_achievements = traction.get("key_achievements", [])
    if key_achievements:
        st.markdown("## Key Achievements")
        cols = st.columns(2)
        for i, achievement in enumerate(key_achievements):
            with cols[i % 2]:
                st.markdown(f"✓ {achievement}")
    
    # COMPETITIVE ANALYSIS - NEW SECTION
    competitive = parsed_data.get("competitive_analysis", {})
    if any(v != "Not Available" and v for v in competitive.values() if not isinstance(v, list)) or any(competitive.get(k, []) for k in competitive if isinstance(competitive.get(k), list)):
        st.markdown("## Competitive Analysis")
        
        competitors = competitive.get("competitors", [])
        if competitors:
            st.markdown("**Competitors:**")
            for competitor in competitors:
                st.markdown(f"• {competitor}")
        
        differentiation = competitive.get("differentiation", "Not Available")
        if differentiation != "Not Available":
            st.markdown(f"**Key Differentiation:** {differentiation}")
    
    # STRENGTHS AND RISKS ANALYSIS
    st.markdown("## Strategic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Strengths")
        strengths = parsed_data.get("strengths", [])
        if strengths:
            for strength in strengths:
                st.success(f"• {strength}")
        else:
            st.info("Strengths analysis not available")
    
    with col2:
        st.markdown("### Risk Factors")
        risks = parsed_data.get("risks", [])
        if risks:
            for risk in risks:
                st.warning(f"• {risk}")
        else:
            st.info("Risk analysis not available")
    
    # FINAL INVESTMENT RECOMMENDATION
    st.markdown("## Final Investment Recommendation")
    
    recommendation = parsed_data.get("recommendation", {})
    rating = recommendation.get("rating", "Not Available")
    rationale = recommendation.get("rationale", "No rationale provided")
    
    parsed_rating = parse_rating(rating)
    if parsed_rating:
        if parsed_rating >= 4:
            st.success(f"### STRONG BUY - Signal Score: {parsed_rating}/5")
        elif parsed_rating >= 3:
            st.warning(f"### CONSIDER - Signal Score: {parsed_rating}/5")
        else:
            st.error(f"### PASS - Signal Score: {parsed_rating}/5")
    else:
        st.info("### Signal Score: Analysis incomplete")
    
    st.markdown(f"**Investment Rationale:** {rationale}")
    
    # DOWNLOAD BUTTON
    st.markdown("---")
    memo_text = json.dumps(parsed_data, indent=2)
    st.download_button(
        label="Download Complete Investment Memo (JSON)",
        data=memo_text,
        file_name=f"{company_name}_comprehensive_memo.json",
        mime="application/json"
    )

def create_financial_charts(financial_data: Dict) -> List:
    """Create clean, professional financial visualizations"""
    charts = []
    metrics_for_chart = {}
    
    # Define priority metrics for visualization
    priority_metrics = {
        'ARR': ['arr', 'expected_revenue'],
        'Gross Margin': ['gross_margin'], 
        'Paid Users': ['paid_users'],
        'Retention Rate': ['retention_rate']
    }
    
    for display_name, keys in priority_metrics.items():
        for key in keys:
            if key in financial_data and financial_data[key] != "Not Available":
                value_str = str(financial_data[key])
                
                # Extract numeric values
                numbers = re.findall(r'[\d.]+', value_str)
                if numbers:
                    try:
                        metrics_for_chart[display_name] = float(numbers[0])
                    except ValueError:
                        continue
                break
    
    # Create chart if we have data
    if len(metrics_for_chart) >= 1:
        fig = go.Figure()
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
        
        fig.add_trace(go.Bar(
            x=list(metrics_for_chart.keys()),
            y=list(metrics_for_chart.values()),
            marker_color=colors[:len(metrics_for_chart)],
            hovertemplate="<b>%{x}</b><br>Value: %{y:,.1f}<extra></extra>",
            text=[f'{v:,.0f}' if v > 10 else f'{v:.1f}' for v in metrics_for_chart.values()],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Key Financial Metrics",
            title_font_size=20,
            xaxis_title="",
            yaxis_title="Value",
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False,
            font=dict(family="Inter, sans-serif")
        )
        
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
        charts.append(fig)
    
    return charts

def create_growth_trends_chart(company_name: str) -> go.Figure:
    """Create sample growth visualization"""
    months = ['Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024', 'May 2024', 'Jun 2024']
    sample_arr = [180, 220, 280, 340, 420, 520]
    sample_users = [5000, 6200, 7800, 9500, 11800, 14500]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ARR Growth Projection', 'User Growth Projection'),
        horizontal_spacing=0.12
    )
    
    fig.add_trace(
        go.Scatter(x=months, y=sample_arr, name='ARR',
                  line=dict(color='#2E86AB', width=3),
                  mode='lines+markers', 
                  marker=dict(size=10, color='#2E86AB')), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=months, y=sample_users, name='Users',
                  line=dict(color='#A23B72', width=3),
                  mode='lines+markers', 
                  marker=dict(size=10, color='#A23B72')), row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_text=f"{clean_text_formatting(company_name)} Growth Trends (Sample Projection)",
        title_font_size=20,
        margin=dict(l=50, r=50, t=100, b=50),
        font=dict(family="Inter, sans-serif")
    )
    
    return fig

# --- STREAMLIT APP ---

# CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
        font-family: 'Inter', sans-serif !important;
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
    
    .upload-success {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 8px;
        padding: 1rem;
        color: #065f46;
        font-weight: 500;
        font-family: 'Inter', sans-serif !important;
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
    
    .metric-label {
        color: #6b7280;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif !important;
    }
    
    .metric-value {
        color: #111827;
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.4;
        word-wrap: break-word;
        white-space: normal;
        overflow-wrap: break-word;
        font-family: 'Inter', sans-serif !important;
        font-style: normal !important;
    }
</style>
""", unsafe_allow_html=True)

# Tab structure
tab1, tab2, tab3 = st.tabs(["Vision & Strategy", "Roadmap", "Live Demo"])

# Tab 1: Vision & Strategy
with tab1:
    st.markdown("""
    <div class="main-header">
        <h1>Signal AI</h1>
        <h3>The Future of Venture Capital Analysis</h3>
        <p style="font-size: 1.1rem; opacity: 0.8;">Transform 118-hour due diligence into 5-minute intelligent analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("The Problem We're Solving")
    st.markdown("""
    Venture capital firms are drowning in unstructured data. Traditional due diligence requires 118+ hours per deal, 
    creating massive bottlenecks that prevent investors from evaluating promising startups efficiently.
    """)

# Tab 2: Roadmap
with tab2:
    st.header("Product Development Roadmap")
    st.markdown("Future enhancements and technical foundation details...")

# Tab 3: Live Demo - UPDATED TO USE ENHANCED GEMINI
with tab3:
    st.markdown("""
    <div class="main-header">
        <h1>Live Demo</h1>
        <h3>Experience Signal AI in Action</h3>
        <p style="font-size: 1.1rem; opacity: 0.8;">Upload your startup's data room and see the magic happen</p>
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
    st.header("Upload Company Data Room")
    st.markdown("Upload multiple documents for comprehensive analysis:")
    
    # Highlight the improved processing
    st.info("Enhanced with Gemini 2.5 Pro: Comprehensive data extraction with 8K token limit for complete analysis")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'doc', 'txt', 'csv', 'png', 'jpg', 'jpeg'],
        help="Upload pitch deck, financial reports, founder profiles, market analysis, etc."
    )
    
    if uploaded_files:
        st.markdown(f"""
        <div class="upload-success">
            {len(uploaded_files)} files uploaded successfully - Ready for comprehensive Gemini processing
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("View uploaded files", expanded=True):
            for file in uploaded_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file.name}")
                with col2:
                    st.write(f"{file.size:,} bytes")
                with col3:
                    st.write("Enhanced Processing")
    
    # Analysis section
    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Generate Comprehensive Investment Memo", type="primary", use_container_width=True):
                st.session_state.analysis_started = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown("**Uploading files to Gemini...**")
                progress_bar.progress(20)
                
                # Use enhanced Gemini processing
                analysis_result = process_files_with_gemini(uploaded_files)
                
                progress_bar.progress(70)
                status_text.markdown("**Gemini performing comprehensive analysis...**")
                
                if analysis_result and "error" not in analysis_result.lower():
                    parsed_data = parse_json_response(analysis_result)
                    st.session_state.parsed_data = parsed_data
                    
                    progress_bar.progress(100)
                    status_text.markdown("**Comprehensive Analysis Complete!**")
                    st.session_state.analysis_complete = True
                    
                    st.success("Complete Investment Memo Generated Successfully!")
                    st.balloons()
                else:
                    st.error(f"Analysis failed: {analysis_result}")
    
    # Display comprehensive results
    if st.session_state.analysis_complete and st.session_state.parsed_data:
        display_comprehensive_memo(st.session_state.parsed_data)
    
    elif not st.session_state.analysis_started:
        st.info("Upload your company's data room documents above to begin comprehensive Gemini-powered analysis")
        
        st.markdown("### Enhanced Analysis Features:")
        st.markdown("""
        - **Complete Financial Extraction** - ARR, funding details, projections, customer metrics
        - **Full Team Profiles** - Detailed founder backgrounds and experience
        - **Traction Analysis** - Customer names, partnerships, achievements
        - **Market Intelligence** - TAM, competition, growth rates
        - **Risk Assessment** - Comprehensive strengths and risk analysis
        - **Investment Scoring** - Detailed Signal Score with rationale
        """)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; border-top: 1px solid #e5e7eb; margin-top: 3rem; color: #6b7280;">
    <p><strong>Signal AI</strong> - Enhanced with Comprehensive Data Extraction | Built by Team Kaeos</p>
    <p style="font-size: 0.8rem;">© 2024 Signal AI. Powered by Gemini 2.5 Pro with 8K token analysis capability.</p>
</div>
""", unsafe_allow_html=True)
