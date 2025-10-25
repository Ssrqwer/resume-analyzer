import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2
import io
import json
import re
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer & Builder",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""

def setup_apis():
    """Setup API configurations"""
    # Gemini API setup
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    
    # OpenAI API setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = None
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    
    return gemini_api_key, openai_client

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def analyze_resume_with_gemini(resume_text: str, model_name: str) -> Dict[str, Any]:
    """Analyze resume using Gemini API"""
    try:
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Analyze the following resume and provide a comprehensive evaluation:

        Resume Text:
        {resume_text}

        Please provide analysis in the following JSON format:
        {{
            "overall_score": "X/10",
            "strengths": ["strength1", "strength2", "strength3"],
            "weaknesses": ["weakness1", "weakness2", "weakness3"],
            "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
            "missing_sections": ["section1", "section2"],
            "keyword_optimization": "analysis of keywords",
            "ats_compatibility": "rating and suggestions",
            "experience_analysis": "detailed analysis of work experience",
            "education_analysis": "analysis of educational background",
            "skills_analysis": "analysis of technical and soft skills",
            "formatting_feedback": "suggestions for better formatting"
        }}
        
        Make sure the response is valid JSON format.
        """
        
        response = model.generate_content(prompt)
        
        # Try to extract JSON from response
        response_text = response.text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # Fallback if JSON parsing fails
            return {
                "overall_score": "Analysis completed",
                "analysis": response_text,
                "error": "Could not parse structured response"
            }
    
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}

def analyze_resume_with_openai(resume_text: str, model_name: str, client) -> Dict[str, Any]:
    """Analyze resume using OpenAI API (updated for v1.0+)"""
    try:
        prompt = f"""
        Analyze the following resume and provide a comprehensive evaluation:

        Resume Text:
        {resume_text}

        Please provide analysis in the following JSON format:
        {{
            "overall_score": "X/10",
            "strengths": ["strength1", "strength2", "strength3"],
            "weaknesses": ["weakness1", "weakness2", "weakness3"],
            "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
            "missing_sections": ["section1", "section2"],
            "keyword_optimization": "analysis of keywords",
            "ats_compatibility": "rating and suggestions",
            "experience_analysis": "detailed analysis of work experience",
            "education_analysis": "analysis of educational background",
            "skills_analysis": "analysis of technical and soft skills",
            "formatting_feedback": "suggestions for better formatting"
        }}
        
        Make sure the response is valid JSON format.
        """
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return {
                "overall_score": "Analysis completed",
                "analysis": response_text,
                "error": "Could not parse structured response"
            }
    
    except Exception as e:
        return {"error": f"OpenAI API error: {str(e)}"}

def generate_resume_suggestions(analysis_result: Dict[str, Any], api_choice: str, model_name: str, client=None) -> str:
    """Generate improved resume suggestions"""
    try:
        if 'error' in analysis_result:
            return "Cannot generate suggestions due to analysis error."
        
        prompt = f"""
        Based on the following resume analysis, create specific, actionable suggestions for improving the resume:
        
        Analysis Results: {json.dumps(analysis_result, indent=2)}
        
        Please provide:
        1. Specific wording improvements
        2. Section restructuring suggestions
        3. Keyword recommendations
        4. Formatting improvements
        5. Content additions/modifications
        
        Make the suggestions practical and implementable.
        """
        
        if api_choice == "Gemini":
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        else:  # OpenAI
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"

def display_analysis_results(analysis_result: Dict[str, Any]):
    """Display analysis results in a structured format"""
    if 'error' in analysis_result:
        st.error(analysis_result['error'])
        return
    
    # Overall Score
    if 'overall_score' in analysis_result:
        st.subheader("📊 Overall Score")
        st.info(f"*Score: {analysis_result['overall_score']}*")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Strengths & Weaknesses", "💡 Suggestions", "🔍 Detailed Analysis", "📝 Raw Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Strengths")
            if 'strengths' in analysis_result and isinstance(analysis_result['strengths'], list):
                for strength in analysis_result['strengths']:
                    st.write(f"• {strength}")
            else:
                st.write("No specific strengths identified.")
        
        with col2:
            st.subheader("⚠ Areas for Improvement")
            if 'weaknesses' in analysis_result and isinstance(analysis_result['weaknesses'], list):
                for weakness in analysis_result['weaknesses']:
                    st.write(f"• {weakness}")
            else:
                st.write("No specific weaknesses identified.")
    
    with tab2:
        st.subheader("💡 Improvement Suggestions")
        if 'suggestions' in analysis_result and isinstance(analysis_result['suggestions'], list):
            for i, suggestion in enumerate(analysis_result['suggestions'], 1):
                st.write(f"{i}. {suggestion}")
        else:
            st.write("No specific suggestions available.")
        
        if 'missing_sections' in analysis_result and isinstance(analysis_result['missing_sections'], list):
            st.subheader("📋 Missing Sections")
            for section in analysis_result['missing_sections']:
                st.write(f"• {section}")
    
    with tab3:
        analysis = [
            ("🎯 Keyword Optimization", "keyword_optimization"),
            ("🤖 ATS Compatibility", "ats_compatibility"),
            ("💼 Experience Analysis", "experience_analysis"),
            ("🎓 Education Analysis", "education_analysis"),
            ("🛠 Skills Analysis", "skills_analysis"),
            ("📄 Formatting Feedback", "formatting_feedback")
        ]
        
        for title, key in analysis:
            if key in analysis_result:
                st.subheader(title)
                st.write(analysis_result[key])
    
    with tab4:
        st.subheader("🔍 Complete Analysis")
        st.json(analysis_result)

def main():
    st.title("🚀 AI Resume Analyzer & Builder")
    st.markdown("Upload your resume and get AI-powered analysis and improvement suggestions!")
    
    # Setup APIs
    gemini_api_key, openai_client = setup_apis()
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙ Configuration")
        
        # API Choice
        api_options = []
        if gemini_api_key:
            api_options.append("Gemini")
        if openai_client:
            api_options.append("OpenAI")
        
        if not api_options:
            st.error("Please configure at least one API key in your .env file")
            st.stop()
        
        api_choice = st.selectbox("Choose AI Provider:", api_options)
        
        # Model selection based on API choice
        if api_choice == "Gemini":
            # Fetch available Gemini models dynamically
            try:
                models = genai.list_models()
                model_options = [
                    m.name for m in models if "generateContent" in m.supported_generation_methods
                ]

                if not model_options:
                    st.warning("⚠️ No available Gemini models found. Check your API key or permissions.")
                    model_options = ["gemini-pro"]  # fallback
            except Exception as e:
                st.error(f"Error fetching models: {e}")
                model_options = ["gemini-pro"]

            model_choice = st.selectbox("Choose Gemini Model:", model_options)

        else:
            # Warning about OpenAI's API policies
            st.warning(
                "Please note: OpenAI's API is subject to their policies regarding the free tier plan. "
                "If no free tier plan is available or if your usage exceeds the free quota, the service may not work."
            )
            
            try:
                # Fetch available models from OpenAI API
                response = openai_client.models.list()  # This fetches the available models from OpenAI API.
                model_options = [model.id for model in response]  # Extract model IDs
                
                # Model selection dropdown
                model_choice = st.selectbox("Choose OpenAI Model:", model_options)

            except Exception as e:
                st.error(f"Error fetching models: {e}")
                model_options = []  # If error, leave the model options empty
                st.error("No models available. Please check your API connection or plan.")
                
            st.markdown("---")
            st.markdown("*API Status:*")
        if gemini_api_key:
            st.success("✅ Gemini API Key Found")
        else:
            st.warning("⚠ Gemini API Key Not Found")
        
        if openai_client:
            st.success("✅ OpenAI API Key Found")
        else:
            st.warning("⚠ OpenAI API Key Not Found")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Resume")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'txt'],
            help="Upload your resume in PDF or TXT format"
        )
        
        resume_text = ""
        
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            else:  # txt file
                resume_text = str(uploaded_file.read(), "utf-8")
            
            if resume_text:
                st.session_state.resume_text = resume_text
                st.success("✅ Resume uploaded successfully!")
                
                # Show preview
                with st.expander("📄 Resume Preview"):
                    st.text_area("Resume Content", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200, disabled=True)
        
        # Manual text input option
        st.subheader("📝 Or paste your resume text:")
        manual_text = st.text_area("Paste resume content here:", height=300)
        
        if manual_text:
            st.session_state.resume_text = manual_text
            resume_text = manual_text
    
    with col2:
        st.header("🎯 Analysis & Results")
        
        if st.session_state.resume_text:
            if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing resume with {api_choice}..."):
                    if api_choice == "Gemini":
                        analysis_result = analyze_resume_with_gemini(st.session_state.resume_text, model_choice)
                    else:
                        analysis_result = analyze_resume_with_openai(st.session_state.resume_text, model_choice, openai_client)
                    
                    st.session_state.analysis_result = analysis_result
            
            # Display results if available
            if st.session_state.analysis_result:
                display_analysis_results(st.session_state.analysis_result)
                
                # Generate improvement suggestions
                if st.button("💡 Generate Improvement Suggestions", use_container_width=True):
                    with st.spinner("Generating suggestions..."):
                        if api_choice == "Gemini":
                            suggestions = generate_resume_suggestions(
                                st.session_state.analysis_result, 
                                api_choice, 
                                model_choice, 
                                gemini_api_key
                            )
                        else:
                            suggestions = generate_resume_suggestions(
                                st.session_state.analysis_result, 
                                api_choice, 
                                model_choice, 
                                openai_api_key
                            )
                        
                        st.subheader("🚀 Personalized Improvement Plan")
                        st.markdown(suggestions)
        else:
            st.info("👆 Please upload a resume or paste resume text to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤ using Streamlit | Powered by Gemini & OpenAI APIs</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()