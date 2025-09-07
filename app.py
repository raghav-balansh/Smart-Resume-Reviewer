import streamlit as st
import pdfplumber
from io import BytesIO
import json
import re
from datetime import datetime
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import hashlib
import tempfile

# Import our custom modules
from hf_analyzer import HuggingFaceAnalyzer
from resume_processor import ResumeProcessor
from pdf_generator import ResumePDFGenerator
from feedback_generator import SectionWiseFeedbackGenerator, ComprehensiveFeedback

# Configuration
class Config:
    """Application configuration"""
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    SUPPORTED_FORMATS = ['.pdf', '.txt']
    HUGGINGFACE_MODEL = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'comprehensive_feedback' not in st.session_state:
        st.session_state.comprehensive_feedback = None
    if 'optimized_resume' not in st.session_state:
        st.session_state.optimized_resume = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'hf_analyzer' not in st.session_state:
        st.session_state.hf_analyzer = None
    if 'resume_processor' not in st.session_state:
        st.session_state.resume_processor = None
    if 'pdf_generator' not in st.session_state:
        st.session_state.pdf_generator = None
    if 'feedback_generator' not in st.session_state:
        st.session_state.feedback_generator = None

# Initialize modules
def initialize_modules():
    """Initialize all required modules"""
    if st.session_state.hf_analyzer is None:
        st.session_state.hf_analyzer = HuggingFaceAnalyzer()
    if st.session_state.resume_processor is None:
        st.session_state.resume_processor = ResumeProcessor()
    if st.session_state.pdf_generator is None:
        st.session_state.pdf_generator = ResumePDFGenerator()
    if st.session_state.feedback_generator is None:
        st.session_state.feedback_generator = SectionWiseFeedbackGenerator(st.session_state.hf_analyzer)

# PDF Processing Helper
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting PDF: {str(e)}")

# Main Streamlit Application
def main():
    st.set_page_config(
        page_title="AI Resume Optimizer with Hugging Face",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #2E4057;
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .section-feedback {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #048A81;
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    initialize_modules()
    
    # Header
    st.title("🤖 AI Resume Optimizer with Hugging Face")
    st.markdown("### Powered by Meta Llama 3.2 - Get professional resume feedback and optimization!")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Information
        st.subheader("🤖 AI Model")
        st.info(f"Using: {Config.HUGGINGFACE_MODEL}")
        st.caption("This model provides comprehensive resume analysis and optimization")
        
        st.divider()
        
        # Job Configuration
        st.subheader("Job Details")
        job_roles = ["Data Scientist", "Software Engineer", "Product Manager", 
                    "Marketing Manager", "Business Analyst", "Project Manager", 
                    "DevOps Engineer", "UI/UX Designer", "Cybersecurity Analyst", "Custom"]
        
        selected_role = st.selectbox("Select Job Role:", job_roles)
        
        if selected_role == "Custom":
            custom_role = st.text_input("Enter Custom Role:")
            job_role = custom_role if custom_role else "General"
        else:
            job_role = selected_role
        
        # Optional Job Description
        with st.expander("Add Job Description (Optional)"):
            job_description = st.text_area("Paste the job description here:", height=200)
        
        st.divider()
        
        # Analysis History
        if st.session_state.analysis_history:
            st.subheader("📊 Analysis History")
            for idx, history in enumerate(st.session_state.analysis_history[-3:], 1):
                st.info(f"{idx}. {history['role']} - Score: {history['score']:.1f}%")
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Resume")
        
        # File Upload
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=['pdf', 'txt'],
            help="Upload your resume in PDF or TXT format (max 5MB)"
        )
        
        # Text Input Alternative
        with st.expander("Or paste your resume text"):
            pasted_text = st.text_area("Paste resume text here:", height=300)
        
        # Process Resume
        if st.button("🤖 Analyze Resume with AI", type="primary"):
            # Get resume text
            resume_text = ""
            
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = str(uploaded_file.read(), "utf-8")
            elif pasted_text:
                resume_text = pasted_text
            else:
                st.warning("Please upload a resume or paste text!")
                return
            
            if not resume_text:
                st.error("Could not extract text from the resume!")
                return
            
            # Store in session state
            st.session_state.resume_text = resume_text
            
            # Generate Comprehensive Feedback
            with st.spinner("🤖 AI is analyzing your resume..."):
                try:
                    comprehensive_feedback = st.session_state.feedback_generator.generate_comprehensive_feedback(
                        resume_text, job_role, job_description
                    )
                    st.session_state.comprehensive_feedback = comprehensive_feedback
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    return
            
            # Generate Optimized Version
            with st.spinner("✨ Creating optimized resume..."):
                try:
                    optimized = st.session_state.hf_analyzer.optimize_resume(resume_text, job_role)
                    st.session_state.optimized_resume = optimized
                except Exception as e:
                    st.warning(f"Could not generate optimized version: {str(e)}")
                    st.session_state.optimized_resume = resume_text
            
            # Add to history
            st.session_state.analysis_history.append({
                'role': job_role,
                'score': comprehensive_feedback.ats_analysis.total_score,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            
            st.success("✅ AI Analysis Complete!")
    
    with col2:
        st.header("📊 AI Analysis Results")
        
        if st.session_state.comprehensive_feedback:
            feedback = st.session_state.comprehensive_feedback
            
            # Overall Score Display
            score = feedback.ats_analysis.total_score
            
            # Score color coding
            if score >= 80:
                score_color = "🟢"
                score_message = "Excellent ATS compatibility!"
            elif score >= 60:
                score_color = "🟡"
                score_message = "Good, but room for improvement"
            else:
                score_color = "🔴"
                score_message = "Needs significant optimization"
            
            # Display main score
            st.markdown(f"### {score_color} Overall ATS Score: {score:.1f}/100")
            st.caption(score_message)
            
            # Progress bar
            st.progress(score / 100)
            
            # Component Scores
            st.subheader("📈 Detailed Breakdown")
            
            scores = feedback.ats_analysis.component_scores
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Keyword Match", f"{scores['keyword_match']:.0f}%")
                st.metric("Formatting", f"{scores['formatting']:.0f}%")
                st.metric("Readability", f"{scores['readability']:.0f}%")
            
            with col_b:
                st.metric("Sections", f"{scores['section_completeness']:.0f}%")
                st.metric("Length", f"{scores['length_appropriateness']:.0f}%")
            
            # Estimated Improvement
            if feedback.estimated_improvement > score:
                improvement = feedback.estimated_improvement - score
                st.success(f"📈 Estimated improvement potential: +{improvement:.1f} points")
            
            # Missing Keywords
            if feedback.ats_analysis.missing_keywords:
                st.subheader("🔑 Missing Keywords")
                missing = feedback.ats_analysis.missing_keywords
                st.warning(f"Add these keywords: {', '.join(missing[:5])}")
            
            # Priority Improvements
            if feedback.priority_improvements:
                st.subheader("🎯 Priority Improvements")
                for i, improvement in enumerate(feedback.priority_improvements[:3], 1):
                    st.info(f"{i}. {improvement}")
    
    # Comprehensive Feedback Section
    if st.session_state.comprehensive_feedback:
        st.divider()
        st.header("💡 AI-Powered Feedback & Recommendations")
        
        feedback = st.session_state.comprehensive_feedback
        
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Overall Feedback", "📋 Section Analysis", "✨ Optimized Resume", "⬇️ Download"])
        
        with tab1:
            st.subheader("🤖 AI Overall Assessment")
            
            # Overall feedback from HF model
            if feedback.overall_feedback.strengths:
                st.subheader("✅ Strengths")
                for strength in feedback.overall_feedback.strengths:
                    st.success(f"• {strength}")
            
            if feedback.overall_feedback.weaknesses:
                st.subheader("⚠️ Areas for Improvement")
                for weakness in feedback.overall_feedback.weaknesses:
                    st.warning(f"• {weakness}")
            
            if feedback.overall_feedback.recommendations:
                st.subheader("💡 Recommendations")
                for rec in feedback.overall_feedback.recommendations:
                    st.info(f"• {rec}")
        
        with tab2:
            st.subheader("📋 Section-wise Analysis")
            
            for section in feedback.section_analyses:
                with st.expander(f"{section.section_name.title()} Section (Score: {section.score:.0f}/100)"):
                    st.write(f"**Feedback:** {section.feedback}")
                    
                    if section.suggestions:
                        st.write("**Suggestions:**")
                        for suggestion in section.suggestions:
                            st.write(f"• {suggestion}")
                    
                    if section.missing_keywords:
                        st.write("**Missing Keywords:**")
                        st.write(", ".join(section.missing_keywords[:5]))
        
        with tab3:
            if st.session_state.optimized_resume:
                st.subheader("✨ AI-Optimized Resume")
                st.text_area("Enhanced Resume Content:", 
                           value=st.session_state.optimized_resume,
                           height=500)
                
                # Show improvement
                if feedback.estimated_improvement > feedback.ats_analysis.total_score:
                    improvement = feedback.estimated_improvement - feedback.ats_analysis.total_score
                    st.success(f"📈 Estimated score improvement: {feedback.ats_analysis.total_score:.0f}% → {feedback.estimated_improvement:.0f}% (+{improvement:.1f} points)")
        
        with tab4:
            st.subheader("📄 Download Professional Resume")
            
            if st.session_state.optimized_resume:
                try:
                    # Generate enhanced PDF
                    pdf_buffer = BytesIO()
                    temp_pdf_path = "temp_optimized_resume.pdf"
                    st.session_state.pdf_generator.generate_enhanced_pdf(
                        st.session_state.optimized_resume, temp_pdf_path
                    )
                    
                    with open(temp_pdf_path, 'rb') as f:
                        pdf_data = f.read()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📄 Download Enhanced PDF",
                            data=pdf_data,
                            file_name=f"enhanced_resume_{job_role.lower().replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )
                    
                    with col2:
                        st.download_button(
                            label="📝 Download as TXT",
                            data=st.session_state.optimized_resume,
                            file_name=f"optimized_resume_{job_role.lower().replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                    
                    # Clean up temp file
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)
                    
                    st.info("💡 Tip: The enhanced PDF uses professional formatting and is ATS-optimized!")
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    # Fallback to text download
                    st.download_button(
                        label="📝 Download as TXT",
                        data=st.session_state.optimized_resume,
                        file_name=f"optimized_resume_{job_role.lower().replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🔒 Your data is processed securely and not stored</p>
        <p>Built with ❤️ using Streamlit and Hugging Face Transformers</p>
        <p>Powered by Meta Llama 3.2 - Advanced AI Resume Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()