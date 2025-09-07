import gradio as gr
import pdfplumber
from io import BytesIO
from datetime import datetime
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import hashlib
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import plotly.express as px
import plotly.graph_objects as go

from hf_analyzer import OptimizedHuggingFaceAnalyzer
from resume_processor import OptimizedResumeProcessor
from pdf_generator import OptimizedResumePDFGenerator
from feedback_generator import OptimizedFeedbackGenerator, ComprehensiveFeedback

# Configuration
class Config:
    MAX_FILE_SIZE = 5 * 1024 * 1024
    SUPPORTED_FORMATS = ['.pdf', '.txt']
    HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium"
    BATCH_SIZE = 1
    MAX_LENGTH = 512
    CACHE_SIZE = 100

# Global variables with thread safety
class AppState:
    def __init__(self):
        self.hf_analyzer = None
        self.resume_processor = None
        self.pdf_generator = None
        self.feedback_generator = None
        self.analysis_history = []
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.initialization_status = "Not initialized"

app_state = AppState()

def initialize_modules():
    try:
        with app_state.lock:
            if app_state.hf_analyzer is None:
                app_state.hf_analyzer = OptimizedHuggingFaceAnalyzer(
                    model_name="microsoft/DialoGPT-medium",
                    max_length=256,
                    batch_size=1
                )

                app_state.resume_processor = OptimizedResumeProcessor()
                app_state.pdf_generator = OptimizedResumePDFGenerator()
                app_state.feedback_generator = OptimizedFeedbackGenerator(app_state.hf_analyzer)

                app_state.initialization_status = "AI modules initialized successfully!"
        return app_state.initialization_status

    except Exception as e:
        error_msg = f"Error initializing modules: {str(e)}"
        app_state.initialization_status = error_msg
        print(error_msg)
        return error_msg

def extract_text_from_pdf(pdf_file) -> str:
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            pages_to_process = min(len(pdf.pages), 5)

            for i in range(pages_to_process):
                page = pdf.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

                if len(text) > 2000:
                    break

        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting PDF: {str(e)}")

def get_cache_key(resume_text: str, job_role: str, job_description: str) -> str:
    #Generate cache key for analysis results
    content = f"{resume_text[:500]}{job_role}{job_description[:200]}"
    return hashlib.md5(content.encode()).hexdigest()

def analyze_resume(uploaded_file, pasted_text, job_role, custom_role, job_description, progress=gr.Progress()):
    #Optimized main function to analyze resume using AI
    progress(0.1, desc="Starting analysis...")
    if app_state.initialization_status != "AI modules initialized successfully!":
        init_status = initialize_modules()
        if "Error" in init_status:
            return init_status, "", "", "", "", "", "", "", "", None, None, None

    progress(0.2, desc="Processing input...")

    target_role = custom_role if job_role == "Custom" and custom_role else (job_role if job_role != "Custom" else "General")

    resume_text = ""
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.pdf'):
                progress(0.3, desc="Extracting text from PDF...")
                resume_text = extract_text_from_pdf(uploaded_file.name)
            else:
                with open(uploaded_file.name, 'r', encoding='utf-8') as f:
                    resume_text = f.read()
        elif pasted_text.strip():
            resume_text = pasted_text
        else:
            return "Please upload a resume file or paste resume text!", "", "", "", "", "", "", "", "", None, None, None

        if not resume_text:
            return "Could not extract text from the resume!", "", "", "", "", "", "", "", "", None, None, None

        if len(resume_text) > 2000:
            resume_text = resume_text[:2000] + "..."

    except Exception as e:
        return f"Error processing resume: {str(e)}", "", "", "", "", "", "", "", "", None, None, None

    progress(0.4, desc="Checking cache...")
    cache_key = get_cache_key(resume_text, target_role, job_description or "")
    if cache_key in app_state.cache:
        print("Using cached results...")
        cached_result = app_state.cache[cache_key]
        progress(1.0, desc="Analysis complete (from cache)!")
        # Unpack the cached result, which now includes the Plotly figures
        status_msg, overall_score, strengths_text, weaknesses_text, recommendations_text, \
        section_analysis, optimized_resume, missing_keywords, optimized_resume_for_pdf, \
        ats_breakdown_plot, history_plot = cached_result

        return status_msg, overall_score, strengths_text, weaknesses_text, \
               recommendations_text, section_analysis, optimized_resume, \
               missing_keywords, optimized_resume_for_pdf, ats_breakdown_plot, history_plot

    try:
        progress(0.5, desc="Analyzing resume with AI...")

        start_time = time.time()

        def analyze_with_timeout():
            return app_state.feedback_generator.generate_comprehensive_feedback(
                resume_text, target_role, job_description or ""
            )

        future = app_state.executor.submit(analyze_with_timeout)

        try:
            comprehensive_feedback = future.result(timeout=30)  # 30 second timeout
        except Exception as e:
            return f"Analysis timeout or error: {str(e)}", "", "", "", "", "", "", "", "", None, None, None

        progress(0.8, desc="Generating optimized version...")

        # Generate optimized version with fallback
        try:
            optimized_resume = app_state.hf_analyzer.optimize_resume(resume_text, target_role)
        except Exception as e:
            print(f"Warning: Could not generate optimized version: {e}")
            optimized_resume = resume_text

        progress(0.9, desc="Finalizing results...")

        # Add to history
        with app_state.lock:
            app_state.analysis_history.append({
                'role': target_role,
                'score': comprehensive_feedback.ats_analysis.total_score,
                'timestamp': datetime.now() # Store datetime object for plotting
            })

            # Keep only last 20 entries
            if len(app_state.analysis_history) > 20:
                app_state.analysis_history = app_state.analysis_history[-20:]

        # Format results for display
        score = comprehensive_feedback.ats_analysis.total_score

        # Overall assessment
        if score >= 80:
            score_message = "Excellent ATS compatibility!"
        elif score >= 60:
            score_message = "Good, but room for improvement"
        else:
            score_message = "Needs significant optimization"

        overall_result = f"**Overall ATS Score: {score:.1f}/100**\n{score_message}"

        # Component scores
        scores = comprehensive_feedback.ats_analysis.component_scores

        # Strengths and weaknesses
        strengths_text = ""
        if hasattr(comprehensive_feedback, 'overall_feedback') and comprehensive_feedback.overall_feedback.strengths:
            strengths_text = "**Strengths:**\n" + "\n".join([f"• {s}" for s in comprehensive_feedback.overall_feedback.strengths[:]])

        weaknesses_text = ""
        if hasattr(comprehensive_feedback, 'overall_feedback') and comprehensive_feedback.overall_feedback.weaknesses:
            weaknesses_text = "**Areas for Improvement:**\n" + "\n".join([f"• {w}" for w in comprehensive_feedback.overall_feedback.weaknesses[:]])

        # Recommendations
        recommendations_text = ""
        if hasattr(comprehensive_feedback, 'overall_feedback') and comprehensive_feedback.overall_feedback.recommendations:
            recommendations_text = "**Recommendations:**\n" + "\n".join([f"• {r}" for r in comprehensive_feedback.overall_feedback.recommendations[:]])

        # Section-wise analysis
        section_analysis = "**Section-wise Analysis:**\n\n"
        if hasattr(comprehensive_feedback, 'section_analyses'):
            for section in comprehensive_feedback.section_analyses[:5]:  # Limit to 5 sections
                section_analysis += f"**{section.section_name.title()} (Score: {section.score:.0f}/100)**\n"
                section_analysis += f"Feedback: {section.feedback[:200]}...\n"  # Truncate feedback
                if section.suggestions:
                    section_analysis += "Top Suggestions:\n" + "\n".join([f"• {s}" for s in section.suggestions[:5]]) + "\n"
                section_analysis += "\n"

        # Missing keywords
        missing_keywords = ""
        if comprehensive_feedback.ats_analysis.missing_keywords:
            missing_keywords = f"**Missing Keywords:** {', '.join(comprehensive_feedback.ats_analysis.missing_keywords[:])}"

        # Generate Plotly figures
        ats_breakdown_plot = create_ats_breakdown_plot(scores)
        history_plot = create_history_plot()

        # Performance info
        analysis_time = time.time() - start_time
        print(f"Analysis completed in {analysis_time:.1f} seconds")

        result = (
            "AI Analysis Complete!",
            overall_result,
            strengths_text,
            weaknesses_text,
            recommendations_text,
            section_analysis,
            optimized_resume,
            missing_keywords,
            optimized_resume,  # Add optimized resume for PDF generation
            ats_breakdown_plot,
            history_plot
        )

        # Cache the result
        if len(app_state.cache) > Config.CACHE_SIZE:
            # Remove oldest entry
            oldest_key = list(app_state.cache.keys())[0]
            del app_state.cache[oldest_key]

        app_state.cache[cache_key] = result

        progress(1.0, desc="Analysis complete!")
        return result

    except Exception as e:
        error_msg = f"Error during AI analysis: {str(e)}"
        print(error_msg)
        return error_msg, "", "", "", "", "", "", "", "", None, None, None

def create_ats_breakdown_plot(scores: Dict[str, float]):
    """Generates a Plotly bar chart for ATS component scores."""
    if not scores:
        return go.Figure() # Return empty figure JSON

    df_scores = pd.DataFrame(list(scores.items()), columns=['Component', 'Score'])
    df_scores['Score'] = df_scores['Score'].round(1)

    fig = px.bar(
        df_scores,
        x='Score',
        y='Component',
        orientation='h',
        title='ATS Component Scores',
        text='Score',
        height=300,
        color='Score',
        color_continuous_scale=px.colors.sequential.Viridis,
        range_x=[0, 100]
    )

    fig.update_layout(
        plot_bgcolor='rgba(30, 41, 59, 0.7)',
        paper_bgcolor='rgba(30, 41, 59, 0.7)',
        font_color='#e2e8f0',
        title_font_color='#e2e8f0',
        xaxis_title='',
        yaxis_title='',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        coloraxis_showscale=False
    )
    fig.update_traces(marker_line_width=0, textposition='outside', texttemplate='%{text:.1f}%')
    return fig

def create_history_plot():
    """Generates a Plotly line chart for analysis history."""
    with app_state.lock:
        if not app_state.analysis_history:
            return go.Figure() # Return empty figure JSON

        df_history = pd.DataFrame(app_state.analysis_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp']) # Ensure datetime type
        df_history = df_history.sort_values(by='timestamp')

        fig = px.line(
            df_history,
            x='timestamp',
            y='score',
            color='role',
            title='ATS Score History',
            markers=True,
            height=400
        )

        fig.update_layout(
            plot_bgcolor='rgba(30, 41, 59, 0.7)',
            paper_bgcolor='rgba(30, 41, 59, 0.7)',
            font_color='#e2e8f0',
            title_font_color='#e2e8f0',
            xaxis_title='Time',
            yaxis_title='ATS Score (%)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, range=[0, 100]),
            hovermode="x unified",
            legend_title_text='Job Role'
        )
        return fig

def generate_pdf_download(optimized_text, job_role="General"):
    #Generate PDF and return file path for download
    if not optimized_text or not optimized_text.strip():
        return None

    try:
        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_pdf_path = f"optimized_resume_{timestamp}.pdf"

        # Generate PDF using the PDF generator
        app_state.pdf_generator.generate_enhanced_pdf(optimized_text, temp_pdf_path)

        if os.path.exists(temp_pdf_path):
            return temp_pdf_path
        else:
            return None

    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

def get_analysis_history():
    #Get formatted analysis history
    # This function is now superseded by create_history_plot for the UI
    with app_state.lock:
        if not app_state.analysis_history:
            return "No analysis history yet."

        history_text = "**Recent Analysis History:**\n\n"
        for i, entry in enumerate(app_state.analysis_history[-5:], 1):  # Show last 5 entries
            history_text += f"{i}. **{entry['role']}** - Score: {entry['score']:.1f}% ({entry['timestamp'].strftime('%Y-%m-%d %H:%M')})\n"

        return history_text

def clear_cache():
    #Clear analysis cache
    with app_state.lock:
        app_state.cache.clear()
    return "Cache cleared successfully!"

# Dark Theme CSS with improved layout
DARK_THEME_CSS = """
/* Global Styles */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    min-height: 100vh !important;
    color: #e2e8f0 !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* Header Styles */
.app-header {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    color: white !important;
    padding: 1.5rem 2rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    border-radius: 0 0 16px 16px !important;
}

.app-header h1 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
}

.app-header p {
    font-size: 1.1rem !important;
    opacity: 0.9 !important;
    margin: 0 !important;
}

/* Main Content Layout */
.main-content {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 2rem !important;
    padding: 0 2rem 2rem 2rem !important;
}

/* Panel Styling (Input and Output) */
.input-panel, .output-panel {
    background: #1e293b !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    border: 1px solid #334155 !important;
    height: fit-content !important; /* Ensure input panel adjusts to content */
}

.output-panel {
    min-height: 600px !important; /* Keep a minimum height for output */
}

.input-panel h2, .output-panel h2 {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    margin-bottom: 1.5rem !important;
    color: #e2e8f0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    padding-bottom: 0.75rem !important;
    border-bottom: 2px solid #334155 !important;
}

/* File Upload Area */
.upload-container {
    border: 2px dashed #475569 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    text-align: center !important;
    background: rgba(30, 41, 59, 0.5) !important;
    transition: all 0.3s ease !important;
    margin-bottom: 1.5rem !important;
    cursor: pointer !important;
}

.upload-container:hover {
    border-color: #7c3aed !important;
    background: rgba(124, 58, 237, 0.1) !important;
}

.upload-container .icon {
    font-size: 2.5rem !important;
    color: #94a3b8 !important;
    margin-bottom: 1rem !important;
}

.upload-container h3 {
    margin: 0 0 0.5rem 0 !important;
    color: #e2e8f0 !important;
}

.upload-container p {
    margin: 0 !important;
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
}

/* Input Groups */
.input-group {
    margin-bottom: 1.5rem !important;
}

.input-group label {
    display: block !important;
    font-weight: 500 !important;
    margin-bottom: 0.5rem !important;
    color: #cbd5e1 !important;
}

/* Select and Input Styling */
.gradio-dropdown, .gradio-textbox {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    color: #e2e8f0 !important;
}

.gradio-dropdown:focus, .gradio-textbox:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2) !important;
    outline: none !important;
}

/* Button Styling */
.primary-button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    border: none !important;
    color: white !important;
    padding: 1rem 1.5rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3) !important;
}

.primary-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
}

.secondary-button {
    background: #1e293b !important;
    border: 2px solid #334155 !important;
    color: #cbd5e1 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
}

.secondary-button:hover {
    border-color: #7c3aed !important;
    color: #7c3aed !important;
    transform: translateY(-1px) !important;
}

/* Status Indicators */
.status-indicator {
    padding: 1rem !important;
    border-radius: 10px !important;
    margin-bottom: 1.5rem !important;
    font-weight: 500 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
}

.status-success {
    background: rgba(16, 185, 129, 0.1) !important;
    color: #10b981 !important;
    border-left: 4px solid #10b981 !important;
}

.status-warning {
    background: rgba(245, 158, 11, 0.1) !important;
    color: #f59e0b !important;
    border-left: 4px solid #f59e0b !important;
}

.status-error {
    background: rgba(239, 68, 68, 0.1) !important;
    color: #ef4444 !important;
    border-left: 4px solid #ef4444 !important;
}

/* Results Styling */
.score-display {
    text-align: center !important;
    padding: 1.5rem !important;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%) !important;
    border-radius: 12px !important;
    margin-bottom: 1.5rem !important;
    border: 1px solid rgba(14, 165, 233, 0.2) !important;
}

.score-number {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #0ea5e9 !important;
    margin: 0.5rem 0 !important;
}

/* Feedback Cards */
.feedback-card {
    background: rgba(30, 41, 59, 0.7) !important;
    border-radius: 10px !important;
    padding: 1.25rem !important;
    margin-bottom: 1rem !important;
    border-left: 4px solid !important;
}

.feedback-positive {
    border-left-color: #10b981 !important;
    background: rgba(16, 185, 129, 0.1) !important;
}

.feedback-warning {
    border-left-color: #f59e0b !important;
    background: rgba(245, 158, 11, 0.1) !important;
}

.feedback-info {
    border-left-color: #0ea5e9 !important;
    background: rgba(14, 165, 233, 0.1) !important;
}

/* Tabs Styling */
.tab-nav {
    background: #0f172a !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
    margin-bottom: 1.5rem !important;
    display: flex !important;
    gap: 0.5rem !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    padding: 0.75rem 1rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    color: #94a3b8 !important;
    flex: 1 !important;
}

.tab-nav button.selected {
    background: #1e293b !important;
    color: #7c3aed !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}

/* Footer */
.app-footer {
    text-align: center !important;
    padding: 2rem !important;
    margin-top: 3rem !important;
    color: #64748b !important;
    font-size: 0.9rem !important;
    border-top: 1px solid #334155 !important;
}

/* Responsive Design */
@media (max-width: 1200px) {
    .main-content {
        grid-template-columns: 1fr !important;
        gap: 1.5rem !important;
    }
}

@media (max-width: 768px) {
    .main-content {
        padding: 0 1rem 1rem 1rem !important;
    }

    .app-header {
        padding: 1rem !important;
        border-radius: 0 !important;
    }

    .app-header h1 {
        font-size: 1.8rem !important;
    }

    .input-panel, .output-panel {
        padding: 1.5rem !important;
    }
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
    animation: fadeIn 0.5s ease-out !important;
}

/* Loading Spinner */
.loading-spinner {
    border: 2px solid rgba(255, 255, 255, 0.1) !important;
    border-top: 2px solid #7c3aed !important;
    border-radius: 50% !important;
    width: 20px !important;
    height: 20px !important;
    animation: spin 1s linear infinite !important;
    display: inline-block !important;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Utility Classes */
.text-center {
    text-align: center !important;
}

.mb-2 {
    margin-bottom: 1rem !important;
}

.mt-2 {
    margin-top: 1rem !important;
}

.hidden {
    display: none !important;
}

/* Markdown content styling */
.gradio-markdown {
    color: #e2e8f0 !important;
}

.gradio-markdown strong {
    color: #e2e8f0 !important;
}

.gradio-markdown h1, .gradio-markdown h2, .gradio-markdown h3, .gradio-markdown h4 {
    color: #e2e8f0 !important;
}

.gradio-markdown ul, .gradio-markdown ol {
    color: #cbd5e1 !important;
}
.gradio-plot {
    background: rgba(30, 41, 59, 0.7) !important;
    border-radius: 10px !important;
    padding: 10px !important;
    margin-bottom: 1rem !important;
    border: 1px solid #334155 !important;
}
"""

def create_interface():

    with gr.Blocks(
        title="Smart Resume Reviewer",
        css=DARK_THEME_CSS
    ) as demo:

        # App Header
        gr.HTML("""
        <div class="app-header">
            <h1>Smart Resume Reviewer</h1>
            <p>AI-Powered ATS Optimization & Resume Enhancement</p>
        </div>
        """)

        # Status Indicator
        status_indicator = gr.HTML("""
        <div class="status-indicator status-warning">
            <div class="loading-spinner"></div>
            <strong>Initializing AI modules...</strong>
        </div>
        """)

        # Main Content Area
        with gr.Row(elem_classes="main-content"):
            # Input Panel (Left Side)
            with gr.Column(elem_classes="input-panel"):
                gr.HTML("""
                <h2>Resume Input</h2>
                """)

                # File Upload Section
                with gr.Group(elem_classes="input-group"):
                    file_upload = gr.UploadButton(
                        "Upload Resume (PDF or TXT or Docx)",
                        file_types=[".pdf", ".txt",".docx"],
                        file_count="single",
                        scale=1
                    )

                    gr.HTML("""
                    # <div class="upload-container" onclick="document.querySelector('#file-upload input[type=file]').click()">
                    #     <div class="icon">📄</div>
                    #     <h3>Drag & Drop or Click to Upload</h3>
                    #     <p>Supported formats: PDF|TXT|Docx (Max 5MB)</p>
                    # </div>  
                    # """)

                    resume_file = gr.File(
                        label=" ",
                        file_types=[".pdf", ".txt", ".docx"],
                        file_count="single",
                        elem_classes=["hidden"],
                        elem_id="file-upload"
                    )

                # Text Input Alternative
                with gr.Group(elem_classes="input-group"):
                    gr.HTML("""
                    <div class="text-center mb-2" style="color: #94a3b8; font-weight: 500;">- OR -</div>
                    """)

                    resume_text = gr.Textbox(
                        label="Paste Resume Text",
                        placeholder="Paste your resume content here...",
                        lines=5,
                        max_lines=8
                    )

                # Job Configuration Section
                gr.HTML("""
                <h2>Job Configuration</h2>
                """)

                # Job Role Selection
                with gr.Group(elem_classes="input-group"):
                    job_role = gr.Dropdown(
                        choices=[
                            "Data Scientist", "Software Engineer", "Product Manager",
                            "Marketing Manager", "Business Analyst", "Project Manager",
                            "DevOps Engineer", "UI/UX Designer", "Cybersecurity Analyst",
                            "Sales Manager", "HR Manager", "Financial Analyst",
                            "Data Engineer", "Machine Learning Engineer", "Frontend Developer",
                            "Backend Developer", "Full Stack Developer", "Custom"
                        ],
                        value="Software Engineer",
                        label="Target Job Role",
                        info="Select the role you're applying for"
                    )

                # Custom Role Input
                with gr.Group(elem_classes="input-group"):
                    custom_role = gr.Textbox(
                        label="Custom Role (if selected above)",
                        placeholder="Enter custom job role...",
                        visible=False
                    )

                # Job Description
                with gr.Group(elem_classes="input-group"):
                    job_description = gr.Textbox(
                        label="Job Description (Optional)",
                        placeholder="Paste the job description here for better analysis...",
                        lines=3,
                        info="Including job description improves analysis accuracy"
                    )

                # Action Buttons
                analyze_btn = gr.Button(
                    "Analyze Resume",
                    variant="primary",
                    elem_classes=["primary-button"]
                )

                with gr.Row():
                    clear_cache_btn = gr.Button(
                        "Clear Cache",
                        variant="secondary",
                        elem_classes=["secondary-button"],
                        scale=1
                    )

                    refresh_btn = gr.Button(
                        "Refresh Analysis History", # Changed button text
                        variant="secondary",
                        elem_classes=["secondary-button"],
                        scale=1
                    )

            # Output Panel (Right Side)
            with gr.Column(elem_classes="output-panel"):
                gr.HTML("""
                <h2>Analysis Results</h2>
                """)

                # Status Message
                status_msg = gr.HTML("")

                # Main Score and Missing Keywords (Always Visible)
                with gr.Group():
                    overall_score = gr.Markdown("", elem_classes=["score-display"])

                    # Removed component_scores markdown, replaced by plot
                    missing_keywords = gr.Markdown("")

                # ATS Breakdown Plot
                gr.HTML("""<h3>ATS Component Breakdown</h3>""")
                ats_breakdown_plot = gr.Plot(label="ATS Component Breakdown", elem_classes="gradio-plot")

                # Results Tabs
                with gr.Tabs() as results_tabs:
                    # Detailed Analysis Tab
                    with gr.TabItem("Detailed Feedback"): # Changed tab title
                        with gr.Row():
                            with gr.Column():
                                strengths_output = gr.Markdown("", elem_classes=["feedback-card", "feedback-positive"])
                            with gr.Column():
                                weaknesses_output = gr.Markdown("", elem_classes=["feedback-card", "feedback-warning"])

                        recommendations_output = gr.Markdown("", elem_classes=["feedback-card", "feedback-info"])

                    # Section Analysis Tab
                    with gr.TabItem("Section Analysis"):
                        section_analysis_output = gr.Markdown("")

                    # Optimized Resume Tab
                    with gr.TabItem("Optimized Resume"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                optimized_resume_output = gr.Textbox(
                                    label="AI-Optimized Resume Content",
                                    lines=12,
                                    show_copy_button=True
                                )

                            with gr.Column(scale=1):
                                gr.HTML("""
                                <div style="background: rgba(30, 41, 59, 0.7); padding: 1.5rem; border-radius: 10px; border: 1px solid #334155;">
                                    <h3 style="margin-top: 0; color: #e2e8f0;">Export Options</h3>
                                    <p style="color: #94a3b8; font-size: 0.9rem;">Download your optimized resume</p>
                                </div>
                                """)

                                pdf_download_btn = gr.DownloadButton(
                                    "Download PDF",
                                    variant="primary",
                                    elem_classes=["primary-button"],
                                    visible=False
                                )

                                txt_download_btn = gr.DownloadButton(
                                    "Download TXT",
                                    variant="secondary",
                                    elem_classes=["secondary-button"],
                                    visible=False
                                )

                    # History Tab
                    with gr.TabItem("Analysis History"): # Changed tab title
                        history_output_plot = gr.Plot(label="ATS Score History", elem_classes="gradio-plot") # Use Plot for history
                        history_markdown_summary = gr.Markdown("") # Keep a small markdown summary if needed


        # Hidden component to store optimized resume for PDF generation
        optimized_resume_for_pdf = gr.Textbox(visible=False)

        # Connect upload button to file component
        file_upload.upload(
            lambda files: files,
            inputs=[file_upload],
            outputs=[resume_file]
        )

        # Show/hide custom role based on job role selection
        job_role.change(
            lambda x: gr.update(visible=x=="Custom"),
            inputs=[job_role],
            outputs=[custom_role]
        )

        # Auto-initialize on load
        def auto_init():
            init_result = initialize_modules()
            if "Error" in init_result:
                return f"""<div class="status-indicator status-error"><strong>{init_result}</strong></div>""", None, None
            else:
                initial_history_plot = create_history_plot()
                return f"""<div class="status-indicator status-success"><strong>System Ready!</strong> AI modules initialized successfully</div>""", None, initial_history_plot

        demo.load(auto_init, outputs=[status_indicator, ats_breakdown_plot, history_output_plot]) # Update outputs

        # Event handlers
        analyze_btn.click(
            fn=analyze_resume,
            inputs=[resume_file, resume_text, job_role, custom_role, job_description],
            outputs=[
                status_msg, overall_score,
                strengths_output, weaknesses_output, recommendations_output,
                section_analysis_output, optimized_resume_output, missing_keywords,
                optimized_resume_for_pdf, # Hidden output for PDF generation
                ats_breakdown_plot, history_output_plot # New Plotly outputs
            ],
            show_progress=True
        )

        # PDF Download handler
        def handle_pdf_download(optimized_text):
            if not optimized_text or not optimized_text.strip():
                return gr.update(visible=False, value=None)

            try:
                pdf_path = generate_pdf_download(optimized_text)
                if pdf_path and os.path.exists(pdf_path):
                    return gr.update(visible=True, value=pdf_path)
                else:
                    return gr.update(visible=False, value=None)
            except Exception as e:
                print(f"PDF generation error: {e}")
                return gr.update(visible=False, value=None)

        optimized_resume_for_pdf.change(
            fn=handle_pdf_download,
            inputs=[optimized_resume_for_pdf],
            outputs=[pdf_download_btn]
        )

        # TXT Download handler
        def handle_txt_download(optimized_text):
            if optimized_text and optimized_text.strip():
                try:
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
                        tmp_file.write(optimized_text)
                        temp_txt_path = tmp_file.name
                    return gr.update(visible=True, value=temp_txt_path)
                except Exception as e:
                    print(f"Error creating temporary TXT file: {e}")
                    return gr.update(visible=False, value=None)
            else:
                return gr.update(visible=False, value=None)

        optimized_resume_for_pdf.change( # Trigger TXT download visibility
            fn=handle_txt_download,
            inputs=[optimized_resume_for_pdf],
            outputs=[txt_download_btn]
        )

        # Clear cache handler
        def handle_clear_cache_and_plots():
            clear_cache()
            # Also clear the plots when cache is cleared
            return """<div class="status-indicator status-success"><strong>Cache cleared successfully!</strong></div>""", \
                   None, go.Figure().to_json() # Clear ATS breakdown and history plots

        clear_cache_btn.click(
            fn=handle_clear_cache_and_plots,
            outputs=[status_msg, ats_breakdown_plot, history_output_plot]
        )

        # Refresh history handler (now directly updates the plot)
        def get_enhanced_history_plot():
            return create_history_plot()

        refresh_btn.click(
            fn=get_enhanced_history_plot,
            outputs=[history_output_plot]
        )

        # Auto-refresh history plot on page load
        demo.load(
            fn=get_enhanced_history_plot,
            outputs=[history_output_plot]
        )

    return demo

# Main execution
def main():
    # Pre-initialize modules
    print("Pre-initializing AI modules...")
    init_status = initialize_modules()
    print(f"Initialization status: {init_status}")

    # Create and launch interface
    try:
        demo = create_interface()
        # Launch with optimized settings
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=False,
            debug=False,
            inbrowser=True,
            quiet=True
        )

    except Exception as e:
        print(f"Error launching application: {e}")
if __name__ == "__main__":
    main()