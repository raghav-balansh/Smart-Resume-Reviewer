import gradio as gr
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

# Global variables to store analyzer instances
hf_analyzer = None
resume_processor = None
pdf_generator = None
feedback_generator = None
analysis_history = []

def initialize_modules():
    """Initialize all required modules"""
    global hf_analyzer, resume_processor, pdf_generator, feedback_generator
    
    try:
        if hf_analyzer is None:
            hf_analyzer = HuggingFaceAnalyzer()
        if resume_processor is None:
            resume_processor = ResumeProcessor()
        if pdf_generator is None:
            pdf_generator = ResumePDFGenerator()
        if feedback_generator is None:
            feedback_generator = SectionWiseFeedbackGenerator(hf_analyzer)
        return "✅ AI modules initialized successfully!"
    except Exception as e:
        return f"❌ Error initializing modules: {str(e)}"

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

def analyze_resume(uploaded_file, pasted_text, job_role, custom_role, job_description):
    """Main function to analyze resume using AI"""
    global feedback_generator, hf_analyzer, analysis_history
    
    # Initialize modules if not done
    init_status = initialize_modules()
    if "Error" in init_status:
        return init_status, "", "", "", "", "", ""
    
    # Determine job role
    if job_role == "Custom" and custom_role:
        target_role = custom_role
    elif job_role != "Custom":
        target_role = job_role
    else:
        target_role = "General"
    
    # Get resume text
    resume_text = ""
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.pdf'):
                resume_text = extract_text_from_pdf(uploaded_file.name)
            else:
                with open(uploaded_file.name, 'r', encoding='utf-8') as f:
                    resume_text = f.read()
        elif pasted_text.strip():
            resume_text = pasted_text
        else:
            return "⚠️ Please upload a resume file or paste resume text!", "", "", "", "", "", ""
        
        if not resume_text:
            return "❌ Could not extract text from the resume!", "", "", "", "", "", ""
        
    except Exception as e:
        return f"❌ Error processing resume: {str(e)}", "", "", "", "", "", ""
    
    try:
        # Generate comprehensive feedback
        comprehensive_feedback = feedback_generator.generate_comprehensive_feedback(
            resume_text, target_role, job_description
        )
        
        # Generate optimized version
        try:
            optimized_resume = hf_analyzer.optimize_resume(resume_text, target_role)
        except Exception as e:
            print(f"Warning: Could not generate optimized version: {e}")
            optimized_resume = resume_text
        
        # Add to history
        analysis_history.append({
            'role': target_role,
            'score': comprehensive_feedback.ats_analysis.total_score,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        
        # Format results for display
        score = comprehensive_feedback.ats_analysis.total_score
        
        # Overall assessment
        if score >= 80:
            score_emoji = "🟢"
            score_message = "Excellent ATS compatibility!"
        elif score >= 60:
            score_emoji = "🟡"
            score_message = "Good, but room for improvement"
        else:
            score_emoji = "🔴"
            score_message = "Needs significant optimization"
        
        overall_result = f"{score_emoji} **Overall ATS Score: {score:.1f}/100**\n{score_message}"
        
        # Component scores
        scores = comprehensive_feedback.ats_analysis.component_scores
        component_scores = f"""
**Detailed Breakdown:**
• Keyword Match: {scores['keyword_match']:.0f}%
• Formatting: {scores['formatting']:.0f}%
• Sections: {scores['section_completeness']:.0f}%
• Readability: {scores['readability']:.0f}%
• Length: {scores['length_appropriateness']:.0f}%
"""
        
        # Strengths and weaknesses
        strengths_text = ""
        if comprehensive_feedback.overall_feedback.strengths:
            strengths_text = "**Strengths:**\n" + "\n".join([f"• {s}" for s in comprehensive_feedback.overall_feedback.strengths])
        
        weaknesses_text = ""
        if comprehensive_feedback.overall_feedback.weaknesses:
            weaknesses_text = "**Areas for Improvement:**\n" + "\n".join([f"• {w}" for w in comprehensive_feedback.overall_feedback.weaknesses])
        
        # Recommendations
        recommendations_text = ""
        if comprehensive_feedback.overall_feedback.recommendations:
            recommendations_text = "**Recommendations:**\n" + "\n".join([f"• {r}" for r in comprehensive_feedback.overall_feedback.recommendations])
        
        # Section-wise analysis
        section_analysis = "**Section-wise Analysis:**\n\n"
        for section in comprehensive_feedback.section_analyses:
            section_analysis += f"**{section.section_name.title()} (Score: {section.score:.0f}/100)**\n"
            section_analysis += f"Feedback: {section.feedback}\n"
            if section.suggestions:
                section_analysis += "Suggestions:\n" + "\n".join([f"• {s}" for s in section.suggestions]) + "\n"
            section_analysis += "\n"
        
        # Missing keywords
        missing_keywords = ""
        if comprehensive_feedback.ats_analysis.missing_keywords:
            missing_keywords = f"**Missing Keywords:** {', '.join(comprehensive_feedback.ats_analysis.missing_keywords[:10])}"
        
        return (
            "✅ AI Analysis Complete!",
            overall_result,
            component_scores,
            strengths_text,
            weaknesses_text,
            recommendations_text,
            section_analysis,
            optimized_resume,
            missing_keywords
        )
        
    except Exception as e:
        return f"❌ Error during AI analysis: {str(e)}", "", "", "", "", "", "", "", ""

def generate_pdf_download(optimized_text, job_role):
    """Generate PDF for download"""
    global pdf_generator
    
    if not optimized_text:
        return None
    
    try:
        # Create temporary PDF file
        temp_pdf_path = f"temp_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_generator.generate_enhanced_pdf(optimized_text, temp_pdf_path)
        
        if os.path.exists(temp_pdf_path):
            return temp_pdf_path
        else:
            return None
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

def get_analysis_history():
    """Get formatted analysis history"""
    if not analysis_history:
        return "No analysis history yet."
    
    history_text = "**Recent Analysis History:**\n\n"
    for i, entry in enumerate(analysis_history[-5:], 1):  # Show last 5 entries
        history_text += f"{i}. **{entry['role']}** - Score: {entry['score']:.1f}% ({entry['timestamp']})\n"
    
    return history_text

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="AI Resume Optimizer with Hugging Face",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            color: #2E4057;
            margin-bottom: 2rem;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🤖 AI Resume Optimizer with Hugging Face
            ### Powered by Meta Llama 3.2 - Get professional resume feedback and optimization!
            
            Upload your resume and get comprehensive AI-powered analysis, ATS scoring, and optimization suggestions.
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Upload Resume")
                
                # File upload
                resume_file = gr.File(
                    label="Upload Resume (PDF or TXT)",
                    file_types=[".pdf", ".txt"],
                    file_count="single"
                )
                
                # Text input alternative
                resume_text = gr.Textbox(
                    label="Or paste your resume text here:",
                    placeholder="Paste your resume content...",
                    lines=10,
                    max_lines=15
                )
                
                gr.Markdown("## ⚙️ Configuration")
                
                # Job role selection
                job_role = gr.Dropdown(
                    choices=["Data Scientist", "Software Engineer", "Product Manager", 
                            "Marketing Manager", "Business Analyst", "Project Manager", 
                            "DevOps Engineer", "UI/UX Designer", "Cybersecurity Analyst", "Custom"],
                    value="Data Scientist",
                    label="Select Job Role"
                )
                
                custom_role = gr.Textbox(
                    label="Custom Role (if selected above)",
                    placeholder="Enter custom job role...",
                    visible=False
                )
                
                # Show custom role input when Custom is selected
                job_role.change(
                    lambda x: gr.update(visible=x=="Custom"),
                    inputs=[job_role],
                    outputs=[custom_role]
                )
                
                # Job description
                job_description = gr.Textbox(
                    label="Job Description (Optional)",
                    placeholder="Paste the job description here for better optimization...",
                    lines=5
                )
                
                # Analyze button
                analyze_btn = gr.Button(
                    "🤖 Analyze Resume with AI",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 📊 AI Analysis Results")
                
                # Status message
                status_msg = gr.Markdown("")
                
                # Overall score
                overall_score = gr.Markdown("")
                
                # Component scores
                component_scores = gr.Markdown("")
                
                # Missing keywords
                missing_keywords = gr.Markdown("")
        
        # Results section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 💡 AI-Powered Feedback & Recommendations")
                
                with gr.Tabs():
                    with gr.TabItem("Overall Feedback"):
                        strengths_output = gr.Markdown("")
                        weaknesses_output = gr.Markdown("")
                        recommendations_output = gr.Markdown("")
                    
                    with gr.TabItem("Section Analysis"):
                        section_analysis_output = gr.Markdown("")
                    
                    with gr.TabItem("Optimized Resume"):
                        optimized_resume_output = gr.Textbox(
                            label="AI-Optimized Resume Content",
                            lines=20,
                            max_lines=25,
                            show_copy_button=True
                        )
                        
                        # Download buttons
                        with gr.Row():
                            download_pdf_btn = gr.Button("📄 Generate PDF Download", variant="secondary")
                            download_txt_btn = gr.DownloadButton("📝 Download as TXT", variant="secondary")
                    
                    with gr.TabItem("Analysis History"):
                        history_output = gr.Markdown("")
                        refresh_history_btn = gr.Button("🔄 Refresh History")
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_resume,
            inputs=[resume_file, resume_text, job_role, custom_role, job_description],
            outputs=[
                status_msg, overall_score, component_scores, 
                strengths_output, weaknesses_output, recommendations_output,
                section_analysis_output, optimized_resume_output, missing_keywords
            ]
        )
        
        # Update download button with optimized text
        optimized_resume_output.change(
            lambda text: gr.update(value=text, filename="optimized_resume.txt") if text else gr.update(),
            inputs=[optimized_resume_output],
            outputs=[download_txt_btn]
        )
        
        # Refresh history
        refresh_history_btn.click(
            fn=get_analysis_history,
            outputs=[history_output]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            <div style='text-align: center; color: #666; padding: 1rem;'>
                <p>🔒 Your data is processed securely and not stored permanently</p>
                <p>Built with ❤️ using Gradio and Hugging Face Transformers</p>
            </div>
            """)
    
    return demo

# Main execution
def main():
    """Main function to run the Gradio app"""
    print("🚀 Starting AI Resume Optimizer with Gradio...")
    print(f"📊 Using model: {Config.HUGGINGFACE_MODEL}")
    
    # Initialize modules
    init_status = initialize_modules()
    print(init_status)
    
    # Create and launch interface
    demo = create_interface()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=True,             # Create public link
        show_error=True,        # Show errors in interface
        debug=True              # Enable debug mode
    )

if __name__ == "__main__":
    main()