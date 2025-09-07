"""
PDF Generation Module
Creates professional PDF resumes with enhanced formatting
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, darkblue, grey, HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics import renderPDF
import io
import os
from typing import Dict, List, Optional
from datetime import datetime
import re

class ResumePDFGenerator:
    """Generate professional PDF resumes"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for professional resume template"""
        # Professional color scheme
        primary_color = HexColor('#2E4057')  # Dark blue-gray
        accent_color = HexColor('#048A81')   # Teal
        text_color = HexColor('#2C3E50')     # Dark gray
        light_gray = HexColor('#7F8C8D')     # Light gray
        
        # Header style - Large, bold name
        self.styles.add(ParagraphStyle(
            name='CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=28,
            spaceAfter=8,
            textColor=primary_color,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=32
        ))
        
        # Contact info style - Centered, professional
        self.styles.add(ParagraphStyle(
            name='ContactInfo',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=16,
            alignment=TA_CENTER,
            textColor=text_color,
            fontName='Helvetica',
            leading=14
        ))
        
        # Section header style - Professional with accent line
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=8,
            spaceBefore=16,
            textColor=primary_color,
            fontName='Helvetica-Bold',
            leading=18,
            borderWidth=0,
            borderColor=accent_color,
            borderPadding=0
        ))
        
        # Enhanced section header with underline
        self.styles.add(ParagraphStyle(
            name='SectionHeaderUnderline',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=8,
            spaceBefore=16,
            textColor=primary_color,
            fontName='Helvetica-Bold',
            leading=18,
            borderWidth=0,
            borderColor=accent_color,
            borderPadding=0
        ))
        
        # Content style - Clean and readable
        self.styles.add(ParagraphStyle(
            name='Content',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            alignment=TA_LEFT,
            fontName='Helvetica',
            textColor=text_color,
            leading=14
        ))
        
        # Bullet point style - Professional indentation
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=3,
            leftIndent=24,
            fontName='Helvetica',
            textColor=text_color,
            leading=14
        ))
        
        # Job title style - Bold and prominent
        self.styles.add(ParagraphStyle(
            name='JobTitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=2,
            fontName='Helvetica-Bold',
            textColor=primary_color,
            leading=14
        ))
        
        # Company style - Accent color
        self.styles.add(ParagraphStyle(
            name='Company',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=2,
            fontName='Helvetica',
            textColor=accent_color,
            leading=14
        ))
        
        # Date style - Right aligned, subtle
        self.styles.add(ParagraphStyle(
            name='Date',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            fontName='Helvetica-Oblique',
            textColor=light_gray,
            alignment=TA_RIGHT,
            leading=12
        ))
        
        # Skills style - Clean layout
        self.styles.add(ParagraphStyle(
            name='Skills',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=3,
            fontName='Helvetica',
            textColor=text_color,
            leading=14
        ))
        
        # Summary/Objective style
        self.styles.add(ParagraphStyle(
            name='Summary',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            textColor=text_color,
            leading=16
        ))

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume text"""
        import re
        
        contact_info = {}
        
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Phone
        phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group()
        
        # GitHub
        github_match = re.search(r'github\.com/[\w-]+', text, re.IGNORECASE)
        if github_match:
            contact_info['github'] = github_match.group()
        
        return contact_info

    def _extract_name(self, text: str) -> str:
        """Extract name from resume text (usually first line)"""
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if line and len(line) < 50 and not any(char.isdigit() for char in line):
                # Likely a name if it's short, no numbers, and not empty
                return line
        return "Resume"

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Parse resume into sections"""
        sections = {}
        current_section = None
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a section header
            section_keywords = [
                'summary', 'profile', 'objective', 'about',
                'experience', 'work history', 'employment',
                'education', 'academic', 'qualifications',
                'skills', 'technical skills', 'competencies',
                'projects', 'portfolio', 'key projects',
                'certifications', 'certificates', 'licenses',
                'achievements', 'awards', 'honors',
                'languages', 'language skills',
                'interests', 'hobbies', 'activities'
            ]
            
            is_section_header = False
            for keyword in section_keywords:
                if keyword.lower() in line.lower() and len(line) < 50:
                    is_section_header = True
                    break
            
            if is_section_header:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = line.lower()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
                else:
                    # Content before any section header
                    if 'header' not in sections:
                        sections['header'] = []
                    sections['header'].append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

    def _format_contact_info(self, contact_info: Dict[str, str]) -> str:
        """Format contact information for display"""
        contact_parts = []
        
        if 'email' in contact_info:
            contact_parts.append(contact_info['email'])
        if 'phone' in contact_info:
            contact_parts.append(contact_info['phone'])
        if 'linkedin' in contact_info:
            contact_parts.append(f"LinkedIn: {contact_info['linkedin']}")
        if 'github' in contact_info:
            contact_parts.append(f"GitHub: {contact_info['github']}")
        
        return " | ".join(contact_parts)

    def _create_skills_table(self, skills_text: str) -> Table:
        """Create a professionally formatted skills table"""
        # Parse skills (assuming comma or line separated)
        skills = []
        for skill in skills_text.replace('\n', ',').split(','):
            skill = skill.strip()
            if skill:
                skills.append(skill)
        
        if not skills:
            return Paragraph("No skills listed", self.styles['Content'])
        
        # Group skills into rows of 3 for better layout
        skill_rows = []
        for i in range(0, len(skills), 3):
            row = skills[i:i+3]
            # Pad with empty strings if needed
            while len(row) < 3:
                row.append("")
            skill_rows.append(row)
        
        table = Table(skill_rows, colWidths=[2*inch, 2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        return table
    
    def _create_section_header_with_line(self, title: str) -> List:
        """Create a section header with decorative line"""
        elements = []
        
        # Add section title
        elements.append(Paragraph(title.upper(), self.styles['SectionHeader']))
        
        # Add decorative line
        drawing = Drawing(6*inch, 0.1*inch)
        drawing.add(Line(0, 0.05*inch, 6*inch, 0.05*inch, strokeColor=HexColor('#048A81'), strokeWidth=2))
        elements.append(drawing)
        elements.append(Spacer(1, 8))
        
        return elements
    
    def _format_experience_entry(self, entry_text: str) -> List:
        """Format experience entries professionally"""
        elements = []
        lines = entry_text.split('\n')
        
        if not lines:
            return elements
        
        # First line is usually job title and company
        first_line = lines[0].strip()
        if '|' in first_line:
            parts = first_line.split('|')
            if len(parts) >= 2:
                job_title = parts[0].strip()
                company = parts[1].strip()
                date = parts[2].strip() if len(parts) > 2 else ""
                
                # Create table for job title, company, and date
                job_data = [[job_title, date]]
                job_table = Table(job_data, colWidths=[4*inch, 2*inch])
                job_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (0, 0), 12),
                    ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Oblique'),
                    ('FONTSIZE', (1, 0), (1, 0), 10),
                    ('TEXTCOLOR', (0, 0), (0, 0), HexColor('#2E4057')),
                    ('TEXTCOLOR', (1, 0), (1, 0), HexColor('#7F8C8D')),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ]))
                elements.append(job_table)
                
                # Company name
                elements.append(Paragraph(company, self.styles['Company']))
                elements.append(Spacer(1, 4))
        
        # Process bullet points
        for line in lines[1:]:
            line = line.strip()
            if line:
                if line.startswith(('•', '-', '*')):
                    # Clean bullet point
                    clean_line = line.lstrip('•-*').strip()
                    elements.append(Paragraph(f"• {clean_line}", self.styles['BulletPoint']))
                else:
                    elements.append(Paragraph(line, self.styles['Content']))
        
        elements.append(Spacer(1, 8))
        return elements
    
    def _format_education_entry(self, entry_text: str) -> List:
        """Format education entries professionally"""
        elements = []
        lines = entry_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line:
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        degree = parts[0].strip()
                        school = parts[1].strip()
                        date = parts[2].strip() if len(parts) > 2 else ""
                        
                        # Create table for degree and date
                        edu_data = [[degree, date]]
                        edu_table = Table(edu_data, colWidths=[4*inch, 2*inch])
                        edu_table.setStyle(TableStyle([
                            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (0, 0), 12),
                            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Oblique'),
                            ('FONTSIZE', (1, 0), (1, 0), 10),
                            ('TEXTCOLOR', (0, 0), (0, 0), HexColor('#2E4057')),
                            ('TEXTCOLOR', (1, 0), (1, 0), HexColor('#7F8C8D')),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                        ]))
                        elements.append(edu_table)
                        elements.append(Paragraph(school, self.styles['Company']))
                        elements.append(Spacer(1, 4))
                else:
                    elements.append(Paragraph(line, self.styles['Content']))
        
        return elements

    def generate_pdf(self, resume_text: str, output_path: str, enhanced: bool = False) -> str:
        """Generate professionally formatted PDF resume"""
        try:
            # Create PDF document with professional margins
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Parse resume content
            sections = self._parse_sections(resume_text)
            contact_info = self._extract_contact_info(resume_text)
            name = self._extract_name(resume_text)
            
            # Build PDF content
            story = []
            
            # Header with name
            story.append(Paragraph(name, self.styles['CustomHeader']))
            story.append(Spacer(1, 8))
            
            # Contact information
            if contact_info:
                contact_text = self._format_contact_info(contact_info)
                story.append(Paragraph(contact_text, self.styles['ContactInfo']))
                story.append(Spacer(1, 16))
            
            # Process sections in professional order
            section_order = [
                'summary', 'profile', 'objective', 'about',
                'experience', 'work history', 'employment',
                'education', 'academic', 'qualifications',
                'skills', 'technical skills', 'competencies',
                'projects', 'portfolio', 'key projects',
                'certifications', 'certificates', 'licenses',
                'achievements', 'awards', 'honors',
                'languages', 'language skills',
                'interests', 'hobbies', 'activities'
            ]
            
            for section_name in section_order:
                if section_name in sections:
                    content = sections[section_name]
                    if content.strip():
                        # Add section header with decorative line
                        story.extend(self._create_section_header_with_line(section_name.title()))
                        
                        # Section content with enhanced formatting
                        if section_name in ['skills', 'technical skills', 'competencies']:
                            # Special formatting for skills
                            story.append(self._create_skills_table(content))
                        elif section_name in ['experience', 'work history', 'employment']:
                            # Enhanced experience formatting
                            experience_entries = content.split('\n\n')
                            for entry in experience_entries:
                                if entry.strip():
                                    story.extend(self._format_experience_entry(entry))
                        elif section_name in ['education', 'academic', 'qualifications']:
                            # Enhanced education formatting
                            education_entries = content.split('\n\n')
                            for entry in education_entries:
                                if entry.strip():
                                    story.extend(self._format_education_entry(entry))
                        elif section_name in ['summary', 'profile', 'objective', 'about']:
                            # Summary/Objective formatting
                            story.append(Paragraph(content.strip(), self.styles['Summary']))
                        else:
                            # Regular content formatting
                            lines = content.split('\n')
                            for line in lines:
                                line = line.strip()
                                if line:
                                    if line.startswith(('•', '-', '*')):
                                        # Bullet point
                                        clean_line = line.lstrip('•-*').strip()
                                        story.append(Paragraph(f"• {clean_line}", self.styles['BulletPoint']))
                                    else:
                                        # Regular content
                                        story.append(Paragraph(line, self.styles['Content']))
                        
                        story.append(Spacer(1, 12))
            
            # Add enhanced resume watermark if enhanced
            if enhanced:
                story.append(Spacer(1, 20))
                timestamp = datetime.now().strftime("Enhanced Resume - Generated on %B %d, %Y")
                story.append(Paragraph(timestamp, self.styles['Date']))
            
            # Build PDF
            doc.build(story)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating PDF: {str(e)}")
    
    def generate_enhanced_pdf(self, resume_text: str, output_path: str) -> str:
        """Generate enhanced PDF with professional template"""
        return self.generate_pdf(resume_text, output_path, enhanced=True)

    def generate_feedback_pdf(self, feedback_data: Dict, output_path: str) -> str:
        """Generate PDF with resume feedback"""
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            story.append(Paragraph("Resume Review Report", self.styles['CustomHeader']))
            story.append(Spacer(1, 12))
            
            # Overall Score
            if 'ats_metrics' in feedback_data:
                ats_metrics = feedback_data['ats_metrics']
                score_text = f"Overall ATS Score: {ats_metrics.overall_score:.1f}/100"
                story.append(Paragraph(score_text, self.styles['SectionHeader']))
                story.append(Spacer(1, 6))
                
                # Score breakdown
                score_table_data = [
                    ['Metric', 'Score'],
                    ['Keyword Score', f"{ats_metrics.keyword_score:.1f}/100"],
                    ['Formatting Score', f"{ats_metrics.formatting_score:.1f}/100"],
                    ['Readability Score', f"{ats_metrics.readability_score:.1f}/100"],
                    ['Section Score', f"{ats_metrics.section_score:.1f}/100"]
                ]
                
                score_table = Table(score_table_data, colWidths=[3*inch, 2*inch])
                score_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(score_table)
                story.append(Spacer(1, 12))
            
            # Overall Feedback
            if 'overall_feedback' in feedback_data:
                overall = feedback_data['overall_feedback']
                
                # Strengths
                story.append(Paragraph("Strengths", self.styles['SectionHeader']))
                for strength in overall.strengths:
                    story.append(Paragraph(f"• {strength}", self.styles['BulletPoint']))
                story.append(Spacer(1, 6))
                
                # Areas for Improvement
                story.append(Paragraph("Areas for Improvement", self.styles['SectionHeader']))
                for weakness in overall.weaknesses:
                    story.append(Paragraph(f"• {weakness}", self.styles['BulletPoint']))
                story.append(Spacer(1, 6))
                
                # Recommendations
                story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
                for rec in overall.recommendations:
                    story.append(Paragraph(f"• {rec}", self.styles['BulletPoint']))
                story.append(Spacer(1, 6))
            
            # Section-wise Feedback
            if 'section_feedbacks' in feedback_data:
                story.append(Paragraph("Section-wise Feedback", self.styles['SectionHeader']))
                for section_feedback in feedback_data['section_feedbacks']:
                    story.append(Paragraph(f"{section_feedback.section.title()} (Score: {section_feedback.score}/100)", 
                                         self.styles['JobTitle']))
                    story.append(Paragraph(section_feedback.feedback, self.styles['Content']))
                    story.append(Spacer(1, 6))
            
            # Missing Keywords
            if 'ats_metrics' in feedback_data and feedback_data['ats_metrics'].missing_keywords:
                story.append(Paragraph("Missing Keywords", self.styles['SectionHeader']))
                missing_kw = ", ".join(feedback_data['ats_metrics'].missing_keywords[:10])  # Limit to 10
                story.append(Paragraph(missing_kw, self.styles['Content']))
                story.append(Spacer(1, 6))
            
            # Timestamp
            timestamp = datetime.now().strftime("Report Generated on %B %d, %Y at %I:%M %p")
            story.append(Paragraph(timestamp, self.styles['Date']))
            
            # Build PDF
            doc.build(story)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating feedback PDF: {str(e)}")

# Example usage
if __name__ == "__main__":
    generator = ResumePDFGenerator()
    
    sample_resume = """
    John Doe
    john.doe@email.com | (555) 123-4567 | LinkedIn: linkedin.com/in/johndoe
    
    SUMMARY
    Experienced software engineer with 5+ years in full-stack development.
    Passionate about creating scalable web applications and leading development teams.
    
    EXPERIENCE
    Senior Software Engineer | Tech Corp | 2020-2023
    • Developed web applications using Python and React
    • Led team of 3 developers
    • Implemented CI/CD pipelines
    
    Software Engineer | StartupXYZ | 2018-2020
    • Built REST APIs using Django
    • Collaborated with cross-functional teams
    
    EDUCATION
    Bachelor of Computer Science | University of Tech | 2018
    
    SKILLS
    Python, JavaScript, React, SQL, AWS, Docker, Git, Agile
    """
    
    try:
        output_path = generator.generate_pdf(sample_resume, "sample_resume.pdf")
        print(f"PDF generated successfully: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
