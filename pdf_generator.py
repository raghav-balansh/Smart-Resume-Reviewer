from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
import re
import os
from typing import Dict, List
import time

class OptimizedResumePDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_optimized_styles()
        
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'linkedin': re.compile(r'linkedin\.com/in/[\w-]+', re.IGNORECASE),
            'github': re.compile(r'github\.com/[\w-]+', re.IGNORECASE),
            'years': re.compile(r'\b(20\d{2}|19\d{2})\b'),
            'bullets': re.compile(r'^[•\-\*\+]\s*', re.MULTILINE),
            'sections': re.compile(r'^[A-Z\s]{2,}$', re.MULTILINE)
        }
    
    def _setup_optimized_styles(self):
        """Setup essential styles for professional resume"""
        
        # Color scheme
        primary_color = HexColor('#2C3E50')    # Dark blue-gray
        accent_color = HexColor('#3498DB')     # Blue
        text_color = HexColor('#2C3E50')       # Dark gray
        light_gray = HexColor('#7F8C8D')       # Light gray
        
        self.styles.add(ParagraphStyle(
            name='ResumeHeader',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=6,
            textColor=primary_color,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=28
        ))
        
        # Contact information
        self.styles.add(ParagraphStyle(
            name='ContactInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=text_color,
            fontName='Helvetica',
            leading=12
        ))
        
        # Section headers
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=6,
            spaceBefore=12,
            textColor=primary_color,
            fontName='Helvetica-Bold',
            leading=16
        ))
        
        # Normal content
        self.styles.add(ParagraphStyle(
            name='Content',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            fontName='Helvetica',
            textColor=text_color,
            leading=12
        ))
        
        # Bullet points
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=2,
            leftIndent=20,
            fontName='Helvetica',
            textColor=text_color,
            leading=12
        ))
        
        # Job titles
        self.styles.add(ParagraphStyle(
            name='JobTitle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=2,
            fontName='Helvetica-Bold',
            textColor=primary_color,
            leading=13
        ))
        
        # Company/Institution names
        self.styles.add(ParagraphStyle(
            name='Organization',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=1,
            fontName='Helvetica',
            textColor=accent_color,
            leading=12
        ))
        
        # Dates
        self.styles.add(ParagraphStyle(
            name='DateInfo',
            parent=self.styles['Normal'],
            fontSize=9,
            spaceAfter=4,
            fontName='Helvetica-Oblique',
            textColor=light_gray,
            alignment=TA_RIGHT,
            leading=11
        ))
    
    def _extract_basic_info(self, text: str) -> Dict[str, str]:
        info = {}
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:3]:
            if line and len(line) < 50 and not any(char in line for char in ['@', '(', ')', '-', '|']):
                info['name'] = line
                break
        
        if 'name' not in info:
            info['name'] = "Resume"
        
        # Extract contact info using pre-compiled patterns
        email_match = self.patterns['email'].search(text)
        if email_match:
            info['email'] = email_match.group()
        
        phone_match = self.patterns['phone'].search(text)
        if phone_match:
            info['phone'] = phone_match.group()
        
        linkedin_match = self.patterns['linkedin'].search(text)
        if linkedin_match:
            info['linkedin'] = linkedin_match.group()
        
        github_match = self.patterns['github'].search(text)
        if github_match:
            info['github'] = github_match.group()
        
        return info
    
    def _parse_sections_fast(self, text: str) -> Dict[str, str]:
        sections = {}
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        section_keywords = {
            'summary': ['summary', 'profile', 'objective', 'about'],
            'experience': ['experience', 'employment', 'work history', 'professional'],
            'education': ['education', 'academic', 'qualification'],
            'skills': ['skills', 'competencies', 'technical', 'technologies'],
            'projects': ['projects', 'portfolio', 'key projects'],
            'certifications': ['certifications', 'certificates', 'licenses']
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Quick section header detection
            is_section_header = False
            section_name = None
            
            if len(line) < 50 and (line.isupper() or line.istitle()):
                line_lower = line.lower()
                for section, keywords in section_keywords.items():
                    if any(keyword in line_lower for keyword in keywords):
                        is_section_header = True
                        section_name = section
                        break
            
            if is_section_header:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = section_name
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _format_contact_line(self, info: Dict[str, str]) -> str:
        """Format contact information into a single line"""
        contact_parts = []
        
        if 'email' in info:
            contact_parts.append(info['email'])
        if 'phone' in info:
            contact_parts.append(info['phone'])
        if 'linkedin' in info:
            contact_parts.append(f"LinkedIn: {info['linkedin']}")
        if 'github' in info:
            contact_parts.append(f"GitHub: {info['github']}")
        
        return " | ".join(contact_parts)
    
    def _create_experience_elements(self, content: str) -> List:
        """Create formatted experience section elements"""
        elements = []
        
        # Split into job entries (assuming double line breaks separate jobs)
        job_entries = content.split('\n\n')
        
        for entry in job_entries:
            if not entry.strip():
                continue
            
            lines = [line.strip() for line in entry.split('\n') if line.strip()]
            if not lines:
                continue
            
            first_line = lines[0]
            if '|' in first_line:
                parts = [part.strip() for part in first_line.split('|')]
                if len(parts) >= 2:
                    job_title = parts[0]
                    company = parts[1]
                    dates = parts[2] if len(parts) > 2 else ""
                    
                    job_data = [[job_title, dates]] if dates else [[job_title, ""]]
                    job_table = Table(job_data, colWidths=[4*inch, 2*inch])
                    job_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (0, 0), 11),
                        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Oblique'),
                        ('FONTSIZE', (1, 0), (1, 0), 9),
                        ('TEXTCOLOR', (1, 0), (1, 0), HexColor('#7F8C8D')),
                    ]))
                    elements.append(job_table)
                    
                    # Add company
                    elements.append(Paragraph(company, self.styles['Organization']))
                    elements.append(Spacer(1, 2))
            else:
                # Simple job title
                elements.append(Paragraph(first_line, self.styles['JobTitle']))
            
            # Process remaining lines (bullet points)
            for line in lines[1:]:
                if line.startswith(('•', '-', '*', '+')):
                    clean_line = re.sub(r'^[•\-\*\+]\s*', '', line)
                    elements.append(Paragraph(f"• {clean_line}", self.styles['BulletPoint']))
                else:
                    elements.append(Paragraph(line, self.styles['Content']))
            
            elements.append(Spacer(1, 6))
        
        return elements
    
    def _create_skills_elements(self, content: str) -> List:
        """Create formatted skills section"""
        elements = []
        
        # Parse skills (comma, semicolon, or newline separated)
        skills_text = content.replace('\n', ', ')
        skills = [skill.strip() for skill in re.split(r'[,;]', skills_text) if skill.strip()]
        
        if not skills:
            elements.append(Paragraph("Skills section needs content", self.styles['Content']))
            return elements
        
        # Group skills into rows for better layout (3 per row)
        skill_rows = []
        for i in range(0, len(skills), 3):
            row = skills[i:i+3]
            while len(row) < 3:  # Pad with empty strings
                row.append("")
            skill_rows.append(row)
        
        # Create skills table
        skills_table = Table(skill_rows, colWidths=[2*inch, 2*inch, 2*inch])
        skills_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        
        elements.append(skills_table)
        return elements
    
    def _create_education_elements(self, content: str) -> List:
        """Create formatted education section"""
        elements = []
        
        entries = content.split('\n\n') if '\n\n' in content else [content]
        
        for entry in entries:
            if not entry.strip():
                continue
            
            lines = [line.strip() for line in entry.split('\n') if line.strip()]
            
            for line in lines:
                if '|' in line:
                    parts = [part.strip() for part in line.split('|')]
                    if len(parts) >= 2:
                        degree = parts[0]
                        school = parts[1]
                        date = parts[2] if len(parts) > 2 else ""
                        
                        # Create education table
                        edu_data = [[degree, date]] if date else [[degree, ""]]
                        edu_table = Table(edu_data, colWidths=[4*inch, 2*inch])
                        edu_table.setStyle(TableStyle([
                            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (0, 0), 11),
                            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Oblique'),
                            ('FONTSIZE', (1, 0), (1, 0), 9),
                            ('TEXTCOLOR', (1, 0), (1, 0), HexColor('#7F8C8D')),
                        ]))
                        elements.append(edu_table)
                        elements.append(Paragraph(school, self.styles['Organization']))
                        elements.append(Spacer(1, 4))
                else:
                    elements.append(Paragraph(line, self.styles['Content']))
        
        return elements
    
    def _create_generic_section_elements(self, content: str) -> List:
        """Create elements for generic sections"""
        elements = []
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            if line.startswith(('•', '-', '*', '+')):
                clean_line = re.sub(r'^[•\-\*\+]\s*', '', line)
                elements.append(Paragraph(f"• {clean_line}", self.styles['BulletPoint']))
            else:
                elements.append(Paragraph(line, self.styles['Content']))
        
        return elements
    
    def generate_pdf(self, resume_text: str, output_path: str) -> str:
        """Generate PDF with optimized processing"""
        
        start_time = time.time()
        
        try:
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Parse content
            basic_info = self._extract_basic_info(resume_text)
            sections = self._parse_sections_fast(resume_text)
            
            # Build story
            story = []
            
            # Header
            story.append(Paragraph(basic_info['name'], self.styles['ResumeHeader']))
            
            # Contact info
            contact_line = self._format_contact_line(basic_info)
            if contact_line:
                story.append(Paragraph(contact_line, self.styles['ContactInfo']))
            
            story.append(Spacer(1, 8))
            
            # Sections in optimal order
            section_order = ['summary', 'experience', 'education', 'skills', 'projects', 'certifications']
            
            for section_name in section_order:
                if section_name in sections and sections[section_name].strip():
                    # Add section header
                    story.append(Paragraph(section_name.upper(), self.styles['SectionHeader']))
                    
                    # Add section content based on type
                    if section_name == 'experience':
                        story.extend(self._create_experience_elements(sections[section_name]))
                    elif section_name == 'skills':
                        story.extend(self._create_skills_elements(sections[section_name]))
                    elif section_name == 'education':
                        story.extend(self._create_education_elements(sections[section_name]))
                    else:
                        story.extend(self._create_generic_section_elements(sections[section_name]))
                    
                    story.append(Spacer(1, 8))
            
            # Build PDF
            doc.build(story)
            
            processing_time = time.time() - start_time
            print(f"📄 PDF generated in {processing_time:.2f} seconds")
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating PDF: {str(e)}")
    
    def generate_enhanced_pdf(self, resume_text: str, output_path: str) -> str:
        """Generate enhanced PDF (same as regular for optimization)"""
        return self.generate_pdf(resume_text, output_path)

# Performance testing
def test_pdf_generation_performance():
    """Test PDF generation speed"""
    
    generator = OptimizedResumePDFGenerator()
    
    sample_resume = """
    Alex Johnson
    alex.johnson@email.com | (555) 123-4567 | linkedin.com/in/alexjohnson
    
    SUMMARY
    Software Engineer with 6+ years of experience in full-stack development.
    Specialized in React, Node.js, and cloud technologies. Strong background
    in agile development and team leadership.
    
    EXPERIENCE
    Senior Software Engineer | TechCorp | 2021-2024
    • Led development of microservices architecture serving 1M+ users
    • Improved application performance by 45% through optimization
    • Mentored 3 junior developers and conducted code reviews
    • Implemented CI/CD pipelines reducing deployment time by 70%
    
    Software Engineer | StartupXYZ | 2018-2021
    • Developed RESTful APIs using Node.js and Express
    • Built responsive web applications with React and Redux
    • Collaborated with product team on feature requirements
    • Maintained 99.9% uptime for critical production systems
    
    EDUCATION
    Bachelor of Science in Computer Science | MIT | 2018
    Relevant Coursework: Data Structures, Algorithms, Software Engineering
    
    SKILLS
    Programming Languages: JavaScript, Python, Java, TypeScript
    Frontend: React, Vue.js, HTML5, CSS3, Bootstrap
    Backend: Node.js, Express, Django, Spring Boot
    Databases: PostgreSQL, MongoDB, Redis, MySQL
    Cloud: AWS, Docker, Kubernetes, Jenkins
    Tools: Git, GitHub, Jira, Slack, VS Code
    
    PROJECTS
    E-commerce Platform | 2023
    • Built full-stack application with React and Node.js
    • Integrated Stripe payment processing and inventory management
    • Deployed on AWS with auto-scaling and load balancing
    
    Task Management App | 2022
    • Developed mobile-responsive web app with real-time updates
    • Implemented WebSocket connections for live collaboration
    • Used MongoDB for data persistence and caching
    
    CERTIFICATIONS
    • AWS Certified Solutions Architect - Associate
    • Google Cloud Professional Developer
    • Certified Scrum Master (CSM)
    """
    
    print("🔄 Testing OptimizedResumePDFGenerator performance...")
    
    # Test PDF generation speed
    start_time = time.time()
    output_path = "test_optimized_resume.pdf"
    
    try:
        result_path = generator.generate_pdf(sample_resume, output_path)
        generation_time = time.time() - start_time
        
        print(f"✅ PDF generated successfully in {generation_time:.3f} seconds")
        print(f"📄 Output file: {result_path}")
        
        # Check file size
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path)
            print(f"📊 File size: {file_size / 1024:.1f} KB")
        
        # Test multiple generations (for caching performance)
        print(f"\n🔄 Testing multiple generations...")
        times = []
        for i in range(3):
            start = time.time()
            generator.generate_pdf(sample_resume, f"test_resume_{i}.pdf")
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"📈 Average generation time over 3 runs: {avg_time:.3f}s")
        
        # Cleanup test files
        for i in range(3):
            test_file = f"test_resume_{i}.pdf"
            if os.path.exists(test_file):
                os.remove(test_file)
        
        if os.path.exists(output_path):
            print(f"📁 Test file kept: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_pdf_generation_performance()