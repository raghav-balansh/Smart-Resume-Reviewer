from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from hf_analyzer import HuggingFaceAnalyzer, SectionFeedback, OverallFeedback
from resume_processor import ResumeProcessor, ATSAnalysis

@dataclass
class SectionAnalysis:
    section_name: str
    content: str
    score: float
    feedback: str
    suggestions: List[str]
    missing_keywords: List[str]
    improvements: List[str]

@dataclass
class ComprehensiveFeedback:
    overall_feedback: OverallFeedback
    section_analyses: List[SectionAnalysis]
    ats_analysis: ATSAnalysis
    priority_improvements: List[str]
    estimated_improvement: float

class SectionWiseFeedbackGenerator:
    #Generate detailed section-wise feedback for resumes
    
    def __init__(self, hf_analyzer: HuggingFaceAnalyzer = None):
        self.hf_analyzer = hf_analyzer or HuggingFaceAnalyzer()
        self.resume_processor = ResumeProcessor()
        
        # Section-specific improvement guidelines
        self.section_guidelines = {
            'summary': {
                'keywords': ['experienced', 'skilled', 'passionate', 'results-driven', 'innovative'],
                'length_range': (2, 6), 
                'focus_areas': ['value proposition', 'key skills', 'career highlights']
            },
            'experience': {
                'keywords': ['achieved', 'increased', 'improved', 'developed', 'led', 'managed'],
                'focus_areas': ['quantified achievements', 'action verbs', 'relevant technologies'],
                'structure': ['job title', 'company', 'dates', 'bullet points']
            },
            'skills': {
                'categories': ['technical', 'soft', 'tools', 'languages'],
                'focus_areas': ['relevance to role', 'proficiency levels', 'recent technologies']
            },
            'education': {
                'focus_areas': ['degree relevance', 'graduation date', 'academic achievements'],
                'structure': ['degree', 'institution', 'date', 'gpa (if high)']
            },
            'projects': {
                'keywords': ['built', 'developed', 'implemented', 'designed', 'created'],
                'focus_areas': ['problem solved', 'technologies used', 'impact/results'],
                'structure': ['project name', 'description', 'technologies', 'outcome']
            }
        }
    
    def generate_comprehensive_feedback(self, resume_text: str, job_role: str, job_description: str = "") -> ComprehensiveFeedback:
        """Generate comprehensive section-wise feedback"""
        
        # Get overall feedback from HF analyzer
        overall_feedback = self.hf_analyzer.analyze_overall(resume_text, job_role)
        
        # Get ATS analysis
        ats_analysis = self.resume_processor.comprehensive_ats_analysis(resume_text, job_role, job_description)
        
        # Extract sections
        sections = self.resume_processor.extract_sections(resume_text)
        
        # Generate section-wise analysis
        section_analyses = []
        for section_name, section_content in sections.items():
            if section_content.strip():
                section_analysis = self._analyze_section(section_name, section_content, job_role, resume_text)
                section_analyses.append(section_analysis)
        
        # Generate priority improvements
        priority_improvements = self._generate_priority_improvements(section_analyses, ats_analysis)
        
        # Estimate improvement potential
        estimated_improvement = self._estimate_improvement_potential(section_analyses, ats_analysis)
        
        return ComprehensiveFeedback(
            overall_feedback=overall_feedback,
            section_analyses=section_analyses,
            ats_analysis=ats_analysis,
            priority_improvements=priority_improvements,
            estimated_improvement=estimated_improvement
        )
    
    def _analyze_section(self, section_name: str, section_content: str, job_role: str, full_resume: str) -> SectionAnalysis:
        """Analyze a specific section in detail"""
        
        # Get HF model feedback for this section
        hf_feedback = self.hf_analyzer.analyze_section(full_resume, job_role, section_name)
        
        # Calculate section-specific metrics
        score = self._calculate_section_score(section_name, section_content, job_role)
        
        # Generate specific feedback
        feedback = self._generate_section_feedback(section_name, section_content, job_role, hf_feedback)
        
        # Generate suggestions
        suggestions = self._generate_section_suggestions(section_name, section_content, job_role)
        
        # Find missing keywords
        missing_keywords = self._find_missing_keywords(section_name, section_content, job_role)
        
        # Generate improvements
        improvements = self._generate_improvements(section_name, section_content, job_role)
        
        return SectionAnalysis(
            section_name=section_name,
            content=section_content,
            score=score,
            feedback=feedback,
            suggestions=suggestions,
            missing_keywords=missing_keywords,
            improvements=improvements
        )
    
    def _calculate_section_score(self, section_name: str, content: str, job_role: str) -> float:
        """Calculate score for a specific section"""
        score = 0.0
        
        if section_name == 'summary':
            score = self._score_summary(content, job_role)
        elif section_name in ['experience', 'work history', 'employment']:
            score = self._score_experience(content, job_role)
        elif section_name in ['skills', 'technical skills', 'competencies']:
            score = self._score_skills(content, job_role)
        elif section_name in ['education', 'academic', 'qualifications']:
            score = self._score_education(content, job_role)
        elif section_name in ['projects', 'portfolio', 'key projects']:
            score = self._score_projects(content, job_role)
        else:
            score = self._score_generic_section(content, job_role)
        
        return min(score, 100.0)
    
    def _score_summary(self, content: str, job_role: str) -> float:
        """Score the summary/objective section"""
        score = 0.0
        
        # Length check (2-4 sentences ideal)
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        if 2 <= sentence_count <= 4:
            score += 30
        elif sentence_count == 1 or sentence_count == 5:
            score += 20
        else:
            score += 10
        
        # Keyword presence
        keywords = self.section_guidelines['summary']['keywords']
        found_keywords = sum(1 for kw in keywords if kw.lower() in content.lower())
        score += min(found_keywords * 10, 30)
        
        # Value proposition check
        value_words = ['experienced', 'skilled', 'expertise', 'specialized', 'proven']
        if any(word in content.lower() for word in value_words):
            score += 20
        
        # Career highlights
        highlight_words = ['years', 'experience', 'successful', 'achieved', 'led']
        if any(word in content.lower() for word in highlight_words):
            score += 20
        
        return score
    
    def _score_experience(self, content: str, job_role: str) -> float:
        """Score the experience section"""
        score = 0.0
        
        # Check for quantified achievements
        numbers = re.findall(r'\b\d+%|\b\d+\+|\b\d+[km]|\$\d+', content.lower())
        if numbers:
            score += 25
        
        # Action verbs
        action_verbs = self.section_guidelines['experience']['keywords']
        found_verbs = sum(1 for verb in action_verbs if verb.lower() in content.lower())
        score += min(found_verbs * 5, 25)
        
        # Job structure (title, company, dates)
        if '|' in content or any(year in content for year in ['2020', '2021', '2022', '2023', '2024']):
            score += 20
        
        # Bullet points
        bullet_count = len(re.findall(r'[•\-\*]', content))
        if bullet_count >= 3:
            score += 15
        elif bullet_count >= 1:
            score += 10
        
        # Technology relevance
        tech_keywords = self.resume_processor.role_keywords.get(job_role.lower().replace(' ', '_'), [])
        found_tech = sum(1 for tech in tech_keywords if tech.lower() in content.lower())
        score += min(found_tech * 3, 15)
        
        return score
    
    def _score_skills(self, content: str, job_role: str) -> float:
        """Score the skills section"""
        score = 0.0
        
        # Skill count (10-20 skills ideal)
        skills = [s.strip() for s in re.split(r'[,;\n]', content) if s.strip()]
        skill_count = len(skills)
        if 10 <= skill_count <= 20:
            score += 30
        elif 5 <= skill_count < 10 or 20 < skill_count <= 30:
            score += 20
        else:
            score += 10
        
        # Technology relevance
        tech_keywords = self.resume_processor.role_keywords.get(job_role.lower().replace(' ', '_'), [])
        found_tech = sum(1 for tech in tech_keywords if tech.lower() in content.lower())
        score += min(found_tech * 5, 40)
        
        # Skill categorization
        categories = ['programming', 'tools', 'frameworks', 'databases', 'cloud']
        found_categories = sum(1 for cat in categories if any(skill.lower() in content.lower() for skill in [cat]))
        score += found_categories * 6
        
        return score
    
    def _score_education(self, content: str, job_role: str) -> float:
        """Score the education section"""
        score = 0.0
        
        # Degree presence
        degrees = ['bachelor', 'master', 'phd', 'associate', 'certificate']
        if any(degree in content.lower() for degree in degrees):
            score += 30
        
        # Institution name
        if any(word in content.lower() for word in ['university', 'college', 'institute', 'school']):
            score += 20
        
        # Graduation date
        if any(year in content for year in ['2020', '2021', '2022', '2023', '2024', '2019', '2018']):
            score += 20
        
        # GPA (if mentioned and good)
        gpa_match = re.search(r'gpa[:\s]*(\d+\.?\d*)', content.lower())
        if gpa_match:
            gpa = float(gpa_match.group(1))
            if gpa >= 3.5:
                score += 15
            elif gpa >= 3.0:
                score += 10
        
        # Academic achievements
        if any(word in content.lower() for word in ['honors', 'dean', 'magna', 'summa', 'cum laude']):
            score += 15
        
        return score
    
    def _score_projects(self, content: str, job_role: str) -> float:
        """Score the projects section"""
        score = 0.0
        
        # Project count (2-4 projects ideal)
        project_indicators = re.findall(r'project|built|developed|created|designed', content.lower())
        if 2 <= len(project_indicators) <= 4:
            score += 25
        elif len(project_indicators) >= 1:
            score += 15
        
        # Technology usage
        tech_keywords = self.resume_processor.role_keywords.get(job_role.lower().replace(' ', '_'), [])
        found_tech = sum(1 for tech in tech_keywords if tech.lower() in content.lower())
        score += min(found_tech * 5, 30)
        
        # Impact/results
        impact_words = ['improved', 'increased', 'reduced', 'saved', 'achieved', 'resulted']
        if any(word in content.lower() for word in impact_words):
            score += 20
        
        # Project descriptions
        if len(content.split('\n')) >= 3:
            score += 15
        
        # Links/repositories
        if any(link in content.lower() for link in ['github', 'demo', 'link', 'url']):
            score += 10
        
        return score
    
    def _score_generic_section(self, content: str, job_role: str) -> float:
        """Score generic sections"""
        score = 0.0
        
        # Content length
        word_count = len(content.split())
        if 20 <= word_count <= 100:
            score += 40
        elif 10 <= word_count < 20 or 100 < word_count <= 200:
            score += 30
        else:
            score += 20
        
        # Structure
        if '\n' in content or '•' in content:
            score += 30
        
        # Relevance
        tech_keywords = self.resume_processor.role_keywords.get(job_role.lower().replace(' ', '_'), [])
        found_tech = sum(1 for tech in tech_keywords if tech.lower() in content.lower())
        score += min(found_tech * 5, 30)
        
        return score
    
    def _generate_section_feedback(self, section_name: str, content: str, job_role: str, hf_feedback: SectionFeedback) -> str:
        """Generate detailed feedback for a section"""
        feedback_parts = []
        
        # Use HF model feedback if available
        if hf_feedback.feedback:
            feedback_parts.append(hf_feedback.feedback)
        
        # Add specific section feedback
        if section_name == 'summary':
            feedback_parts.append(self._get_summary_feedback(content))
        elif section_name in ['experience', 'work history', 'employment']:
            feedback_parts.append(self._get_experience_feedback(content))
        elif section_name in ['skills', 'technical skills', 'competencies']:
            feedback_parts.append(self._get_skills_feedback(content, job_role))
        elif section_name in ['education', 'academic', 'qualifications']:
            feedback_parts.append(self._get_education_feedback(content))
        elif section_name in ['projects', 'portfolio', 'key projects']:
            feedback_parts.append(self._get_projects_feedback(content, job_role))
        
        return " ".join(feedback_parts) if feedback_parts else "Section content is present but could be enhanced."
    
    def _get_summary_feedback(self, content: str) -> str:
        """Get specific feedback for summary section"""
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count < 2:
            return "Summary is too brief. Consider adding 1-2 more sentences to highlight your key strengths and career focus."
        elif sentence_count > 4:
            return "Summary is too long. Consider condensing to 2-3 impactful sentences that capture your value proposition."
        else:
            return "Summary length is appropriate. Good job highlighting your professional profile."
    
    def _get_experience_feedback(self, content: str) -> str:
        """Get specific feedback for experience section"""
        feedback = []
        
        # Check for quantified achievements
        numbers = re.findall(r'\b\d+%|\b\d+\+|\b\d+[km]|\$\d+', content.lower())
        if not numbers:
            feedback.append("Add quantified achievements (percentages, numbers, dollar amounts) to demonstrate impact.")
        
        # Check for action verbs
        action_verbs = ['achieved', 'increased', 'improved', 'developed', 'led', 'managed']
        found_verbs = sum(1 for verb in action_verbs if verb.lower() in content.lower())
        if found_verbs < 3:
            feedback.append("Use more strong action verbs to describe your accomplishments.")
        
        return " ".join(feedback) if feedback else "Experience section shows good structure and accomplishments."
    
    def _get_skills_feedback(self, content: str, job_role: str) -> str:
        """Get specific feedback for skills section"""
        skills = [s.strip() for s in re.split(r'[,;\n]', content) if s.strip()]
        skill_count = len(skills)
        
        if skill_count < 10:
            return "Consider adding more relevant skills to strengthen your profile."
        elif skill_count > 25:
            return "Skills list is quite long. Consider focusing on the most relevant and recent skills."
        else:
            return "Good variety of skills listed. Consider organizing them by category for better readability."
    
    def _get_education_feedback(self, content: str) -> str:
        """Get specific feedback for education section"""
        if not any(degree in content.lower() for degree in ['bachelor', 'master', 'phd', 'associate']):
            return "Consider adding your degree information for better clarity."
        
        if not any(year in content for year in ['2020', '2021', '2022', '2023', '2024']):
            return "Include graduation date to show recency of education."
        
        return "Education section is well-structured and informative."
    
    def _get_projects_feedback(self, content: str, job_role: str) -> str:
        """Get specific feedback for projects section"""
        if len(content.split('\n')) < 3:
            return "Add more detail to your projects, including technologies used and outcomes achieved."
        
        if not any(word in content.lower() for word in ['built', 'developed', 'created', 'designed']):
            return "Use action verbs to describe what you accomplished in each project."
        
        return "Projects section demonstrates practical experience and technical skills."
    
    def _generate_section_suggestions(self, section_name: str, content: str, job_role: str) -> List[str]:
        """Generate specific suggestions for section improvement"""
        suggestions = []
        
        if section_name == 'summary':
            suggestions.extend([
                "Start with your years of experience and primary expertise",
                "Include 2-3 key skills relevant to the target role",
                "End with your career objective or value proposition"
            ])
        elif section_name in ['experience', 'work history', 'employment']:
            suggestions.extend([
                "Use the STAR method (Situation, Task, Action, Result) for bullet points",
                "Include specific metrics and quantifiable achievements",
                "Start each bullet point with a strong action verb",
                "Focus on accomplishments rather than job duties"
            ])
        elif section_name in ['skills', 'technical skills', 'competencies']:
            suggestions.extend([
                "Group skills by category (Programming, Tools, Frameworks, etc.)",
                "Include proficiency levels for key skills",
                "Add recent and trending technologies relevant to the role",
                "Remove outdated or irrelevant skills"
            ])
        elif section_name in ['education', 'academic', 'qualifications']:
            suggestions.extend([
                "Include relevant coursework if recent graduate",
                "Add academic achievements or honors if applicable",
                "Include relevant certifications or training"
            ])
        elif section_name in ['projects', 'portfolio', 'key projects']:
            suggestions.extend([
                "Include project links or GitHub repositories",
                "Describe the problem you solved and your approach",
                "Highlight technologies used and your specific contributions",
                "Include project outcomes and impact"
            ])
        
        return suggestions
    
    def _find_missing_keywords(self, section_name: str, content: str, job_role: str) -> List[str]:
        """Find missing keywords for a specific section"""
        role_keywords = self.resume_processor.role_keywords.get(job_role.lower().replace(' ', '_'), [])
        content_lower = content.lower()
        
        missing = [kw for kw in role_keywords if kw.lower() not in content_lower]
        return missing[:5]  # Return top 5 missing keywords
    
    def _generate_improvements(self, section_name: str, content: str, job_role: str) -> List[str]:
        """Generate specific improvement recommendations"""
        improvements = []
        
        if section_name == 'summary':
            improvements.extend([
                "Add specific years of experience",
                "Include key technologies or methodologies",
                "Mention your unique value proposition"
            ])
        elif section_name in ['experience', 'work history', 'employment']:
            improvements.extend([
                "Quantify achievements with specific numbers",
                "Use more powerful action verbs",
                "Add context about company size or industry"
            ])
        elif section_name in ['skills', 'technical skills', 'competencies']:
            improvements.extend([
                "Organize skills by proficiency level",
                "Add recent and trending technologies",
                "Include soft skills relevant to the role"
            ])
        
        return improvements
    
    def _generate_priority_improvements(self, section_analyses: List[SectionAnalysis], ats_analysis: ATSAnalysis) -> List[str]:
        """Generate priority improvements based on analysis"""
        priorities = []
        
        # Find lowest scoring sections
        low_score_sections = [sa for sa in section_analyses if sa.score < 60]
        if low_score_sections:
            priorities.append(f"Focus on improving {', '.join([sa.section_name for sa in low_score_sections])} sections")
        
        # ATS-specific improvements
        if ats_analysis.component_scores['keyword_match'] < 60:
            priorities.append("Add more relevant keywords throughout the resume")
        
        if ats_analysis.component_scores['formatting'] < 60:
            priorities.append("Improve resume formatting and structure")
        
        if ats_analysis.missing_keywords:
            priorities.append(f"Include missing keywords: {', '.join(ats_analysis.missing_keywords[:3])}")
        
        return priorities[:5]  # Top 5 priorities
    
    def _estimate_improvement_potential(self, section_analyses: List[SectionAnalysis], ats_analysis: ATSAnalysis) -> float:
        """Estimate potential score improvement"""
        current_score = ats_analysis.total_score
        
        # Calculate improvement potential based on section scores
        improvement_potential = 0
        for section in section_analyses:
            if section.score < 70:
                improvement_potential += (70 - section.score) * 0.3
        
        # Add ATS improvement potential
        if ats_analysis.component_scores['keyword_match'] < 70:
            improvement_potential += 15
        
        if ats_analysis.component_scores['formatting'] < 70:
            improvement_potential += 10
        
        estimated_new_score = min(current_score + improvement_potential, 95)
        return estimated_new_score

# Example usage
if __name__ == "__main__":
    feedback_gen = SectionWiseFeedbackGenerator()
    
    sample_resume = """
    John Doe
    john.doe@email.com | (555) 123-4567
    
    SUMMARY
    Experienced software engineer with 5+ years in full-stack development.
    
    EXPERIENCE
    Senior Software Engineer | Tech Corp | 2020-2023
    • Developed web applications using Python and React
    • Led team of 3 developers
    
    EDUCATION
    Bachelor of Computer Science | University of Tech | 2018
    
    SKILLS
    Python, JavaScript, React, SQL, AWS
    """
    
    # Generate comprehensive feedback
    feedback = feedback_gen.generate_comprehensive_feedback(sample_resume, "software_engineer")
    
    print(f"Overall Score: {feedback.overall_feedback.overall_score}")
    print(f"Estimated Improvement: {feedback.estimated_improvement}")
    print(f"Priority Improvements: {feedback.priority_improvements}")
    
    for section in feedback.section_analyses:
        print(f"\n{section.section_name.title()} Section:")
        print(f"Score: {section.score}/100")
        print(f"Feedback: {section.feedback}")
        print(f"Suggestions: {section.suggestions[:2]}")
