import re
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass

@dataclass
class Section:
    title: str
    content: str
    level: int
    page_number: int
    start_position: int
    end_position: int
    subsections: List['Section'] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []

class StructureExtractor:
    """Extract document structure and hierarchy"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns for detecting sections
        self.section_patterns = [
            # Numbered sections (1., 1.1, 1.1.1, etc.)
            r'^(\d+(?:\.\d+)*)\s*\.?\s*([A-Z][^\n]*?)(?:\n|$)',
            # Lettered sections (A., B., etc.)
            r'^([A-Z])\.\s*([A-Z][^\n]*?)(?:\n|$)',
            # Roman numerals (I., II., III., etc.)
            r'^([IVX]+)\.\s*([A-Z][^\n]*?)(?:\n|$)',
            # All caps headings
            r'^([A-Z\s]{3,}?)(?:\n|$)',
            # Bold or emphasized text (approximated)
            r'^\*\*([^*]+)\*\*\s*(?:\n|$)',
            # Common academic section headers
            r'^(Abstract|Introduction|Literature Review|Methodology|Methods|Results|Discussion|Conclusion|References|Appendix)(?:\s*:?\s*)?(?:\n|$)',
        ]
    
    def extract_structure(self, text: str, page_number: int = 1) -> List[Section]:
        """Extract hierarchical structure from text"""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any section pattern
            section_match = self._is_section_header(line)
            
            if section_match:
                # Save previous section if exists
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    current_section.end_position = i
                    sections.append(current_section)
                
                # Start new section
                title, level = section_match
                current_section = Section(
                    title=title,
                    content="",
                    level=level,
                    page_number=page_number,
                    start_position=i,
                    end_position=i
                )
                current_content = []
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line)
                else:
                    # Create default section for content before first header
                    current_section = Section(
                        title="Introduction",
                        content="",
                        level=1,
                        page_number=page_number,
                        start_position=0,
                        end_position=0
                    )
                    current_content = [line]
        
        # Add final section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            current_section.end_position = len(lines)
            sections.append(current_section)
        
        # Build hierarchy
        return self._build_hierarchy(sections)
    
    def _is_section_header(self, line: str) -> Tuple[str, int]:
        """Check if line is a section header and return title and level"""
        for pattern in self.section_patterns:
            match = re.match(pattern, line, re.IGNORECASE | re.MULTILINE)
            if match:
                if len(match.groups()) >= 2:
                    # Numbered section
                    number = match.group(1)
                    title = match.group(2).strip()
                    level = len(number.split('.')) if '.' in number else 1
                    return f"{number}. {title}", level
                else:
                    # Simple header
                    title = match.group(1) if match.groups() else line
                    return title.strip(), 1
        
        # Check for common academic headers
        academic_headers = [
            'abstract', 'introduction', 'literature review', 'methodology', 
            'methods', 'results', 'discussion', 'conclusion', 'references', 
            'appendix', 'acknowledgments', 'background'
        ]
        
        if any(header in line.lower() for header in academic_headers):
            return line.strip(), 1
        
        return None
    
    def _build_hierarchy(self, sections: List[Section]) -> List[Section]:
        """Build hierarchical structure from flat section list"""
        if not sections:
            return []
        
        root_sections = []
        section_stack = []
        
        for section in sections:
            # Pop sections from stack that are at same or higher level
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()
            
            # Add as subsection if parent exists
            if section_stack:
                parent = section_stack[-1]
                parent.subsections.append(section)
            else:
                root_sections.append(section)
            
            section_stack.append(section)
        
        return root_sections
    
    def extract_subsections(self, section: Section) -> List[Dict]:
        """Extract subsections from a main section"""
        subsections = []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in section.content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 50:  # Filter out very short paragraphs
                subsections.append({
                    'index': i,
                    'content': paragraph,
                    'word_count': len(paragraph.split()),
                    'parent_section': section.title
                })
        
        # Also include formal subsections
        for subsection in section.subsections:
            subsections.append({
                'index': len(subsections),
                'content': subsection.content,
                'word_count': len(subsection.content.split()),
                'parent_section': section.title,
                'title': subsection.title
            })
        
        return subsections
    
    def get_section_statistics(self, sections: List[Section]) -> Dict:
        """Get statistics about extracted sections"""
        total_sections = len(sections)
        total_subsections = sum(len(s.subsections) for s in sections)
        total_content_length = sum(len(s.content) for s in sections)
        
        levels = [s.level for s in sections]
        max_depth = max(levels) if levels else 0
        
        return {
            'total_sections': total_sections,
            'total_subsections': total_subsections,
            'max_depth': max_depth,
            'total_content_length': total_content_length,
            'average_section_length': total_content_length / total_sections if total_sections > 0 else 0
        }
