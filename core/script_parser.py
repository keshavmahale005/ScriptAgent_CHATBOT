"""
Universal Script Parser v2 - Works with ANY call script format
No hardcoding - adapts to any structure automatically
"""

import re
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class UniversalScriptParser:
    """
    Universal parser that works with ANY call script format.
    Automatically detects structure and extracts content.
    NO HARDCODING - Pure pattern detection.
    """
    
    def __init__(self, script_text: str):
        self.raw_text = script_text
        self.lines = [line.rstrip() for line in script_text.split('\n')]
        
        # Parsed data
        self.metadata: Dict[str, str] = {}
        self.sections: List[Dict] = []
        self.variables: Set[str] = set()
        
        # Configuration (auto-detected)
        self.agent_patterns = self._detect_agent_patterns()
        self.section_pattern = self._detect_section_pattern()
        
        # Parse
        self._parse()
    
    def _detect_agent_patterns(self) -> List[str]:
        """
        Auto-detect how agent lines are formatted in THIS script.
        Returns patterns found in the script.
        """
        patterns_found = set()
        
        for line in self.lines[:100]:  # Check first 100 lines
            line = line.strip()
            
            # Pattern 1: Agent: or Agent (Name):
            if re.match(r'^Agent\s*(\([^)]*\))?:', line, re.IGNORECASE):
                patterns_found.add('agent_colon')
            
            # Pattern 2: Name: (like Clare:, Sarah:, John:)
            if re.match(r'^[A-Z][a-z]+:', line):
                patterns_found.add('name_colon')
            
            # Pattern 3: (Name): like (Sarah):
            if re.match(r'^\([A-Z][a-z\s]+\):', line):
                patterns_found.add('name_parens')
            
            # Pattern 4: Just quoted text
            if re.match(r'^"[^"]+"$', line):
                patterns_found.add('quoted_only')
        
        logger.info(f"🔍 Detected agent patterns: {patterns_found}")
        return list(patterns_found)
    
    def _detect_section_pattern(self) -> str:
        """
        Auto-detect section header pattern in THIS script.
        """
        all_caps_count = 0
        title_case_count = 0
        
        for line in self.lines:
            line = line.strip()
            
            # Skip obvious non-headers
            if not line or len(line) < 3:
                continue
            if line.startswith(('Agent', 'If', 'IF', '"', '(')):
                continue
            if ':' in line and len(line.split(':')[0]) < 30:
                continue  # Likely metadata
            
            # Check patterns
            word_count = len(line.split())
            if 1 <= word_count <= 8:
                if line.isupper():
                    all_caps_count += 1
                elif line.istitle():
                    title_case_count += 1
        
        if all_caps_count > title_case_count:
            logger.info("📋 Section pattern: ALL CAPS")
            return 'all_caps'
        else:
            logger.info("📋 Section pattern: Mixed/Title Case")
            return 'mixed'
    
    def _parse(self):
        """Main parser"""
        logger.info("🔄 Starting universal parsing...")
        
        # Step 1: Extract metadata
        self._extract_metadata()
        
        # Step 2: Extract variables
        self._extract_variables()
        
        # Step 3: Parse sections
        self._parse_sections()
        
        logger.info(f"✅ Parsed {len(self.sections)} sections, {len(self.variables)} variables")
    
    def _extract_metadata(self):
        """Extract metadata from header (first section)"""
        in_metadata = True
        
        for line in self.lines[:30]:
            line = line.strip()
            
            if not line:
                continue
            
            # Stop at first clear section header
            if self._is_likely_section_header(line):
                in_metadata = False
                break
            
            # Extract key: value pairs
            if ':' in line and in_metadata:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Validate it's metadata (short key, not agent line)
                    if (len(key) < 30 and 
                        not key.lower().startswith(('agent ', 'if ', 'when ')) and
                        not line.startswith('"')):
                        self.metadata[key.lower()] = value
        
        logger.info(f"📋 Metadata fields: {list(self.metadata.keys())}")
    
    def _extract_variables(self):
        """Extract {{variable}} placeholders"""
        pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'
        matches = re.findall(pattern, self.raw_text)
        self.variables = set(matches)
        
        if self.variables:
            logger.info(f"🔧 Variables: {self.variables}")
    
    def _is_likely_section_header(self, line: str) -> bool:
        """
        Universal section header detection.
        Works with ANY format - no hardcoding.
        """
        line = line.strip()
        
        if not line or len(line) < 3:
            return False
        
        # NOT a header if:
        
        # 1. Starts with agent indicators
        if re.match(r'^(Agent|If|IF|When|WHEN|Unless|UNLESS|User|Customer)\s*[:(]', line, re.IGNORECASE):
            return False
        
        # 2. Is a quoted string
        if line.startswith('"') or line.endswith('"'):
            return False
        
        # 3. Starts with parenthesis (like explanations)
        if line.startswith('('):
            return False
        
        # 4. Is metadata (key: value with short key at start of document)
        if ':' in line:
            key_part = line.split(':')[0]
            # If key is short and at doc start, probably metadata
            if len(key_part) < 30:
                # But could also be section like "Objection: Not Interested"
                # Headers usually don't have long text after colon
                value_part = line.split(':', 1)[1] if len(line.split(':', 1)) > 1 else ''
                if len(value_part) > 50:
                    return False  # Probably metadata or content
        
        # IS a header if:
        
        # 1. Short phrase (1-8 words)
        word_count = len(line.split())
        if word_count < 1 or word_count > 8:
            return False
        
        # 2. Has capitalization
        has_caps = any(c.isupper() for c in line)
        if not has_caps:
            return False
        
        # 3. Matches common patterns
        is_all_caps = line.isupper()
        is_title_case = line.istitle()
        
        # Additional check: common section keywords
        section_keywords = [
            'start', 'end', 'opening', 'closing', 'introduction', 'greeting',
            'collection', 'details', 'information', 'booking', 'confirmation',
            'objection', 'handling', 'qualification', 'verification', 'call'
        ]
        
        line_lower = line.lower()
        has_section_keyword = any(keyword in line_lower for keyword in section_keywords)
        
        # Score-based decision
        score = 0
        if is_all_caps: score += 3
        if is_title_case: score += 2
        if has_section_keyword: score += 2
        if word_count <= 4: score += 1
        if 5 <= word_count <= 8: score += 0.5
        
        return score >= 2
    
    def _is_agent_line(self, line: str) -> bool:
        """
        Universal agent line detection.
        Adapts to patterns found in the script.
        """
        line = line.strip()
        
        if not line:
            return False
        
        # Pattern 1: Agent: or Agent (Name):
        if re.match(r'^Agent\s*(\([^)]*\))?:', line, re.IGNORECASE):
            return True
        
        # Pattern 2: Name: (capitalized name followed by colon)
        if re.match(r'^[A-Z][a-z]{2,15}:', line):
            return True
        
        # Pattern 3: (Name): 
        if re.match(r'^\([A-Z][a-z\s]+\):', line):
            return True
        
        # Pattern 4: Quoted dialogue on its own line
        if re.match(r'^"[^"]+"$', line) and len(line) > 10:
            return True
        
        return False
    
    def _parse_sections(self):
        """Parse all sections - universal approach"""
        current_section = None
        current_content = []
        current_agent_lines = []
        
        i = 0
        while i < len(self.lines):
            line = self.lines[i].strip()
            
            # Skip empty
            if not line:
                i += 1
                continue
            
            # Check if section header
            if self._is_likely_section_header(line):
                # Save previous section
                if current_section:
                    self._save_section(current_section, current_content, current_agent_lines)
                
                # Start new section
                current_section = line
                current_content = []
                current_agent_lines = []
                i += 1
                continue
            
            # Check if agent line
            if self._is_agent_line(line):
                # Extract agent dialogue (could be multi-line)
                agent_text, lines_consumed = self._extract_agent_dialogue(i)
                
                if agent_text:
                    current_agent_lines.append(agent_text)
                    current_content.append(agent_text)
                
                i += lines_consumed
                continue
            
            # Regular content
            current_content.append(line)
            i += 1
        
        # Save last section
        if current_section:
            self._save_section(current_section, current_content, current_agent_lines)
        
        # If no sections found, create one default section
        if not self.sections:
            all_agent_lines = self._extract_all_agent_lines()
            if all_agent_lines:
                self._save_section("CONVERSATION", all_agent_lines, all_agent_lines)
    
    def _extract_agent_dialogue(self, start_idx: int) -> Tuple[str, int]:
        """
        Extract agent dialogue starting at start_idx.
        Returns: (cleaned_dialogue, lines_consumed)
        """
        lines_consumed = 1
        dialogue_parts = [self.lines[start_idx].strip()]
        
        # Check if multi-line dialogue
        i = start_idx + 1
        while i < len(self.lines):
            next_line = self.lines[i].strip()
            
            # Stop if we hit another section, agent line, or empty
            if not next_line:
                break
            if self._is_likely_section_header(next_line):
                break
            if self._is_agent_line(next_line):
                break
            
            # Stop if it looks like a new statement (starts with IF, When, etc)
            if re.match(r'^(If |IF |When |WHEN |Unless )', next_line, re.IGNORECASE):
                break
            
            # Add to dialogue
            dialogue_parts.append(next_line)
            lines_consumed += 1
            i += 1
        
        # Clean the dialogue
        full_dialogue = ' '.join(dialogue_parts)
        cleaned = self._clean_agent_text(full_dialogue)
        
        return cleaned, lines_consumed
    
    def _clean_agent_text(self, text: str) -> str:
        """Remove agent prefixes and clean up text"""
        # Remove Agent: and variants
        text = re.sub(r'^Agent\s*(\([^)]*\))?:\s*', '', text, flags=re.IGNORECASE)
        
        # Remove name prefixes like "Clare:" or "(Sarah):"
        text = re.sub(r'^[A-Z][a-z]+:\s*', '', text)
        text = re.sub(r'^\([A-Z][a-z\s]+\):\s*', '', text)
        
        # Remove surrounding quotes
        text = text.strip('"\'')
        
        # Clean whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _extract_all_agent_lines(self) -> List[str]:
        """Extract all agent lines if no sections detected"""
        agent_lines = []
        
        for line in self.lines:
            if self._is_agent_line(line.strip()):
                cleaned = self._clean_agent_text(line.strip())
                if cleaned:
                    agent_lines.append(cleaned)
        
        return agent_lines
    
    def _save_section(self, name: str, content: List[str], agent_lines: List[str]):
        """Save a section to the sections list"""
        # Create dialogue from agent lines
        dialogue = ' '.join(agent_lines).strip()
        
        self.sections.append({
            'name': name,
            'content': '\n'.join(content),
            'agent_lines': agent_lines.copy(),
            'dialogue': dialogue,
            'variables': self.variables,
            'is_conditional': 'IF ' in name.upper() or 'WHEN ' in name.upper(),
            'line_count': len(agent_lines)
        })
    
    def get_opening_message(self) -> str:
        """Get first agent line"""
        # Look for START/OPENING/GREETING section
        start_keywords = ['start', 'open', 'greet', 'begin', 'hello']
        
        for section in self.sections:
            section_name_lower = section['name'].lower()
            if any(kw in section_name_lower for kw in start_keywords):
                if section['agent_lines']:
                    return section['agent_lines'][0]
        
        # Fallback: first section with agent lines
        for section in self.sections:
            if section['agent_lines']:
                return section['agent_lines'][0]
        
        # Ultimate fallback
        return "Hello! How can I help you today?"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for flow engine"""
        return {
            'metadata': self.metadata,
            'statistics': {
                'total_sections': len(self.sections),
                'agent_lines': sum(len(s['agent_lines']) for s in self.sections),
                'variables': len(self.variables)
            },
            'sections': self.sections,
            'conditionals': {},
            'variables': list(self.variables),
            'total_responses': sum(len(s['agent_lines']) for s in self.sections)
        }