"""
Script Flow Engine - Maintains conversation state and follows script precisely
"""

import re
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ScriptFlowEngine:
    """
    Manages conversation flow according to script structure.
    Ensures AI follows the script precisely and in the correct order.
    """
    
    def __init__(self, parsed_script: Dict[str, Any]):
        """Initialize with parsed script"""
        self.script = parsed_script
        self.sections = parsed_script.get('sections', [])
        self.metadata = parsed_script.get('metadata', {})
        
        # Build flow map
        self.flow_map = self._build_flow_map()
        
        # Initialize state
        self.current_section = None
        self.completed_sections = set()
        self.collected_data = {}
        self.conversation_phase = "START"
        
        logger.info(f"Script Flow Engine initialized with {len(self.sections)} sections")
    
    def _build_flow_map(self) -> Dict[str, Dict]:
        """
        Build a map of conversation flow from script sections.
        
        Returns:
            Dict mapping section names to their data and next steps
        """
        flow = {}
        
        for i, section in enumerate(self.sections):
            section_name = section.get('name', f'SECTION_{i}')
            
            # Determine section type
            section_type = self._classify_section(section_name, section)
            
            flow[section_name] = {
                'index': i,
                'name': section_name,
                'type': section_type,
                'dialogue': section.get('dialogue', ''),
                'conditions': section.get('conditions', []),
                'required_fields': self._extract_required_fields(section),
                'next_section': self._determine_next_section(i),
                'is_conditional': section.get('is_conditional', False),
                'raw_section': section
            }
        
        return flow
    
    def _classify_section(self, name: str, section: Dict) -> str:
        """Classify section type based on name and content"""
        name_lower = name.lower()
        
        if 'start' in name_lower or 'greeting' in name_lower or 'opening' in name_lower:
            return 'OPENING'
        elif 'introduction' in name_lower or 'intro' in name_lower:
            return 'INTRODUCTION'
        elif 'personal details' in name_lower or 'contact' in name_lower or 'name' in name_lower or 'title' in name_lower:
            return 'DATA_COLLECTION'
        elif 'property' in name_lower or 'address' in name_lower:
            return 'PROPERTY_INFO'
        elif 'booking' in name_lower or 'appointment' in name_lower or 'confirm' in name_lower:
            return 'BOOKING'
        elif 'objection' in name_lower or 'if user' in name_lower or 'handle' in name_lower:
            return 'OBJECTION_HANDLING'
        elif 'end' in name_lower or 'goodbye' in name_lower or 'closing' in name_lower:
            return 'CLOSING'
        else:
            return 'CONVERSATION'
    
    def _extract_required_fields(self, section: Dict) -> List[str]:
        """Extract variable names that need to be collected"""
        dialogue = section.get('dialogue', '')
        variables = section.get('variables', set())
        
        # Find {{variable}} patterns
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, dialogue)
        
        # Combine with explicit variables
        return list(set(matches) | variables)
    
    def _determine_next_section(self, current_index: int) -> Optional[str]:
        """Determine the next section in sequence"""
        if current_index + 1 < len(self.sections):
            return self.sections[current_index + 1].get('name')
        return None
    
    def start_conversation(self) -> Dict[str, Any]:
        """
        Start conversation and get opening message.
        
        Returns:
            Dict with section info and agent's opening line
        """
        # Find the CALL START or first section
        for section_name, section_data in self.flow_map.items():
            if section_data['type'] == 'OPENING':
                self.current_section = section_name
                self.conversation_phase = 'OPENING'
                
                return {
                    'section': section_name,
                    'type': 'OPENING',
                    'agent_line': section_data['dialogue'],
                    'required_fields': section_data['required_fields'],
                    'phase': self.conversation_phase
                }
        
        # Fallback: use first section
        if self.sections:
            first_section = self.sections[0]
            section_name = first_section.get('name', 'START')
            self.current_section = section_name
            
            return {
                'section': section_name,
                'type': 'OPENING',
                'agent_line': first_section.get('dialogue', ''),
                'required_fields': [],
                'phase': 'OPENING'
            }
        
        return {
            'section': 'DEFAULT',
            'type': 'OPENING',
            'agent_line': 'Hello! Thank you for your interest.',
            'required_fields': [],
            'phase': 'OPENING'
        }
    
    def get_next_step(self, user_input: str, intent_data: Dict) -> Dict[str, Any]:
        """
        Determine what the agent should say next based on:
        - Current section
        - User's response
        - Required fields
        - Script flow
        
        Args:
            user_input: What user said
            intent_data: Detected intent/sentiment
            
        Returns:
            Dict with next agent line and context
        """
        if not self.current_section:
            return self.start_conversation()
        
        current = self.flow_map.get(self.current_section)
        if not current:
            logger.error(f"Current section '{self.current_section}' not found in flow map")
            return self._fallback_response()
        
        # Check for objections first
        objection_response = self._check_for_objections(user_input, intent_data)
        if objection_response:
            return objection_response
        
        # Analyze user response
        user_intent = intent_data.get('primary_intent', 'NEUTRAL')
        sentiment = intent_data.get('sentiment', 'neutral')
        
        # Handle based on section type and user response
        if current['type'] == 'OPENING':
            return self._handle_opening(user_input, intent_data)
        elif current['type'] == 'INTRODUCTION':
            return self._handle_introduction(user_input, intent_data)
        elif current['type'] == 'DATA_COLLECTION':
            return self._handle_data_collection(user_input, intent_data)
        elif current['type'] == 'PROPERTY_INFO':
            return self._handle_property_info(user_input, intent_data)
        elif current['type'] == 'BOOKING':
            return self._handle_booking(user_input, intent_data)
        else:
            return self._handle_general_conversation(user_input, intent_data)
    
    def _check_for_objections(self, user_input: str, intent_data: Dict) -> Optional[Dict]:
        """Check if user is objecting and return appropriate response"""
        user_lower = user_input.lower()
        
        # Common objection patterns
        objections = {
            'not_good_time': ['not a good time', 'busy', 'call back', 'later', 'not now'],
            'not_interested': ['not interested', "don't want", 'no thanks', 'not for me'],
            'too_long': ['how long', 'too long', 'quick', 'time'],
            'bad_reviews': ['review', 'scam', 'trust', 'legitimate'],
            'thought_instant': ['instant', 'immediate', 'now', 'straight away']
        }
        
        # Find matching objection sections
        for section_name, section_data in self.flow_map.items():
            if section_data['type'] == 'OBJECTION_HANDLING':
                section_lower = section_name.lower()
                
                # Check if this objection matches
                for obj_type, keywords in objections.items():
                    if any(keyword in user_lower for keyword in keywords):
                        if any(keyword in section_lower for keyword in keywords):
                            return {
                                'section': section_name,
                                'type': 'OBJECTION_HANDLING',
                                'agent_line': section_data['dialogue'],
                                'required_fields': [],
                                'phase': 'OBJECTION',
                                'objection_type': obj_type
                            }
        
        return None
    
    def _handle_opening(self, user_input: str, intent_data: Dict) -> Dict[str, Any]:
        """Handle opening section"""
        user_lower = user_input.lower()
        
        # Check if user confirmed they are the right person
        if any(word in user_lower for word in ['yes', 'yeah', 'speaking', 'this is', 'yep', 'yea', 'sure', 'ok', 'okay']):
            # Move to introduction
            next_section = self._get_next_section_by_type('INTRODUCTION')
            if next_section:
                self.current_section = next_section
                self.completed_sections.add('OPENING')
                self.conversation_phase = 'INTRODUCTION'
                
                section_data = self.flow_map[next_section]
                return {
                    'section': next_section,
                    'type': 'INTRODUCTION',
                    'agent_line': section_data['dialogue'],
                    'required_fields': section_data['required_fields'],
                    'phase': 'INTRODUCTION'
                }
        
        # User didn't confirm - ask again or clarify
        return {
            'section': self.current_section,
            'type': 'OPENING',
            'agent_line': "Sorry, may I confirm who I'm speaking with?",
            'required_fields': ['first_name'],
            'phase': 'OPENING'
        }
    
    def _handle_introduction(self, user_input: str, intent_data: Dict) -> Dict[str, Any]:
        """Handle introduction section"""
        user_lower = user_input.lower()
        
        # Check if user agrees to continue (comprehensive positive responses)
        positive_responses = ['yes', 'yeah', 'ok', 'okay', 'sure', 'go ahead', 'right', 'correct', 
                            "that's right", 'yep', 'yea', 'fine', 'good', 'absolutely', 'definitely']
        
        if any(word in user_lower for word in positive_responses):
            # Move to next section (usually data collection)
            next_section = self._find_next_sequential_section()
            if next_section:
                self.current_section = next_section
                self.completed_sections.add('INTRODUCTION')
                
                section_data = self.flow_map[next_section]
                self.conversation_phase = section_data['type']
                
                return {
                    'section': next_section,
                    'type': section_data['type'],
                    'agent_line': section_data['dialogue'],
                    'required_fields': section_data['required_fields'],
                    'phase': self.conversation_phase
                }
        
        # User has questions or concerns
        return {
            'section': self.current_section,
            'type': 'INTRODUCTION',
            'agent_line': "I'd be happy to answer any questions you have. What would you like to know?",
            'required_fields': [],
            'phase': 'INTRODUCTION'
        }
    
    def _handle_data_collection(self, user_input: str, intent_data: Dict) -> Dict[str, Any]:
        """Handle data collection sections"""
        current = self.flow_map[self.current_section]
        required_fields = current['required_fields']
        
        # Try to extract data from user input
        extracted_data = self._extract_data_from_input(user_input, required_fields)
        self.collected_data.update(extracted_data)
        
        # If user provides data or confirms, move forward
        user_lower = user_input.lower()
        
        if user_input.strip() and len(user_input.strip()) > 2:
            # Move to next section
            next_section = self._find_next_sequential_section()
            if next_section:
                self.current_section = next_section
                self.completed_sections.add(current['name'])
                
                section_data = self.flow_map[next_section]
                return {
                    'section': next_section,
                    'type': section_data['type'],
                    'agent_line': section_data['dialogue'],
                    'required_fields': section_data['required_fields'],
                    'phase': section_data['type']
                }
        
        # Fallback
        return self._continue_to_next_section()
    
    def _handle_property_info(self, user_input: str, intent_data: Dict) -> Dict[str, Any]:
        """Handle property information collection"""
        return self._handle_data_collection(user_input, intent_data)
    
    def _handle_booking(self, user_input: str, intent_data: Dict) -> Dict[str, Any]:
        """Handle booking section"""
        user_lower = user_input.lower()
        
        # Check if user agrees to booking
        if any(word in user_lower for word in ['yes', 'ok', 'okay', 'sure', 'fine', 'good', 'yeah']):
            next_section = self._find_next_sequential_section()
            if next_section:
                self.current_section = next_section
                section_data = self.flow_map[next_section]
                
                return {
                    'section': next_section,
                    'type': section_data['type'],
                    'agent_line': section_data['dialogue'],
                    'required_fields': section_data['required_fields'],
                    'phase': 'BOOKING'
                }
        
        return {
            'section': self.current_section,
            'type': 'BOOKING',
            'agent_line': "Would you like to proceed with booking the consultation?",
            'required_fields': [],
            'phase': 'BOOKING'
        }
    
    def _handle_general_conversation(self, user_input: str, intent_data: Dict) -> Dict[str, Any]:
        """Handle general conversation - move to next section"""
        return self._continue_to_next_section()
    
    def _continue_to_next_section(self) -> Dict[str, Any]:
        """Move to next section in sequence"""
        next_section = self._find_next_sequential_section()
        if next_section:
            self.completed_sections.add(self.current_section)
            self.current_section = next_section
            
            section_data = self.flow_map[next_section]
            return {
                'section': next_section,
                'type': section_data['type'],
                'agent_line': section_data['dialogue'],
                'required_fields': section_data['required_fields'],
                'phase': section_data['type']
            }
        
        # No more sections - end call
        return {
            'section': 'END',
            'type': 'CLOSING',
            'agent_line': "Thank you for your time. Goodbye!",
            'required_fields': [],
            'phase': 'CLOSING'
        }
    
    def _find_next_sequential_section(self) -> Optional[str]:
        """Find next section in sequence that hasn't been completed"""
        if not self.current_section:
            return None
        
        current = self.flow_map.get(self.current_section)
        if not current:
            return None
        
        current_index = current['index']
        
        # Find next section that's not completed
        for i in range(current_index + 1, len(self.sections)):
            section = self.sections[i]
            section_name = section.get('name', f'SECTION_{i}')
            
            if section_name not in self.completed_sections:
                # Skip objection handling sections
                section_data = self.flow_map.get(section_name)
                if section_data and section_data['type'] != 'OBJECTION_HANDLING':
                    return section_name
        
        return None
    
    def _get_next_section_by_type(self, section_type: str) -> Optional[str]:
        """Find next section of specific type"""
        for section_name, section_data in self.flow_map.items():
            if section_data['type'] == section_type and section_name not in self.completed_sections:
                return section_name
        return None
    
    def _extract_data_from_input(self, user_input: str, expected_fields: List[str]) -> Dict[str, str]:
        """Extract expected data from user input"""
        extracted = {}
        
        # Simple extraction patterns
        if 'email' in expected_fields:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, user_input)
            if emails:
                extracted['email'] = emails[0]
        
        if 'mobile' in expected_fields or 'phone' in expected_fields:
            phone_pattern = r'\b\d{10,11}\b'
            phones = re.findall(phone_pattern, user_input.replace(' ', ''))
            if phones:
                extracted['mobile'] = phones[0]
        
        # For names, dates, etc., store the raw input
        if expected_fields and not extracted:
            extracted[expected_fields[0]] = user_input
        
        return extracted
    
    def _fallback_response(self) -> Dict[str, Any]:
        """Fallback response when state is unclear"""
        return {
            'section': 'FALLBACK',
            'type': 'CONVERSATION',
            'agent_line': "I'm sorry, could you repeat that?",
            'required_fields': [],
            'phase': self.conversation_phase or 'CONVERSATION'
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """Get conversation progress"""
        total_sections = len([s for s in self.flow_map.values() if s['type'] != 'OBJECTION_HANDLING'])
        completed = len(self.completed_sections)
        
        return {
            'current_section': self.current_section,
            'phase': self.conversation_phase,
            'completed_sections': list(self.completed_sections),
            'total_sections': total_sections,
            'progress_percentage': (completed / total_sections * 100) if total_sections > 0 else 0,
            'collected_data': self.collected_data
        }
    
    def reset(self):
        """Reset conversation state"""
        self.current_section = None
        self.completed_sections = set()
        self.collected_data = {}
        self.conversation_phase = "START"