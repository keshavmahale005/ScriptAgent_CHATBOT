"""
Ollama Engine - Generate responses using local Ollama with Script Flow Awareness
"""

import ollama
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OllamaEngine:
    """Handle Ollama LLM interactions with script flow awareness"""
    
    def __init__(self, config):
        """Initialize Ollama engine"""
        self.config = config
        self.model = config.OLLAMA_MODEL
        self.temperature = config.OLLAMA_TEMPERATURE
        self.max_tokens = config.OLLAMA_MAX_TOKENS
        self.host = config.OLLAMA_HOST
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available"""
        try:
            # Test connection
            models_response = ollama.list()
            logger.info("✅ Connected to Ollama")
            
            # Extract model names safely
            models = models_response.get("models", [])
            model_names = []
            for m in models:
                if "name" in m:
                    model_names.append(m["name"])
                elif "model" in m:
                    model_names.append(m["model"])
            
            logger.info(f"📋 Available models: {model_names}")
            
            # Check if our model is available
            if self.model not in model_names:
                logger.warning(f"⚠️ Model '{self.model}' not found. Downloading...")
                ollama.pull(self.model)
                logger.info(f"✅ Model '{self.model}' downloaded")
            else:
                logger.info(f"✅ Model '{self.model}' is available")
                
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            raise
    
    def generate_response(
        self,
        user_input: str,
        intent_data: Dict[str, Any],
        flow_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        script_metadata: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Generate response using Ollama with flow context
        
        Args:
            user_input: User's message
            intent_data: Detected intent information
            flow_context: Context from ScriptFlowEngine (what section we're in, what to say)
            conversation_history: Previous messages
            script_metadata: Script metadata (tone, style, etc.)
            
        Returns:
            Dict with response and metadata
        """
        try:
            # Get the exact agent line from flow context
            exact_agent_line = flow_context.get('agent_line', '')
            
            # 🔥 CRITICAL FIX: Use COMPLETE dialogue without splitting
            # The flow engine gives us the full response for this turn
            if exact_agent_line and len(exact_agent_line) > 10:
                # Clean and use the COMPLETE script line
                response_text = self._clean_and_prepare_script_line(exact_agent_line, script_metadata)
                
                return {
                    'response': response_text,
                    'confidence': 1.0,
                    'method': 'script_exact',
                    'model': self.model,
                    'fallback_used': False,
                    'section': flow_context.get('section', 'unknown'),
                    'phase': flow_context.get('phase', 'unknown')
                }
            
            # Otherwise, generate with heavy script guidance
            prompt = self._build_flow_aware_prompt(
                user_input=user_input,
                intent_data=intent_data,
                flow_context=flow_context,
                conversation_history=conversation_history,
                script_metadata=script_metadata
            )
            
            # Generate with Ollama
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_flow_aware_system_prompt(script_metadata, flow_context)
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.2,  # Low temperature for strict adherence
                    "top_p": 0.5,        
                    "top_k": 20,         
                    "repeat_penalty": 1.3,
                    "num_predict": 200   # Reasonable limit for full responses
                }
            )
            
            # Extract response text
            response_text = response['message']['content'].strip()
            
            # Clean response
            response_text = self._clean_response(response_text)
            
            return {
                'response': response_text,
                'confidence': 0.85,
                'method': 'ollama_flow_guided',
                'model': self.model,
                'fallback_used': False,
                'section': flow_context.get('section', 'unknown'),
                'phase': flow_context.get('phase', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Fallback: Use exact script line if available
            if flow_context.get('agent_line'):
                fallback_text = self._clean_and_prepare_script_line(
                    flow_context['agent_line'],
                    script_metadata
                )
                
                return {
                    'response': fallback_text,
                    'confidence': 1.0,
                    'method': 'fallback_script_exact',
                    'fallback_used': True,
                    'section': flow_context.get('section', 'unknown'),
                    'phase': flow_context.get('phase', 'unknown')
                }
            else:
                return {
                    'response': "I apologize, could you please repeat that?",
                    'confidence': 0.0,
                    'method': 'fallback_generic',
                    'fallback_used': True
                }
    
    def _clean_and_prepare_script_line(self, script_line: str, metadata: Dict) -> str:
        """
        Clean script line and prepare it for delivery.
        Removes quotes, cleans formatting, ensures proper punctuation.
        """
        if not script_line:
            return ""
        
        # Remove surrounding quotes
        if script_line.startswith('"') and script_line.endswith('"'):
            script_line = script_line[1:-1]
        
        # Remove agent name prefixes if present
        prefixes = ["Agent:", "Agent (Clare):", "Clare:", "Assistant:", "Agent (", "Response:"]
        for prefix in prefixes:
            if script_line.startswith(prefix):
                script_line = script_line[len(prefix):].strip()
                # Also remove closing parenthesis if it was "Agent (Clare):"
                if script_line.startswith(")"):
                    script_line = script_line[1:].strip()
        
        # Remove extra quotes that might be inside
        script_line = script_line.replace('""', '"')
        
        # Clean up whitespace
        script_line = ' '.join(script_line.split())
        
        # Ensure ends with punctuation if it doesn't already
        if script_line and script_line[-1] not in '.!?':
            # Check if it's a question
            question_starters = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 
                               'can', 'could', 'would', 'do', 'does', 'may', 'might']
            if any(script_line.lower().startswith(q) for q in question_starters):
                script_line += '?'
            else:
                script_line += '.'
        
        return script_line.strip()
    
    def _clean_response(self, response: str) -> str:
        """Clean up AI response artifacts"""
        if not response:
            return ""
        
        # Remove markdown
        response = response.replace('**', '').replace('*', '')
        
        # Remove common prefixes
        prefixes = ["Agent: ", "Assistant: ", "Response: ", "Clare: ", "Agent (Clare): ", "Agent (Clare):", "Agent:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove quotes if wrapping entire response
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Clean up extra whitespace
        response = ' '.join(response.split())
        
        # Ensure ends with punctuation
        if response and response[-1] not in '.!?':
            question_starters = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 
                               'can', 'could', 'would', 'do', 'does', 'may', 'might']
            if any(response.lower().startswith(q) for q in question_starters):
                response += '?'
            else:
                response += '.'
        
        return response.strip()
    
    def _get_flow_aware_system_prompt(self, metadata: Dict, flow_context: Dict) -> str:
        """Create system prompt with flow awareness"""
        
        agent_name = metadata.get('agent_name', metadata.get('agent name', 'the agent'))
        tone = metadata.get('tone', metadata.get('style', 'professional and friendly'))
        call_type = metadata.get('call_type', 'outbound').lower()
        
        current_section = flow_context.get('section', 'unknown')
        current_phase = flow_context.get('phase', 'conversation')
        exact_line = flow_context.get('agent_line', '')
        
        return f"""You are {agent_name}, a professional call center agent following a precise script.

🎯 CRITICAL INSTRUCTIONS:
1. You MUST say EXACTLY what's in the script below
2. Use the COMPLETE script line provided - do not shorten or split it
3. Do NOT add extra information not in the script
4. Do NOT skip any part of the script
5. Only make tiny adjustments for natural flow (like "Great!" before the script)

CURRENT CONTEXT:
- Section: {current_section}
- Phase: {current_phase}
- Call Type: {call_type.upper()}
- Tone: {tone}

YOUR EXACT SCRIPT LINE (USE COMPLETE LINE):
"{exact_line}"

WHAT TO DO:
- Say the complete line above word-for-word
- Keep it natural and conversational
- Follow the script precisely
- Do not split this into multiple parts
- Say it all in one response

Remember: Script compliance is critical. Say the COMPLETE line."""
    
    def _build_flow_aware_prompt(
        self,
        user_input: str,
        intent_data: Dict,
        flow_context: Dict,
        conversation_history: List[Dict],
        script_metadata: Dict
    ) -> str:
        """Build prompt with flow context"""
        
        parts = []
        
        # Get exact line
        exact_line = flow_context.get('agent_line', '')
        
        # Flow context
        parts.append("=== CURRENT SITUATION ===")
        parts.append(f"Section: {flow_context.get('section', 'unknown')}")
        parts.append(f"Phase: {flow_context.get('phase', 'conversation')}")
        parts.append("")
        
        # Recent conversation (last 3 exchanges)
        if len(conversation_history) > 1:
            parts.append("=== RECENT CONVERSATION ===")
            for msg in conversation_history[-6:]:
                role = "Agent" if msg['role'] == 'assistant' else "Customer"
                parts.append(f"{role}: {msg['content']}")
            parts.append("")
        
        # User's current input
        parts.append("=== CUSTOMER JUST SAID ===")
        parts.append(f'"{user_input}"')
        parts.append("")
        
        # Exact script line (MOST IMPORTANT)
        if exact_line:
            parts.append("=== YOUR COMPLETE RESPONSE (SAY THIS EXACTLY) ===")
            parts.append(f'"{exact_line}"')
            parts.append("")
            parts.append("🎯 Use the COMPLETE line above as your response.")
            parts.append("   Do not shorten it. Do not split it. Say it all.")
        else:
            parts.append("=== YOUR RESPONSE ===")
            parts.append("Respond briefly and naturally to continue the conversation.")
        
        return "\n".join(parts)
    
    def generate_response_legacy(
        self,
        user_input: str,
        intent_data: Dict[str, Any],
        relevant_sections: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
        script_metadata: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Legacy method for backwards compatibility.
        Converts to new flow-aware format.
        """
        # Create a basic flow context from relevant sections
        flow_context = {
            'section': relevant_sections[0]['section_name'] if relevant_sections else 'GENERAL',
            'agent_line': relevant_sections[0]['text'] if relevant_sections else '',
            'phase': 'CONVERSATION',
            'type': 'CONVERSATION'
        }
        
        return self.generate_response(
            user_input=user_input,
            intent_data=intent_data,
            flow_context=flow_context,
            conversation_history=conversation_history,
            script_metadata=script_metadata
        )