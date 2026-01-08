"""
Retell AI Clone - Main Dashboard with Script Flow Engine
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import logging
import sys

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent / 'core'))

# Import core modules
from core.script_parser import UniversalScriptParser
from core.intent_detector import IntentDetector
from core.ollama_engine import OllamaEngine
from core.script_flow_engine import ScriptFlowEngine
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Retell AI Clone - Script Flow",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'script_loaded': False,
        'script_text': '',
        'parsed_script': None,
        'script_parser': None,
        'intent_detector': None,
        'ollama_engine': None,
        'flow_engine': None,  # NEW: Script flow engine
        'conversation_history': [],
        'call_active': False,
        'conversation_log': [],
        'call_type': 'outbound'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_call_type(script_metadata: dict, script_text: str) -> str:
    """Detect if inbound or outbound call"""
    script_lower = script_text.lower()
    
    # Check metadata
    if 'call_type' in script_metadata or 'call type' in script_metadata:
        call_type = script_metadata.get('call_type', script_metadata.get('call type', '')).lower()
        if 'inbound' in call_type:
            return 'inbound'
        elif 'outbound' in call_type:
            return 'outbound'
    
    # Check content
    inbound_indicators = [
        'thank you for calling',
        'thanks for calling',
        'how can i help you',
        'customer support'
    ]
    
    outbound_indicators = [
        'may i speak with',
        'is this',
        'calling from',
        'i\'m calling'
    ]
    
    inbound_score = sum(1 for indicator in inbound_indicators if indicator in script_lower)
    outbound_score = sum(1 for indicator in outbound_indicators if indicator in script_lower)
    
    return 'inbound' if inbound_score > outbound_score else 'outbound'

def load_script(script_text: str):
    """Load and parse script with flow engine"""
    try:
        with st.spinner("Parsing script..."):
            # Parse script
            parser = UniversalScriptParser(script_text)
            st.session_state.script_parser = parser
            st.session_state.parsed_script = parser.to_dict()
            
            # Detect call type
            st.session_state.call_type = detect_call_type(
                st.session_state.parsed_script['metadata'],
                script_text
            )
            
            # Add call_type to metadata
            st.session_state.parsed_script['metadata']['call_type'] = st.session_state.call_type
            
            logger.info(f"Detected call type: {st.session_state.call_type}")
            
            # Initialize intent detector
            st.session_state.intent_detector = IntentDetector(
                fuzzy_threshold=config.FUZZY_THRESHOLD
            )
            
            # Initialize Ollama engine
            st.session_state.ollama_engine = OllamaEngine(config)
            
            # Initialize Flow Engine (NEW!)
            st.session_state.flow_engine = ScriptFlowEngine(st.session_state.parsed_script)
            
            st.session_state.script_loaded = True
            st.session_state.script_text = script_text
            
            logger.info("Script loaded successfully with flow engine")
            return True
            
    except Exception as e:
        logger.error(f"Error loading script: {e}")
        st.error(f"Error loading script: {str(e)}")
        return False

def start_call():
    """Start a new call using flow engine"""
    st.session_state.call_active = True
    st.session_state.conversation_history = []
    
    # Reset flow engine
    st.session_state.flow_engine.reset()
    
    # Get opening from flow engine
    opening_context = st.session_state.flow_engine.start_conversation()
    opening_msg = opening_context['agent_line']
    
    # If empty, use default
    if not opening_msg or len(opening_msg.strip()) < 5:
        if st.session_state.call_type == 'inbound':
            opening_msg = "Thank you for calling! How can I help you today?"
        else:
            opening_msg = "Hello, this is calling. Is this a good time to talk?"
    
    # Add to conversation
    st.session_state.conversation_history.append({
        'role': 'assistant',
        'content': opening_msg,
        'timestamp': datetime.now().isoformat(),
        'section': opening_context['section'],
        'phase': opening_context['phase']
    })
    
    logger.info(f"Call started - Type: {st.session_state.call_type}, Section: {opening_context['section']}")

def end_call():
    """End current call"""
    # Log conversation
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'call_type': st.session_state.call_type,
        'conversation': st.session_state.conversation_history,
        'total_messages': len(st.session_state.conversation_history),
        'progress': st.session_state.flow_engine.get_progress()
    }
    st.session_state.conversation_log.append(log_entry)
    
    # Save to file
    log_file = config.LOGS_DIR / f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    st.session_state.call_active = False
    logger.info("Call ended")

def process_user_message(user_input: str):
    """Process user input using flow engine"""
    try:
        # Step 1: Detect intent
        intent_data = st.session_state.intent_detector.detect(user_input)
        
        logger.info(f"Intent: {intent_data['primary_intent']}, User said: {user_input}")
        
        # Step 2: Get next step from flow engine (MOST IMPORTANT!)
        flow_context = st.session_state.flow_engine.get_next_step(
            user_input=user_input,
            intent_data=intent_data
        )
        
        logger.info(f"Flow: Section={flow_context['section']}, Phase={flow_context['phase']}")
        
        # Step 3: Generate response using flow context
        generation_result = st.session_state.ollama_engine.generate_response(
            user_input=user_input,
            intent_data=intent_data,
            flow_context=flow_context,
            conversation_history=st.session_state.conversation_history,
            script_metadata=st.session_state.parsed_script['metadata']
        )
        
        response = generation_result['response']
        
        # Add to conversation history
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat(),
            'intent': intent_data['primary_intent'],
            'sentiment': intent_data['sentiment']
        })
        
        st.session_state.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat(),
            'confidence': generation_result.get('confidence', 0.0),
            'method': generation_result.get('method', 'generated'),
            'section': flow_context['section'],
            'phase': flow_context['phase']
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return "I apologize, but I'm having trouble processing that. Could you please repeat?"

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🤖 Retell AI Clone")
    st.caption("Script-Aware Call Agent")
    
    st.divider()
    
    # Script Upload
    st.subheader("📄 Upload Script")
    
    upload_method = st.radio(
        "Choose upload method:",
        ["Paste Text", "Upload File"],
        label_visibility="collapsed"
    )
    
    script_input = None
    
    if upload_method == "Paste Text":
        script_input = st.text_area(
            "Paste your call script here:",
            height=200,
            placeholder="Paste your script here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload script file",
            type=['txt'],
            help="Upload a .txt file containing your call script"
        )
        if uploaded_file:
            script_input = uploaded_file.read().decode('utf-8')
    
    # Load Script Button
    if script_input:
        if st.button("🚀 Load Script", type="primary", use_container_width=True):
            if load_script(script_input):
                st.success("✅ Script loaded successfully!")
                st.rerun()
    
    st.divider()
    
    # Call Controls
    if st.session_state.script_loaded:
        st.subheader("📞 Call Controls")
        
        if not st.session_state.call_active:
            if st.button("▶️ Start Call", type="primary", use_container_width=True):
                start_call()
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏸️ End Call", use_container_width=True):
                    end_call()
                    st.rerun()
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    end_call()
                    st.session_state.conversation_history = []
                    st.rerun()
        
        st.divider()
        
        # Script Info
        st.subheader("📊 Script Info")
        
        # Call type
        call_type_emoji = "📞" if st.session_state.call_type == 'inbound' else "📱"
        st.info(f"{call_type_emoji} **Call Type:** {st.session_state.call_type.upper()}")
        
        # Progress (if call active)
        if st.session_state.call_active and st.session_state.flow_engine:
            progress = st.session_state.flow_engine.get_progress()
            
            st.metric("Progress", f"{progress['progress_percentage']:.0f}%")
            st.progress(progress['progress_percentage'] / 100)
            
            with st.expander("📍 Current Position"):
                st.write(f"**Section:** {progress['current_section']}")
                st.write(f"**Phase:** {progress['phase']}")
                st.write(f"**Completed:** {len(progress['completed_sections'])}/{progress['total_sections']}")
        
        # Metadata
        metadata = st.session_state.parsed_script.get('metadata', {})
        if metadata:
            with st.expander("📋 Script Details"):
                for key, value in metadata.items():
                    if key != 'call_type':  # Already shown above
                        st.text(f"{key.title()}: {value}")

# ============================================================================
# MAIN AREA
# ============================================================================

# Title
if st.session_state.script_loaded and st.session_state.call_active:
    call_type_text = "📞 Inbound" if st.session_state.call_type == 'inbound' else "📱 Outbound"
    st.title(f"💬 Call Agent Conversation ({call_type_text})")
    
    # Show current section prominently
    if st.session_state.flow_engine:
        progress = st.session_state.flow_engine.get_progress()
        st.caption(f"📍 **Current Section:** {progress['current_section']} | **Phase:** {progress['phase']}")
else:
    st.title("💬 Call Agent Conversation")

# Instructions
if not st.session_state.script_loaded:
    st.info("👈 Please upload and load a script to begin")
    
    st.subheader("📝 How It Works")
    st.markdown("""
    This system follows your script **precisely** using a flow engine:
    
    1. **Upload your script** - Paste or upload your call script
    2. **Start the call** - System detects call type (inbound/outbound)
    3. **Follow the flow** - AI follows script sections in order
    4. **Stay on track** - System ensures conversation stays on script
    
    **Key Features:**
    - ✅ Follows script sections sequentially
    - ✅ Uses exact agent lines from script
    - ✅ Tracks conversation progress
    - ✅ Handles objections per script
    - ✅ Collects required data in order
    """)
    
    st.stop()

if not st.session_state.call_active:
    st.info("📞 Click 'Start Call' in the sidebar to begin")
    st.stop()

# Display conversation
for message in st.session_state.conversation_history:
    role = message['role']
    content = message['content']
    
    if role == 'user':
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            st.write(content)
            
            # Show section info quietly
            if 'section' in message:
                st.caption(f"📍 {message.get('section', 'Unknown')} | {message.get('phase', 'N/A')}")

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_user_message(prompt)
            st.write(response)
            
            # Show section
            if st.session_state.flow_engine:
                progress = st.session_state.flow_engine.get_progress()
                st.caption(f"📍 {progress['current_section']} | {progress['phase']}")
    
    # Rerun to update conversation
    st.rerun()