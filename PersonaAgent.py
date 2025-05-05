from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import Google's Generative AI library for Gemini
import google.generativeai as genai

# Configure Gemini API 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------------------------------------------------
# Utility Function to Extract Text
# ---------------------------------------------------------------------
def extract_text(response):
    """Safely extract text from Gemini response."""
    if hasattr(response, 'text'):
        return response.text
    return str(response)

# ---------------------------------------------------------------------
# Conversation Review Agent
# ---------------------------------------------------------------------
def create_conversation_review_agent():
    """
    Creates an agent that can review and provide insights on the conversation
    based on the conversation history.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    def review_conversation(conversation_log, review_focus=None):
        """
        Review the conversation with optional focus areas.
        
        :param conversation_log: Full conversation history
        :param review_focus: Optional specific area to focus on (e.g., 'communication', 'sales strategy')
        :return: Detailed review and insights
        """
        prompt = ("You are an advanced conversation analysis AI. Carefully review the following conversation "
                  "and provide a comprehensive analysis. ")
        
        if review_focus:
            prompt += f"Pay special attention to the {review_focus} aspects of the conversation. "
        
        prompt += ("\n\nKey areas to analyze:\n"
                   "1. Communication Dynamics\n"
                   "2. Effectiveness of Sales Approach\n"
                   "3. Client's Underlying Needs and Concerns\n"
                   "4. Missed Opportunities\n"
                   "5. Potential Improvements\n\n"
                   f"Conversation Log:\n{conversation_log}\n\n"
                   "Provide a detailed, objective analysis with actionable insights.")
        
        response = model.generate_content(prompt)
        return response.text
    
    return review_conversation

# ---------------------------------------------------------------------
# Gemini-2.0-Flash for Personality/Behavioral Analysis
# ---------------------------------------------------------------------
def create_gemini_analysis_agent(person_name, context):
    """
    Creates a specialized Gemini agent to analyze the provided context
    and generate a comprehensive personality profile.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = (f"You are an advanced personality analysis AI specialized in generating detailed personality insights. "
              f"Your task is to analyze the provided context for {person_name} and create a comprehensive profile with insights, "
              "including personality type, traits, communication style, buying preferences, and suggestions for effective engagement.\n\n"
              "Context:\n{context}\n\n"
              "Based on the above, generate a structured personality profile with the following sections:\n"
              "1. Personality Overview\n2. Personality Compatibility\n3. Communication Style\n"
              "4. Tips for Selling and Engagement\n5. Advanced Insights (DISC, OCEAN, etc.).\n\n"
              "Analysis:").format(person_name=person_name, context=context)
    
    response = model.generate_content(prompt)
    return response.text

# ---------------------------------------------------------------------
# Persona Agent
# ---------------------------------------------------------------------
def create_persona_agent(person_name, analysis_text=""):
    """
    Creates a persona agent for 'person_name' that can optionally include
    the Gemini analysis_text to inform responses about the person's style.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_response(chat_history, user_input):
        prompt = (f"You are an AI version of {person_name}. Below is your personality analysis:\n"
                  f"{analysis_text}\n\n"
                  "You are participating in a virtual meeting with a BeGig sales representative. "
                  "You have reviewed your public content and are ready to share your opinions. "
                  "Keep your responses natural, thoughtful, and reflective of your style, background, and expertise.\n\n"
                  f"Meeting Conversation History:\n{chat_history}\n\n"
                  f"Query: {user_input}\n\n"
                  "Response:")
        
        response = model.generate_content(prompt)
        return response.text
    
    return generate_response

# ---------------------------------------------------------------------
# Sales Conversation Agent
# ---------------------------------------------------------------------
def create_sales_conversation_agent():
    """
    Sales agent representing BeGig, explaining the value proposition to the persona.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_response(chat_history, user_input):
        prompt = ("You are a sales expert representing BeGig using the Gemini 2.0 Flash model. You are in a virtual meeting with the client. "
                  "Your goal is to clearly and persuasively explain BeGig's value proposition while being sensitive to the client's preferences. "
                  "Keep in mind that the client typically prefers full-time employees but may be open to hearing how flexible solutions can also benefit them.\n\n"
                  f"Meeting Conversation History:\n{chat_history}\n\n"
                  f"Sales Query: {user_input}\n\n"
                  "Sales Response:")
        
        response = model.generate_content(prompt)
        return response.text
    
    return generate_response

# ---------------------------------------------------------------------
# Dynamic Persona Questions
# ---------------------------------------------------------------------
def generate_dynamic_persona_questions(person_name, conversation_log=""):
    """
    Generates dynamic, curiosity-driven, reflective, and open-ended questions for the persona
    to inquire about deeper aspects of BeGig's offerings.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = ("You are an insightful AI that helps generate dynamic, curiosity-driven, and reflective questions for a persona in a virtual meeting. "
              f"Based on the persona name '{person_name}' and the following conversation context:\n\n"
              f"{conversation_log}\n\n"
              "Generate 3 to 5 open-ended questions that {person_name} might ask to gain deeper insights about BeGig's offerings. "
              "Focus on topics such as talent matching, how flexible hires can transition to full-time roles, compliance across markets, and success stories. "
              "Each question should be clear and engaging.")
    
    response = model.generate_content(prompt)
    text_response = response.text
    dynamic_questions = [q.strip() for q in text_response.split('\n') if q.strip()]
    return dynamic_questions

# ---------------------------------------------------------------------
# Simulated Conversation with Dynamic Flow
# ---------------------------------------------------------------------
def simulate_meeting_conversation_with_fulltime_preference(persona_agent, sales_agent, persona_name="Unni Koroth", max_rounds=5):
    """
    Simulates a conversation between the persona and sales agent.
    The conversation includes dynamic greetings, preference statements, and interactive Q&A.
    """
    conversation_log = "Meeting Conversation Start:\n"
    st.info(f"--- Virtual Meeting Begins: {persona_name} with BeGig Sales ---")
    
    # Persona greeting
    greeting_query = "Please provide a friendly greeting, introducing yourself and your role."
    persona_greeting = persona_agent(conversation_log, greeting_query)
    conversation_log += f"\n{persona_name}: {persona_greeting}\n"
    
    # Sales agent greeting
    sales_greeting_query = "Please greet the client warmly and ask about their biggest challenge in scaling their team."
    sales_greeting = sales_agent(conversation_log, sales_greeting_query)
    conversation_log += f"\nBeGig Sales: {sales_greeting}\n"
    
    # Persona states full-time hiring preference
    fulltime_query = "Please state your preference for full-time employees over freelancers, and explain why full-time hires offer more stability for your projects."
    fulltime_preference = persona_agent(conversation_log, fulltime_query)
    conversation_log += f"\n{persona_name}: {fulltime_preference}\n"
    
    # Sales agent objection handling and synergy exploration
    objection_query = (
        "Based on the client's preference for full-time hires, please provide an objection handling message "
        "that explains how BeGig's hybrid solution can start with flexible hires that eventually transition to full-time roles. "
        "Also, ask about the challenges the client faces with their current full-time hiring process."
    )
    sales_objection = sales_agent(conversation_log, objection_query)
    conversation_log += f"\nBeGig Sales: {sales_objection}\n"
    
    # Generate dynamic questions and responses
    dynamic_questions = generate_dynamic_persona_questions(persona_name, conversation_log)
    rounds = min(max_rounds, len(dynamic_questions))
    
    for i in range(rounds):
        # Persona asks a dynamic question
        persona_question = dynamic_questions[i]
        persona_text = persona_agent(conversation_log, persona_question)
        conversation_log += f"\n{persona_name}: {persona_text}\n"
        
        # Sales dynamic response with a changing scenario
        if i == 0:
            scenario = "ai_matching"
        elif i == 1:
            scenario = "success_story"
        elif i == 2:
            scenario = "compliance"
        else:
            scenario = "success_story"
        sales_query = f"Based on the conversation so far, please address the following question dynamically: '{persona_question}'. Provide an answer related to {scenario}."
        sales_text = sales_agent(conversation_log, sales_query)
        conversation_log += f"\nBeGig Sales: {sales_text}\n"
    
    return conversation_log

# ---------------------------------------------------------------------
# Refined Analysis & Final Tailored Pitch
# ---------------------------------------------------------------------
def create_refined_analysis(person_name, conversation_log):
    """
    Analyzes the conversation log to extract behavioral cues, communication style,
    and decision-making preferences.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = (f"You are an AI analyst who has observed a virtual meeting conversation with {person_name}, a potential client. "
              "Based on the conversation history, analyze their communication style, personality traits, and behavioral cues. "
              "Then, provide a summary of their style and suggest the best way to pitch BeGig to them, keeping in mind their preference "
              "for full-time hires and overall strategic goals.\n\n"
              f"Meeting Conversation History:\n{conversation_log}\n\n"
              "Provide a detailed analysis focusing on communication style, key motivations, and tailored pitch strategies.")
    
    response = model.generate_content(prompt)
    return response.text

def generate_final_tailored_pitch(refined_analysis_text, conversation_log):
    """
    Uses the refined analysis to generate a final tailored sales pitch for BeGig.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = ("Based on the following refined analysis and conversation history, create a final, tailored sales pitch "
              "that addresses the client's specific needs, communication style, and potential concerns:\n\n"
              f"Refined Analysis:\n{refined_analysis_text}\n\n"
              f"Conversation History:\n{conversation_log}\n\n"
              "Craft a compelling, personalized pitch that highlights how BeGig can solve their specific challenges.")
    
    response = model.generate_content(prompt)
    return response.text

# ---------------------------------------------------------------------
# Cold Email Drafting
# ---------------------------------------------------------------------
def draft_cold_email(final_pitch, refined_analysis, analysis_text):
    """
    Uses the final tailored pitch, refined analysis, and detailed personality analysis
    to draft a cold email.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = ("You are a sales expert crafting a cold email to a prospective client. "
              "Even though no meeting has taken place, you have in-depth insights about the client's personality, "
              "communication style, and business needs derived from detailed analysis. Use the following insights to draft a compelling email:\n\n"
              f"Detailed Personality Analysis:\n{analysis_text}\n\n"
              f"Refined Analysis & Pitch Strategy:\n{refined_analysis}\n\n"
              f"Final Tailored Pitch:\n{final_pitch}\n\n"
              "Draft a cold email that introduces yourself, explains why you believe the client would benefit from BeGig's solution, "
              "and invites them for a discussion. Do not reference any prior meeting or conversation—present it as a genuine cold outreach email. "
              "Ensure the tone is professional, insightful, and engaging.")
    
    response = model.generate_content(prompt)
    return response.text

# ---------------------------------------------------------------------
# Streamlit UI Integration
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Persona Simulation", layout="wide")
    
    # Initialize session state for conversation records
    if 'conversation_records' not in st.session_state:
        st.session_state.conversation_records = []
    
    st.title("Persona Simulation")
    st.markdown(
        "This simulation processes the following outputs in sequence:\n\n"
        "1. Detailed Personality Analysis (Gemini 2.0 Flash)\n"
        "2. Final Conversation Log (Simulated Meeting)\n"
        "3. Refined Analysis & Pitch Strategy\n"
        "4. Final Tailored Pitch\n"
        "5. Cold Email Draft (Cold outreach using the above insights)\n"
    )
    
    st.sidebar.header("Configuration")
    person_name = st.sidebar.text_input("Persona Name", " ")
    context_text = st.sidebar.text_area("Persona Context", height=300, 
        value="""Name: Unni Koroth
Title: Founder & CEO – Othor AI | World's Fastest Business Intelligence
Location: Bengaluru, Karnataka, India
Contact: ukknair@icloud.com, LinkedIn: www.linkedin.com/in/unni-koroth

Professional Summary:
I am the Founder and CEO of Othor AI. This is my second coming as a startup founder and CEO, and this time, I come armed with coding skills.
As a visionary tech entrepreneur and an MIT TR35 awardee, I'm passionate about pushing the boundaries of innovation across various fields. At Othor AI, I'm on a mission to redefine decision making by equipping leaders with insights delivered at lightning speed—leveraging cutting-edge data science and decision science.
Leading Othor AI as a remote-first company, I focus on transforming how businesses utilize data to provide real-time, impactful insights to visionary leaders worldwide. My previous roles at Whatfix and Foradian Technologies demonstrate my ability to drive growth and turn bold ideas into global movements.

Top Skills:
- Circuit Design
- Digital Communication
- Analog Signal Processing

Honors & Awards:
- MIT TR35 (2012)

Professional Experience:
Othor AI:
  - Founder & CEO (June 2024 – Present, Bangalore Urban, Karnataka, India)
    Highlights: Transforming business data interactions with advanced AI and machine learning; pioneering augmented analytics that turn complex data into clear, actionable stories.
Whatfix:
  - Business Intelligence Project Lead (March 2023 – May 2024, Bengaluru, Karnataka, India)
    Highlights: Led strategic initiatives to future-proof the Enterprise BI platform with augmented analytics; optimized SaaS licensing spend through automation.
  - Director – Labs (November 2021 – February 2023, Bengaluru, Karnataka, India)
    Highlights: Established the Labs division; transformed innovative concepts into scalable MVPs.
Foradian Technologies:
  - CEO & Cofounder (January 2009 – March 2019, Bengaluru, Karnataka, India)
    Highlights: Led the company from inception to becoming a global leader in EdTech ERP software; successfully raised multiple rounds of funding and scaled operations internationally; pioneered growth strategies that resulted in exponential user adoption.
Idea Cellular Ltd:
  - Assistant Manager (July 2006 – December 2008, Kerala, India)
    Highlights: Managed telecommunications network and operations for a district; improved network reliability and reduced costs through strategic management.

Education:
- Golden Gate University: Doctor of Business Administration (DBA) in Generative AI (Dec 2023 – Dec 2027)
- Indian Institute of Technology, Delhi: CEP Certificate Program, Advanced Certification in Data Science and Decision Science (Dec 2023 – Dec 2024)
- Indian School of Business: Certificate in Understanding Public Policy in India, Public Policy Analysis (Mar 2024 – Jun 2024)
- Kannur University: Bachelor of Technology (BTech) in Electronics and Communications Engineering (May 2002 – May 2006)
""")
    
    # Conversation Review Agent in Sidebar
    review_agent = create_conversation_review_agent()
    
    # View Conversation Records Button
    if st.sidebar.button("View Conversation Records"):
        if st.session_state.conversation_records:
            st.sidebar.subheader("Conversation Records")
            for i, record in enumerate(st.session_state.conversation_records, 1):
                with st.sidebar.expander(f"Conversation {i}"):
                    st.write("Persona:", record['persona_name'])
                    st.text_area("Conversation Log", record['conversation_log'], height=200)
                    st.text_area("Conversation Review", record['conversation_review'], height=200)
        else:
            st.sidebar.info("No conversation records available. Run a simulation first.")
    
    if st.button("Run Analysis & Simulate Conversation"):
        with st.spinner("Running personality analysis using Gemini 2.0 Flash..."):
            # Create Gemini agent
            analysis_text = create_gemini_analysis_agent(person_name, context_text)
        
        # Create Persona and Sales Agents
        persona_agent = create_persona_agent(person_name, analysis_text=analysis_text)
        sales_agent = create_sales_conversation_agent()
        
        with st.spinner("Simulating conversation..."):
            conversation_log = simulate_meeting_conversation_with_fulltime_preference(
                persona_agent, sales_agent, persona_name=person_name, max_rounds=4
            )
        
        with st.spinner("Analyzing conversation for tailored pitch..."):
            refined_analysis = create_refined_analysis(person_name, conversation_log)
            final_pitch = generate_final_tailored_pitch(refined_analysis, conversation_log)
        
        with st.spinner("Drafting cold email..."):
            cold_email = draft_cold_email(final_pitch, refined_analysis, analysis_text)
        
        # Review the conversation
        with st.spinner("Generating conversation review..."):
            conversation_review = review_agent(conversation_log)
        
        # Store conversation record in session state
        conversation_record = {
            'persona_name': person_name,
            'conversation_log': conversation_log,
            'conversation_review': conversation_review,
            'analysis_text': analysis_text,
            'final_pitch': final_pitch,
            'cold_email': cold_email
        }
        st.session_state.conversation_records.append(conversation_record)
        
        # Display the processed outputs in order
        st.subheader("1. Detailed Personality Analysis")
        st.text_area("Personality Analysis (Gemini 2.0 Flash)", analysis_text, height=300)
        
        st.subheader("2. Final Conversation Log")
        st.text_area("Conversation Log", conversation_log, height=400)
        
        st.subheader("3. Conversation Review")
        st.text_area("Conversation Review", conversation_review, height=250)
        
        st.subheader("4. Refined Analysis & Pitch Strategy")
        st.text_area("Refined Analysis", refined_analysis, height=200)
        
        st.subheader("5. Final Tailored Pitch")
        st.text_area("Final Tailored Pitch", final_pitch, height=150)
        
        st.subheader("6. Cold Email Draft")
        st.text_area("Cold Email Draft", cold_email, height=250)
    
if __name__ == "__main__":
    main()