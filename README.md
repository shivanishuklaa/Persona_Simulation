# Sales Agent Simulation

## Overview
The Sales Agent Simulation is an AI-powered tool that simulates sales conversations with potential clients using Google's Gemini 2.0 Flash model. This application helps sales team practice and refine their pitch strategies by analyzing personality traits, communication styles, and preferences of potential clients.

## Features

### Personality Analysis
The system uses Gemini 2.0 Flash to analyze a potential client's profile and generate a comprehensive personality assessment, including:
- Personality overview
- Communication style
- Tips for selling and engagement
- Advanced personality insights (DISC, OCEAN, etc.)

### Simulated Conversations
The application simulates realistic sales conversations between:
- A persona agent that mimics the potential client's communication style
- A sales agent that presents the company's value proposition

The simulation includes dynamic question generation and objection handling, particularly focusing on clients who prefer full-time employees over flexible hiring solutions.

### Conversation Analysis
After each simulated conversation, the system provides:
- Refined analysis of the client's communication style and behavioral cues
- Tailored pitch strategies based on the conversation
- A final pitch customized to the client's specific needs and preferences

### Cold Email Generation
The system can draft personalized cold emails based on the insights gathered from the personality analysis and simulated conversation.

## Installation

### Prerequisites
- Python 3.12


### Setup
1. Clone the repository or download the source code
2. Create and activate a virtual environment (recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Unix/MacOS
   ```
3. Install the required dependencies:
   ```
   pip install -r requirement.txt
   ```
4. Create a `.env` file in the project root with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run PersonaAgent.py
   ```
2. Access the application in your web browser at http://localhost:8501
3. Configure the simulation:
   - Enter the persona name
   - Provide context information about the persona (professional background, experience, etc.)
4. Click "Run Analysis & Simulate Conversation" to start the simulation
5. Review the outputs:
   - Detailed personality analysis
   - Simulated conversation log
   - Refined analysis and pitch strategy
   - Final tailored pitch
   - Cold email draft

## Key Components

- **Conversation Review Agent**: Analyzes conversation dynamics and provides actionable insights
- **Gemini Analysis Agent**: Generates comprehensive personality profiles based on provided context
- **Persona Agent**: Simulates responses from the potential client based on their personality profile
- **Dynamic Persona Questions**: Generates realistic questions that the persona might ask

## Use Cases

- **Sales Training**: Help sales representatives practice handling objections and tailoring pitches
- **Client Research**: Analyze potential clients before meetings to understand their communication style
- **Cold Outreach Optimization**: Generate personalized cold emails based on detailed personality insights
- **Pitch Refinement**: Test different pitch strategies in a simulated environment

## Notes

- This application requires a valid Google API key with access to the Gemini 2.0 Flash model
- The quality of the personality analysis depends on the richness of the provided context
- The simulation is designed to handle the common objection that clients prefer full-time employees over flexible hiring solutions