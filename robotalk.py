# Bring in deps
from decouple import config
import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
import elevenlabs
from elevenlabs.client import ElevenLabs
from elevenlabs import *
from elevenlabs import save, voices
import urllib.parse
import feedparser
from datetime import datetime
from pydub import AudioSegment
import nltk
import openai
import re
from openai import OpenAI

# Access the environment variables
API_KEYS = {
    'OPENAI_API_KEY': config('OPENAI_API_KEY'),
    'ELEVENLABS_API_KEY': config('ELEVENLABS_API_KEY'),
    'ELEVENLABS_VOICE_1_ID': config('ELEVENLABS_VOICE_1_ID'),
    'ELEVENLABS_VOICE_2_ID': config('ELEVENLABS_VOICE_2_ID'),
    'ELEVENLABS_VOICE_3_ID': config('ELEVENLABS_VOICE_3_ID'),
    'ELEVENLABS_VOICE_4_ID': config('ELEVENLABS_VOICE_4_ID'),
    'ELEVENLABS_VOICE_5_ID': config('ELEVENLABS_VOICE_5_ID'),
    'ELEVENLABS_VOICE_6_ID': config('ELEVENLABS_VOICE_6_ID'),
    'ELEVENLABS_VOICE_7_ID': config('ELEVENLABS_VOICE_7_ID'),
    'ELEVENLABS_VOICE_8_ID': config('ELEVENLABS_VOICE_8_ID'),
    'GOOGLE_CSE_ID': config('CUSTOM_SEARCH_ENGINE_ID'),
    'GOOGLE_API_KEY': config('GOOGLE_API_KEY'),
}

# Voice settings for language learning
voice_settings = {
    'Teacher_Voice': {
        'voice_id': API_KEYS['ELEVENLABS_VOICE_1_ID'],
        'settings': {
            'stability': 0.8,
            'similarity_boost': 0.7,
            'style': 0.3,
            'speaking_rate': 0.9
        }
    },
    'Student_Voice': {
        'voice_id': API_KEYS['ELEVENLABS_VOICE_2_ID'],
        'settings': {
            'stability': 0.75,
            'similarity_boost': 0.7,
            'style': 0.4,
            'speaking_rate': 1.0
        }
    }
}

# Application Framework
st.title('English Learning Podcast Creator')

# Basic Settings
topic_category = st.selectbox("Select Topic Category", [
    "Daily Conversations",
    "Business English",
    "Travel English",
    "Social Situations",
    "Academic English"
])

topic = st.text_input("Enter specific topic", value="Asking for Directions")

# Host Settings
host1_name = st.text_input("Host 1 Name", value="Lisa")
host2_name = st.text_input("Host 2 Name", value="Kevin")

# Teaching Style Settings
teaching_style = st.selectbox("Teaching Style", [
    "Casual and Friendly",
    "Professional and Structured",
    "Fun and Interactive",
    "Academic and Detailed"
])

# Language Level
target_level = st.selectbox("Target English Level", [
    "Beginner (A1-A2)",
    "Intermediate (B1-B2)",
    "Advanced (C1-C2)"
])

# Initialize environment
os.environ.update(API_KEYS)

# Initialize components
google_search_tool = GoogleSearchAPIWrapper()

# Initialize OpenAI API
openai_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.7)

# Define templates
title = PromptTemplate.from_template(
    "Create an engaging title for an English learning podcast about {topic}"
)

script = PromptTemplate.from_template("""
    Create an English learning podcast script with the following structure:
    
    Topic: {topic}
    Teaching Style: {teaching_style}
    Target Level: {target_level}
    Host 1: {host1_name}
    Host 2: {host2_name}
    Duration: 10-15 minutes
    
    Required Sections:
    1. [Introduction]
    - Warm welcome
    - Topic introduction
    - Learning objectives
    
    2. [Vocabulary Preview]
    - 3-4 key words/phrases
    - Clear examples
    - Natural usage
    
    3. [Dialogue]
    - Natural conversation using target language
    - Clear speaker labels
    - Realistic scenario
    
    4. [Language Focus]
    - Breakdown of key phrases
    - Grammar explanations if needed
    - Common usage tips
    
    5. [Practice Section]
    - Example variations
    - Cultural notes
    - Learning tips
    
    6. [Conclusion]
    - Review of key points
    - Practice encouragement
    - Preview next episode
    
    Format each section with clear [SECTION] markers.
    Use natural, conversational language appropriate for {target_level} level.
    Use {host1_name}: and {host2_name}: to clearly mark the speaker for each line.
""")

news = PromptTemplate.from_template("Summarize teaching resources and methods for: {story}")
research = PromptTemplate.from_template("Research teaching methodologies and common challenges for: {topic}")

# Initialize chains
chains = {
    'title': LLMChain(llm=openai_llm, prompt=title, verbose=True, output_key='title'),
    'script': LLMChain(llm=openai_llm, prompt=script, verbose=True, output_key='script'),
    'news': LLMChain(llm=openai_llm, prompt=news, verbose=True, output_key='summary'),
    'research': LLMChain(llm=openai_llm, prompt=research, verbose=True, output_key='research'),
}

# Initialize session state
if 'script' not in st.session_state:
    st.session_state.script = "Script will appear here"
if 'title' not in st.session_state:
    st.session_state.title = "Podcast Title Will Appear Here"
if 'news' not in st.session_state:
    st.session_state.news = ""
if 'research' not in st.session_state:
    st.session_state.research = ""
if 'podcast_dir' not in st.session_state:
    st.session_state.podcast_dir = ""
if 'sections' not in st.session_state:
    st.session_state.sections = []

def extract_news_text(url):
    """Extract the text of a news story given its URL."""
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    story_text = ' '.join([p.get_text() for p in paragraphs])
    tokens = nltk.word_tokenize(story_text)
    tokens = tokens[:2800]
    story_text = ' '.join(tokens)
    return story_text

def get_top_news_stories(topic, num_stories=5):
    topic = urllib.parse.quote_plus(topic)
    feed = feedparser.parse(f'https://news.google.com/rss/search?q=english+teaching+{topic}')
    return feed.entries[:num_stories]

def summarize_news_stories(stories):
    summaries = []
    total_tokens = 0
    for story in stories:
        url = story.get('link', '')
        if url:
            story_text = extract_news_text(url)
            summary = chains['news'].run(story_text)
            summary_tokens = len(summary.split())
            if total_tokens + summary_tokens <= 10000:
                summaries.append(summary)
                total_tokens += summary_tokens
            else:
                break
    return summaries

def validate_inputs(topic, host1_name, host2_name):
    return all([topic, host1_name, host2_name])

def create_podcast_directory():
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    podcast_dir = f"Podcast_{date_time}"
    if not os.path.exists(podcast_dir):
        os.makedirs(podcast_dir)
    return podcast_dir

def optimize_voice_for_learning(voice_id, is_teacher=True):
    """Optimize voice settings for language learning"""
    settings = voice_settings['Teacher_Voice' if is_teacher else 'Student_Voice']
    return {
        'voice_id': voice_id,
        'settings': settings['settings']
    }

def process_script_sections(script_text):
    """Process script into sections for audio generation"""
    sections = re.split(r'\[([^\]]+)\]', script_text)
    processed_sections = []
    
    for i in range(1, len(sections), 2):
        section_name = sections[i]
        content = sections[i+1].strip()
        processed_sections.append({
            'section': section_name,
            'content': content,
            'speaking_rate': 0.9 if section_name == 'Vocabulary Preview' else 1.0
        })
    
    return processed_sections

def test_openai_api():
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=10
        )
        st.success("OpenAI API connection successful!")
        return True
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return False

# Voice selection UI
st.subheader("Voice Settings")
col1, col2 = st.columns(2)

voice_options = {
    'Voice 1': 'ELEVENLABS_VOICE_1_ID',
    'Voice 2': 'ELEVENLABS_VOICE_2_ID',
    'Voice 3': 'ELEVENLABS_VOICE_3_ID',
    'Voice 4': 'ELEVENLABS_VOICE_4_ID',
    'Voice 5': 'ELEVENLABS_VOICE_5_ID',
    'Voice 6': 'ELEVENLABS_VOICE_6_ID',
    'Voice 7': 'ELEVENLABS_VOICE_7_ID',
    'Voice 8': 'ELEVENLABS_VOICE_8_ID',
}

with col1:
    host1_voice = st.selectbox(
        f"Select voice for {host1_name}", 
        list(voice_options.keys()),
        key="host1_voice"
    )

with col2:
    host2_voice = st.selectbox(
        f"Select voice for {host2_name}", 
        list(voice_options.keys()),
        key="host2_voice"
    )

def convert_script_to_audio(script_text, podcast_dir):
    # Set API key first
    client = ElevenLabs(api_key=API_KEYS['ELEVENLABS_API_KEY'])
    
    voices = {
        host1_name: API_KEYS.get(voice_options[host1_voice]),
        host2_name: API_KEYS.get(voice_options[host2_voice])
    }
    
    combined_segments = []
    
    try:
        sections = process_script_sections(script_text)
        for section in sections:
            st.write(f"Processing section: {section['section']}")
            lines = section['content'].split('\n')
            for line in lines:
                if ':' in line:
                    speaker, text = line.split(':', 1)
                    speaker = speaker.strip()
                    text = text.strip()
                    
                    if speaker in voices and text:
                        st.write(f"Generating audio for {speaker}: {text[:50]}...")
                        voice_id = voices[speaker]
                        
                        try:
                            # Generate audio using generate_stream and collect all chunks
                            audio_stream = client.generate(
                                text=text,
                                voice=voice_id,
                                model="eleven_multilingual_v2"
                            )
                            
                            # Create temporary file
                            temp_file = f"{podcast_dir}/temp_{len(combined_segments)}.mp3"
                            
                            # Handle generator stream and write to file
                            if hasattr(audio_stream, '__iter__'):
                                with open(temp_file, 'wb') as f:
                                    for chunk in audio_stream:
                                        f.write(chunk)
                            elif isinstance(audio_stream, bytes):
                                with open(temp_file, 'wb') as f:
                                    f.write(audio_stream)
                            else:
                                st.error(f"Unexpected audio format: {type(audio_stream)}")
                                continue
                            
                            # Load as AudioSegment
                            segment = AudioSegment.from_mp3(temp_file)
                            combined_segments.append(segment)
                            
                            # Clean up temp file
                            os.remove(temp_file)
                            
                            st.write(f"Successfully generated audio for: {speaker}")
                            
                        except Exception as e:
                            st.error(f"Error generating audio for line: {str(e)}")
                            st.write(f"Failed text: {text}")
                            st.write(f"Voice ID used: {voice_id}")
                            continue
        
        if combined_segments:
            final_audio = combined_segments[0]
            for segment in combined_segments[1:]:
                final_audio = final_audio + segment
            
            audio_file = f"{podcast_dir}/podcast.mp3"
            final_audio.export(audio_file, format="mp3")
            st.success("Audio generation completed!")
            return [audio_file]
        else:
            st.error("No audio segments were generated")
            return []
            
    except Exception as e:
        st.error(f"Error in audio generation: {str(e)}")
        st.write(f"Voice mapping used: {voices}")
        return []
    
    
# Start application logic
if not test_openai_api():
    st.stop()

if st.button('Research') and validate_inputs(topic, host1_name, host2_name):
    stories = get_top_news_stories(topic)
    news_summaries = summarize_news_stories(stories)
    research_summary = chains['research'].run(topic=topic)
    st.session_state.research = research_summary
    st.session_state.podcast_dir = create_podcast_directory()
    with open(f"{st.session_state.podcast_dir}/podcast_research.txt", 'w') as f:
        f.write(st.session_state.research)
    st.success(f"Research saved in {st.session_state.podcast_dir}/podcast_research.txt")

if st.button('Generate Script') and validate_inputs(topic, host1_name, host2_name):
    title_result = chains['title'].run(topic=topic)
    st.session_state.title = title_result

    script_result = chains['script'].run(
        topic=topic,
        teaching_style=teaching_style,
        target_level=target_level,
        host1_name=host1_name,
        host2_name=host2_name
    )
    st.session_state.script = script_result
    
    # Process and store script sections
    st.session_state.sections = process_script_sections(script_result)

    edited_script = st.text_area('Edit the Script', st.session_state.script, key='edit_script', height=300)
    if edited_script != st.session_state.script:
        st.session_state.script = edited_script

if st.button('Save Script') and 'edit_script' in st.session_state:
    edited_script = st.session_state.edit_script
    st.session_state.script = edited_script
    with open(f"{st.session_state.podcast_dir}/podcast_script.txt", 'w') as f:
        f.write(edited_script)
    st.success(f"Edited script saved in {st.session_state.podcast_dir}/podcast_script.txt")
    st.write(f'Script: \n{st.session_state.script}')

if st.button('Create Podcast') and st.session_state.script:
    audio_files = convert_script_to_audio(st.session_state.script, st.session_state.podcast_dir)
    if audio_files:
        st.audio(audio_files[0], format='audio/mp3')

with st.expander('Research Notes'):
    st.write(st.session_state.research)

with st.expander('Script'):
    st.write(f'Title: {st.session_state.title}')
    st.write(f'Script: \n{st.session_state.script}')