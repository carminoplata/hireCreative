import streamlit as st
import cohere
from dotenv import load_dotenv
from datetime import datetime
import os
import openai
load_dotenv()

co = cohere.Client(os.getenv('COHERE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')
openAiExpiration = datetime.strptime(os.getenv('OPENAI_DEADLINE'), 
                                     '%Y-%m-%d %H:%M')
                                    
if 'output' not in st.session_state:
    st.session_state['output'] = 'Output:'
    st.session_state['openaiResults'] = 'openaiResults:'
    st.session_state['tokenlimits'] = float(os.getenv('OPENAI_TOKEN_AMOUNT'))
    st.session_state['expires'] = openAiExpiration
    missingDays = openAiExpiration  - datetime.now()
    st.text(f'Missing Days {missingDays.days}')
    st.session_state['availableTokens'] = (
        st.session_state.tokenlimits / missingDays.days
    )


def find_roles(target, industry):
    if not target:
        st.session_state['output'] = ""
    else:
        prompt=f"""Given an idea and its industry, this program will define which are the roles to develop it
        Industry: Retail
        Idea: Develop an ecommerce for selling my innovative blazor in 1 month
        Roles: FullStack Developer, Designer
        --
        Industry: Automotive
        Idea: Develop a plaftorm for the live streaming of the car race taking the videos from the car and 
        showing them on a mobile app in 1 year
        Roles: Hardware Engineer, Backend Engineer, Designer, Android Developer, iOS Developer, Mechanical Engineer,
        Product Manager
        --
        Industry: Healthcare
        Idea: Develop a mobile app for sharing in a safe way healthcare documents between doctors and their patients
        Roles: Fullstack Developer, Designer, Android Developer, iOS Developer, Digital Lawyer, Doctor, Cybersecurity Architect
        --
        Industry: {industry}
        Idea: {target}
        Roles:"""

        response = co.generate(
            model='xlarge',
            prompt=prompt,
            max_tokens=100,
            temperature=0.5,
            k=0, 
            p=1, 
            frequency_penalty=0, 
            presence_penalty=0, 
            stop_sequences=["--"], 
            return_likelihoods='NONE'
        )
        st.session_state['output'] = response.generations[0].text

def find_rolesByOpenAi(target, industry):
    if not target:
        st.session_state['openaiResults'] = ""
    else:
        prompt = f"""How many resources do we need to {target}?
            Which skills do they have got?
            Make a list of tasks?"""

        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=300,
            temperature=0.5,
            top_p=0.5)

        st.session_state['openaiResults'] = response.choices[0].text
    

sectors=['Aerospace', 'Fashion', 'Medical', 'Automotive', 'Consultancy', 
         'Digital Transformation', 'CyberSecurity', 'Oil and gas',
         'Energy', 'Pharmaceutical', 'Defense', 'Software']
st.title('CreativeLink')
st.subheader('The Italian creative are ready to make your dreams happen')
st.text(f'Available Tokens: {st.session_state.availableTokens}')
industry = st.radio("What\'s the sector of your idea", sectors)
input = st.text_area('Describe briefly your idea', height=100)

st.button('Find Roles', on_click = find_roles(input, industry))
st.write(st.session_state.output)

st.button('Find Roles via ChatGpt', on_click = find_rolesByOpenAi(input, industry))
st.write(st.session_state.openaiResults)