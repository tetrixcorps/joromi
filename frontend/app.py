import streamlit as st
import aiohttp
import asyncio
import json
from PIL import Image
import io
import base64
import plotly.graph_objects as go
import pandas as pd
from auth import Auth
import altair as alt
from datetime import datetime, timedelta
import numpy as np
from model_handler import DolphinHandler
import logging

logger = logging.getLogger(__name__)

class JoromiGPTUI:
    def __init__(self):
        self.api_url = "http://localhost:8000"  # Gateway service URL
        self.auth = Auth()
        self.dolphin = DolphinHandler()
        
    async def chat_request(self, messages):
        """Send chat request to API"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/chat",
                json={"messages": messages}
            ) as response:
                return await response.json()

    async def stream_chat(self, message):
        """Stream chat responses"""
        try:
            system_prompt = "You are a helpful AI assistant."
            response = await self.dolphin.generate_response(message, system_prompt)
            
            # Simulate streaming by yielding chunks
            words = response.split()
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i:i+3])
                yield chunk + " "
                await asyncio.sleep(0.05)  # Add slight delay for natural feel
                
        except Exception as e:
            logger.error(f"Error in stream_chat: {e}")
            raise

    def show_metrics(self):
        """Display usage metrics and visualizations"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Calls Today", "1.2K", "+12%")
        with col2:
            st.metric("Response Time", "245ms", "-8%")
        with col3:
            st.metric("Active Users", "156", "+23%")
            
        # Usage over time
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'calls': np.random.randint(800, 1500, size=len(dates))
        })
        
        chart = alt.Chart(data).mark_line().encode(
            x='date:T',
            y='calls:Q',
            tooltip=['date', 'calls']
        ).properties(
            title='API Usage Over Time'
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Model performance
        fig = go.Figure(data=[
            go.Bar(name='Response Time', x=['ASR', 'TTS', 'Translation', 'Banking'],
                  y=[245, 180, 150, 320])
        ])
        fig.update_layout(title='Model Response Times (ms)')
        st.plotly_chart(fig)

    def show_settings(self):
        """Display user settings"""
        st.sidebar.title("Settings")
        
        # Model selection
        model = st.sidebar.selectbox(
            "Select Model",
            ["Banking Assistant", "General Chat", "Translation"]
        )
        
        # Language settings
        language = st.sidebar.selectbox(
            "Interface Language",
            ["English", "Spanish", "French", "Chinese"]
        )
        
        # Theme selection
        theme = st.sidebar.radio(
            "Theme",
            ["Light", "Dark", "System"]
        )
        
        # User profile
        st.sidebar.divider()
        st.sidebar.subheader("User Profile")
        st.sidebar.text(f"User: {st.session_state.user}")
        st.sidebar.text(f"Role: {st.session_state.role}")
        
        if st.sidebar.button("Logout"):
            del st.session_state.user
            del st.session_state.role
            st.rerun()

    def show_image_analysis(self):
        """Show image analysis interface"""
        st.subheader("Image Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=["jpg", "png", "jpeg"]
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image")
        
        with col2:
            prompt = st.text_input("Enter your prompt")
            
            if uploaded_file and prompt:
                with st.spinner("Analyzing image..."):
                    try:
                        result = asyncio.run(
                            self.dolphin.process_image(image, prompt)
                        )
                        st.write("Analysis Result:")
                        st.write(result)
                        
                        # Add to chat history
                        st.session_state.messages.extend([
                            {
                                "role": "user",
                                "content": f"[Image Analysis] {prompt}",
                                "type": "image"
                            },
                            {
                                "role": "assistant",
                                "content": result,
                                "type": "text"
                            }
                        ])
                        
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")

    def run(self):
        """Main application loop"""
        # Check authentication
        if "user" not in st.session_state:
            self.auth.show_login_page()
            return
            
        st.title("JoromiGPT Interface")
        self.show_settings()
        
        # Updated tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Chat",
            "Image Analysis",
            "Analytics",
            "History"
        ])
        
        with tab1:
            self.show_chat_interface()
        with tab2:
            self.show_image_analysis()
        with tab3:
            self.show_metrics()
        with tab4:
            self.show_chat_history()

    def show_chat_interface(self):
        """Display chat interface"""
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("What's on your mind?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    for chunk in asyncio.run(self.stream_chat(prompt)):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return
            
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Image or Audio",
            type=["jpg", "png", "wav", "mp3"]
        )
        
        if uploaded_file:
            self.handle_file_upload(uploaded_file)

    def show_chat_history(self):
        """Display chat history with analytics"""
        if not st.session_state.messages:
            st.info("No chat history yet")
            return
            
        # Chat statistics
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("User Messages", user_msgs)
        with col2:
            st.metric("Bot Responses", bot_msgs)
            
        # Message timeline
        st.subheader("Conversation Timeline")
        for idx, msg in enumerate(st.session_state.messages):
            with st.expander(f"Message {idx + 1} ({msg['role']})"):
                st.write(msg["content"])
                st.caption(f"Timestamp: {datetime.now() - timedelta(minutes=idx*5)}")

    def handle_file_upload(self, uploaded_file):
        """Handle uploaded files"""
        try:
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image")
                
                with st.spinner("Analyzing image..."):
                    bytes_data = uploaded_file.getvalue()
                    response = asyncio.run(self.chat_request([{
                        "type": "image",
                        "content": base64.b64encode(bytes_data).decode()
                    }]))
                    st.write(response)
                    
            elif uploaded_file.type.startswith('audio'):
                st.audio(uploaded_file)
                
                with st.spinner("Transcribing audio..."):
                    bytes_data = uploaded_file.getvalue()
                    response = asyncio.run(self.chat_request([{
                        "type": "audio",
                        "content": base64.b64encode(bytes_data).decode()
                    }]))
                    st.write(response)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    app = JoromiGPTUI()
    app.run() 