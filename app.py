import streamlit as st
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
import os

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")


# Define the Weather Tool
class WeatherTool(BaseTool):
    name: str = "Weather Tool"
    description: str = "Get the current weather and temperature for a given location."

    def _run(self, location: str) -> str:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
            response = requests.get(url).json()
            if response["cod"] != 200:
                return f"Error: {response['message']}"
            weather = response["weather"][0]["main"]
            temperature = response["main"]["temp"]
            return f"{weather}, {temperature}°C"
        except Exception as e:
            return f"Failed to retrieve weather: {str(e)}"


class ClothingSuggestionTool(BaseTool):
    name: str = "Clothing Suggestion Tool"
    description: str = "Suggest appropriate clothing based on weather conditions and temperature."

    def _run(self, weather_and_temp: str) -> str:
        try:
            parts = [p.strip() for p in weather_and_temp.split(",")]
            if len(parts) != 2:
                return "Invalid input format. Expected 'weather, temperature'."

            weather = None
            temperature = None
            for part in parts:
                if "°c" in part.lower():
                    try:
                        temperature = float(part.lower().replace("°c", "").strip())
                    except ValueError:
                        return f"Invalid temperature value: {part}"
                else:
                    weather = part.strip()

            if weather is None or temperature is None:
                return "Invalid input format. Ensure both weather and temperature are provided."

            weather = weather.lower()
            suggestion = ""

            if "rain" in weather:
                suggestion = "🌧️ Wear a waterproof jacket and carry an umbrella."
            elif "clear" in weather:
                suggestion = "☀️ A light t-shirt and sunglasses would be great for sunny weather."
            elif "clouds" in weather or "cloudy" in weather:
                suggestion = "☁️ Wear comfortable layers as it might feel cool."
            elif "snow" in weather:
                suggestion = "❄️ Bundle up with a heavy coat, gloves, and a scarf."
            elif "haze" in weather:
                suggestion = "🌫️ Wear light and breathable clothing, and consider using a mask if the air quality is poor."
            else:
                suggestion = "👕 Dress comfortably for unique or uncommon weather conditions."

            if temperature < 10:
                suggestion += " 🧥 Also, wear warm clothing as it's quite cold."
            elif 10 <= temperature <= 20:
                suggestion += " 🧥 A light jacket or sweater is recommended."
            elif temperature > 30:
                suggestion += " 🌴 Stay cool with breathable fabrics like cotton or linen."

            return suggestion
        except Exception as e:
            return f"Failed to provide clothing suggestion: {str(e)}"


# Instantiate tools
weather_tool = WeatherTool()
clothing_tool = ClothingSuggestionTool()

# Initialize LangChain Agent
tools = [weather_tool, clothing_tool]
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Streamlit UI
st.title("🌦️ Weather and Clothing Suggestion Tool")
st.markdown(
    """
    <div style="text-align: center; font-size: 20px; color: #4CAF50;">
        🌍 Enter your travel destination to get the current weather and personalized clothing suggestions. 🌈
    </div>
    """,
    unsafe_allow_html=True,
)

# Input field for location
st.markdown(
    "<div style='color: #ff0000; font-size: 18px;'>🌐 Where are you traveling?</div>",
    unsafe_allow_html=True,
)
location = st.text_input("", placeholder="Enter a city name...")

# Generate button
if st.button("✨ Get Weather and Clothing Suggestion ✨"):
    if location:
        with st.spinner("⏳ Fetching weather and generating suggestions..."):
            try:
                # Get the weather and clothing suggestion
                response = agent.run(f"What's the weather in {location}? Suggest what to wear.")
                st.success("✅ Here's what we found!")
                st.markdown(
                    f"""
                    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; background-color: #F5F5F5; font-size: 16px;">
                        {response}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please enter a location!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        <b>Powered by:</b> LangChain, ChatGroq, and OpenWeather API<br>
        <i>💡 Making your travel decisions smarter!</i>
    </div>
    """,
    unsafe_allow_html=True,
)
