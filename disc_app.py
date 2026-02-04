import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from streamlit_folium import st_folium
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai import OpenAI
import base64
import json
import matplotlib.pyplot as plt
import requests
from geopy.distance import geodesic

# --- 1. KONFIGURATION & SETUP ---
st.set_page_config(page_title="Scuderia Wonka Caddy", page_icon="🏎️", layout="wide")

# Google Sheets Setup
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

@st.cache_resource
def get_gsheet_client():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        return gspread.authorize(creds)
    except Exception as e:
        return None

def load_data_from_sheet():
    client = get_gsheet_client()
    if not client: return pd.DataFrame(), pd.DataFrame()
    try:
        sheet = client.open("DiscCaddy_DB")
        # Inventory
        try: ws_inv = sheet.worksheet("Inventory")
        except: ws_inv = sheet.add_worksheet("Inventory", 100, 10); ws_inv.append_row(["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"])
        inv_data = ws_inv.get_all_records()
        df_inv = pd.DataFrame(inv_data)
        if df_inv.empty: df_inv = pd.DataFrame(columns=["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"])
        else:
            for col in ["Speed", "Glide", "Turn", "Fade"]:
                df_inv[col] = pd.to_numeric(df_inv[col], errors='coerce').fillna(0)
            if "Status" not in df_inv.columns: df_inv["Status"] = "Shelf"
            df_inv["Status"] = df_inv["Status"].fillna("Shelf")

        # History
        try: ws_hist = sheet.worksheet("History")
        except: ws_hist = sheet.add_worksheet("History", 100, 10); ws_hist.append_row(["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
        hist_data = ws_hist.get_all_records()
        df_hist = pd.DataFrame(hist_data)
        if df_hist.empty: df_hist = pd.DataFrame(columns=["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
        
        return df_inv, df_hist
    except Exception as e:
        st.error(f"Databas-fel: {e}"); return pd.DataFrame(), pd.DataFrame()

def save_to_sheet(df, worksheet_name):
    client = get_gsheet_client()
    if not client: return
    try:
        sheet = client.open("DiscCaddy_DB")
        try: ws = sheet.worksheet(worksheet_name)
        except: ws = sheet.add_worksheet(worksheet_name, 100, 10)
        ws.clear()
        ws.update([df.columns.values.tolist()] + df.values.tolist())
    except Exception as e: st.error(f"Sparfel: {e}")

# --- COURSE DATABASE ---
DEFAULT_COURSES = {
    "Kungsbackaskogen": {"lat": 57.492, "lon": 12.075, "holes": {str(x):{"l": y, "p": 3, "shape": "Rak"} for x,y in zip(range(1,10), [63,81,48,65,75,55,62,78,52])}},
    "Lygnevi Sätila": {"lat": 57.545, "lon": 12.433, "holes": {str(x):{"l": 100, "p": 3, "shape": "Rak"} for x in range(1,19)}},
    "Åbyvallen": {"lat": 57.480, "lon": 12.070, "holes": {str(x):{"l": 70, "p": 3, "shape": "Vänster"} for x in range(1,9)}},
    "Skatås (Gul)": {"lat": 57.704, "lon": 12.036, "holes": {str(x):{"l": 90, "p": 3, "shape": "Rak"} for x in range(1,19)}},
    "Skatås (Vit)": {"lat": 57.704, "lon": 12.036, "holes": {str(x):{"l": 120, "p": 4 if x in [5,12] else 3, "shape": "Rak"} for x in range(1,19)}},
    "Ale Discgolf (Vit)": {"lat": 57.947, "lon": 12.134, "holes": {str(x):{"l": 100, "p": 3, "shape": "Skog"} for x in range(1,19)}},
    "Ymer (Borås)": {"lat": 57.747, "lon": 12.909, "holes": {str(x):{"l": 95, "p": 3, "shape": "Kuperat"} for x in range(1,28)}},
}

# --- GPS & WEATHER ---
def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&windspeed_unit=ms"
        res = requests.get(url, timeout=3)
        data = res.json()
        if "current_weather" in data: return data["current_weather"]
    except: pass
    return None

def find_courses_via_osm(lat, lon, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"[out:json];(node['sport'='disc_golf'](around:{radius},{lat},{lon});way['sport'='disc_golf'](around:{radius},{lat},{lon}););out center;"
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=10)
        data = response.json()
        found = []
        for element in data.get('elements', []):
            name = element.get('tags', {}).get('name', 'Okänd Bana (OSM)')
            if 'lat' in element: clat, clon = element['lat'], element['lon']
            else: clat, clon = element.get('center', {}).get('lat'), element.get('center', {}).get('lon')
            found.append({"name": name, "lat": clat, "lon": clon})
        return found
    except: return []

# --- AI CORE ---
def ask_ai(messages):
    try:
        client = OpenAI(api_key=st.secrets["openai_key"])
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        return response.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

def analyze_image(image_bytes):
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        client = OpenAI(api_key=st.secrets["openai_key"])
        prompt = "Identifiera discen. VIKTIGT OM TYP: Måste vara exakt en av: 'Putter', 'Midrange', 'Fairway Driver', 'Distance Driver'. Svara EXAKT JSON: {\"Modell\": \"Tillverkare Modell\", \"Typ\": \"Fairway Driver\", \"Speed\": 7.0, \"Glide\": 5.0, \"Turn\": 0.0, \"Fade\": 2.0}"
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], max_tokens=300)
        return response.choices[0].message.content
    except: return None

def get_tactical_advice(player, bag_df, dist, weather, situation, obstacles, image_bytes=None):
    bag_str = ", ".join([f"{r['Modell']} ({r['Speed']}/{r['Glide']}/{r['Turn']}/{r['Fade']})" for i, r in bag_df.iterrows()])
    
    # Bygg tydligare hinderbeskrivning för AI
    obs_str = ', '.join(obstacles)
    
    prompt = f"""
    Du är en elit-discgolf caddy. Ge ett EXAKT, TAKTISKT råd.
    
    SPELARE: {player}
    VÄSKA: {bag_str}
    LÄGE: {situation} (Korg: {dist}m bort)
    HINDER/BANA: {obs_str}
    VÄDER: {weather['wind']} m/s, {weather['temp']} grader.
    
    OBS: Om "Port/Gap" nämns, prioritera precision över kraft.
    OBS: Om "Smal Korridor" nämns, prioritera raka discar (low fade/turn).
    
    Uppgift:
    1. Rekommendera BÄSTA discen.
    2. Rekommendera KASTTYP & LINJE.
    3. Ge tips om höjd och kraft.
    
    Format: "**Disc:** [Val] \n**Plan:** [BH/FH] [Linje] \n**Nyckel:** [Tips]"
    """
    
    messages = [{"role": "system", "content": "Du är en professionell discgolf-caddy."}]
    
    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        user_content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]
    else:
        user_content = prompt

    messages.append({"role": "user", "content": user_content})
    return ask_ai(messages)

# --- STATE INIT ---
if 'data_loaded' not in st.session_state:
    with st.spinner("Startar system..."):
        i, h = load_data_from_sheet()
        st.session_state.inventory = i
        st.session_state.history = h
        st.session_state.courses = DEFAULT_COURSES.copy()
    st.session_state.data_loaded = True

if 'active_players' not in st.session_state: st.session_state.active_players = []
if 'current_scores' not in st.session_state: st.session_state.current_scores = {}
if 'selected_discs' not in st.session_state: st.session_state.selected_discs = {}
if 'daily_forms' not in st.session_state: st.session_state.daily_forms = {}
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'ai_disc_data' not in st.session_state: st.session_state.ai_disc_data = None
if 'camera_active' not in st.session_state: st.session_state.camera_active = False
if 'suggested_pack' not in st.session_state: st.session_state.suggested_pack = []
if 'warmup_shots' not in st.session_state: st.session_state.warmup_shots = []
if 'weather_data' not in st.session_state: st.session_state.weather_data = {"temp": 15, "wind": 2, "dir": 0}
if 'user_location' not in st.session_state: st.session_state.user_location = {"lat": 57.492, "lon": 12.075, "name": "Kungsbacka"}
if 'hole_advice' not in st.session_state: st.session_state.hole_advice = {}

# --- UI LOGIC ---
with st.sidebar:
    st.title("🏎️ SCUDERIA CLOUD")
    st.caption("🟢 v39.0 Precision Pilot")
    
    with st.expander("📍 Plats & Väder", expanded=True):
        loc_presets = {"Kungsbacka": (57.492, 12.075), "Göteborg": (57.704, 12.036), "Borås": (57.721, 12.940), "Ale": (57.947, 12.134)}
        sel_loc = st.selectbox("Område", list(loc_presets.keys()))
        st.session_state.user_location = {"lat":
