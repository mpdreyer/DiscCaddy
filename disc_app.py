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
import random
import tempfile
import cv2 

# Try import scipy
try:
    from scipy.interpolate import make_interp_spline
except ImportError:
    make_interp_spline = None

# --- 1. KONFIGURATION & SETUP ---
st.set_page_config(page_title="Scuderia Wonka Caddy", page_icon="🏎️", layout="wide")

# SCUDERIA LIVERY CSS
st.markdown("""
    <style>
    .stApp { background-color: #b80000; color: #ffffff; }
    h1, h2, h3, h4 { color: #fff200 !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase; text-shadow: 2px 2px 0px #000000; }
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 3px solid #fff200; }
    .stat-card { background-color: #1a1a1a; border-left: 5px solid #fff200; padding: 15px; border-radius: 6px; box-shadow: 3px 3px 10px rgba(0,0,0,0.5); margin-bottom: 10px; color: white; }
    .stat-value { font-size: 26px; font-weight: bold; color: #ffffff; }
    .stat-label { font-size: 13px; text-transform: uppercase; color: #fff200; letter-spacing: 1px; font-weight: bold;}
    div.stButton > button { background-color: #000000; color: #fff200; border: 2px solid #fff200; border-radius: 4px; font-weight: bold; text-transform: uppercase; }
    div.stButton > button:hover { background-color: #fff200; color: #000000; border-color: #000000; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background-color: #fff200 !important; color: #000000 !important; border-color: #000000 !important; font-weight: bold; }
    input, .stSelectbox div[data-baseweb="select"] span { color: #000000 !important; }
    ul[data-baseweb="menu"] { background-color: #fff200 !important; }
    ul[data-baseweb="menu"] li { color: #000000 !important; font-weight: bold; }
    .streamlit-expanderContent { background-color: #1a1a1a; color: white; border: 1px solid #fff200; border-radius: 0 0 5px 5px; }
    </style>
""", unsafe_allow_html=True)

# Google Sheets Setup
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# --- MASTER COURSE LIST ---
MASTER_COURSES = {
    "Kungsbackaskogen": {"lat": 57.492, "lon": 12.075, "holes": {str(x):{"l": y, "p": 3, "shape": s} for x,y,s in zip(range(1,10), [63,81,48,65,75,55,62,78,52], ["Rak","Vänster","Rak","Höger","Rak","Vänster","Rak","Rak","Rak"])}},
    "Onsala Discgolf": {"lat": 57.416, "lon": 12.029, "holes": {str(x):{"l": 65, "p": 3, "shape": "Rak"} for x in range(1,19)}},
    "Lygnevi Sätila": {"lat": 57.545, "lon": 12.433, "holes": {str(x):{"l": 100, "p": 3, "shape": "Rak"} for x in range(1,19)}},
    "Åbyvallen": {"lat": 57.480, "lon": 12.070, "holes": {str(x):{"l": 70, "p": 3, "shape": "Vänster"} for x in range(1,9)}},
    "Skatås (Gul)": {"lat": 57.704, "lon": 12.036, "holes": {str(x):{"l": 85, "p": 3, "shape": "Skog"} for x in range(1,19)}},
    "Skatås (Vit)": {"lat": 57.704, "lon": 12.036, "holes": {str(x):{"l": 120, "p": 3, "shape": "Lång"} for x in range(1,19)}},
    "Slottsskogen": {"lat": 57.685, "lon": 11.943, "holes": {str(x):{"l": 60, "p": 3, "shape": "Park"} for x in range(1,10)}},
    "Ale Discgolf (Gul)": {"lat": 57.947, "lon": 12.134, "holes": {str(x):{"l": 75, "p": 3, "shape": "Skog"} for x in range(1,19)}},
    "Ale Discgolf (Vit)": {"lat": 57.947, "lon": 12.134, "holes": {str(x):{"l": 110, "p": 3, "shape": "Lång/Skog"} for x in range(1,19)}},
    "Ymer (Borås)": {"lat": 57.747, "lon": 12.909, "holes": {str(x):{"l": 95, "p": 3, "shape": "Kuperat"} for x in range(1,28)}},
    "Sankt Hans (Lund)": {"lat": 55.723, "lon": 13.208, "holes": {str(x):{"l": 90, "p": 3, "shape": "Extrem Backe"} for x in range(1,19)}},
    "Vipeholm (Lund)": {"lat": 55.701, "lon": 13.220, "holes": {str(x):{"l": 70, "p": 3, "shape": "Park"} for x in range(1,19)}},
}

@st.cache_resource
def get_gsheet_client():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        return gspread.authorize(creds)
    except Exception as e: return None

def load_data_from_sheet():
    client = get_gsheet_client()
    if not client: return None, None, None, None
    courses_dict = {}
    users_df = pd.DataFrame()
    try:
        sheet = client.open("DiscCaddy_DB")
        try: ws_users = sheet.worksheet("Users")
        except: 
            ws_users = sheet.add_worksheet("Users", 100, 5)
            ws_users.append_row(["Username", "PIN", "Role", "Active", "Municipality"])
            ws_users.append_row(["Admin", "1234", "Admin", "True", "Kungsbacka"]) 
        users_df = pd.DataFrame(ws_users.get_all_records())
        if users_df.empty: users_df = pd.DataFrame(columns=["Username", "PIN", "Role", "Active"])
        if "Municipality" not in users_df.columns: users_df["Municipality"] = "Unknown"

        try: ws_inv = sheet.worksheet("Inventory")
        except: ws_inv = sheet.add_worksheet("Inventory", 100, 10); ws_inv.append_row(["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"])
        df_inv = pd.DataFrame(ws_inv.get_all_records())
        req_cols = ["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"]
        if df_inv.empty: df_inv = pd.DataFrame(columns=req_cols)
        else:
            for c in req_cols: 
                if c not in df_inv.columns: df_inv[c] = ""
            for col in ["Speed", "Glide", "Turn", "Fade"]: df_inv[col] = pd.to_numeric(df_inv[col], errors='coerce').fillna(0)
            if "Status" not in df_inv.columns: df_inv["Status"] = "Shelf"
            df_inv["Status"] = df_inv["Status"].fillna("Shelf")

        try: ws_hist = sheet.worksheet("History")
        except: ws_hist = sheet.add_worksheet("History", 100, 10); ws_hist.append_row(["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
        df_hist = pd.DataFrame(ws_hist.get_all_records())
        if df_hist.empty: df_hist = pd.DataFrame(columns=["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])

        try: ws_courses = sheet.worksheet("Courses")
        except: 
            ws_courses = sheet.add_worksheet("Courses", 100, 5)
            ws_courses.append_row(["Name", "Lat", "Lon", "Holes_JSON"])
        course_data = ws_courses.get_all_records()
        if not course_data: courses_dict = MASTER_COURSES.copy()
        else:
            for r in course_data:
                try:
                    h_json = json.loads(r["Holes_JSON"]) if isinstance(r["Holes_JSON"], str) else r["Holes_JSON"]
                    courses_dict[r["Name"]] = {"lat": float(r["Lat"]), "lon": float(r["Lon"]), "holes": h_json}
                except: pass
        return df_inv, df_hist, courses_dict, users_df
    except Exception as e: st.error(f"DB Error: {e}"); return pd.DataFrame(), pd.DataFrame(), MASTER_COURSES, pd.DataFrame()

def save_to_sheet(df, worksheet_name):
    client = get_gsheet_client()
    if not client: return
    try:
        sheet = client.open("DiscCaddy_DB")
        ws = sheet.worksheet(worksheet_name)
        if df.empty and worksheet_name == "Inventory": pass 
        ws.clear(); ws.update([df.columns.values.tolist()] + df.values.tolist())
    except Exception as e: st.error(f"Save Error: {e}")

def add_course_to_sheet(name, lat, lon, holes_dict):
    client = get_gsheet_client()
    if not client: return
    try:
        sheet = client.open("DiscCaddy_DB")
        ws = sheet.worksheet("Courses")
        ws.append_row([name, lat, lon, json.dumps(holes_dict)])
    except: pass

def get_lat_lon_from_query(query):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': query, 'format': 'json', 'limit': 1}; headers = {'User-Agent': 'DiscCaddy/1.0'}
        r = requests.get(url, params=params, headers=headers).json()
        if r: return float(r[0]['lat']), float(r[0]['lon'])
    except: pass
    return None, None

def find_courses_via_osm_api(lat, lon, radius=10000):
    try:
        url = "http://overpass-api.de/api/interpreter"
        q = f"[out:json];(node['sport'='disc_golf'](around:{radius},{lat},{lon});way['sport'='disc_golf'](around:{radius},{lat},{lon}););out center;"
        r = requests.get(url, params={'data': q}, timeout=15); d = r.json()
        found = []
        for e in d.get('elements', []):
            name = e.get('tags', {}).get('name', 'Okänd Bana')
            lat = e['lat'] if 'lat' in e else e.get('center',{}).get('lat')
            lon = e['lon'] if 'lon' in e else e.get('center',{}).get('lon')
            found.append({"name": name, "lat": lat, "lon": lon})
        return found
    except: return []

# --- AI & ANALYTICS ---
def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&windspeed_unit=ms"
        res = requests.get(url, timeout=3); data = res.json()
        if "current_weather" in data: return data["current_weather"]
    except: pass
    return None

def ask_ai(messages):
    try:
        client = OpenAI(api_key=st.secrets["openai_key"])
        res = client.chat.completions.create(model="gpt-4o", messages=messages)
        return res.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

def analyze_image(image_bytes):
    try:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        client = OpenAI(api_key=st.secrets["openai_key"])
        prompt = "Identifiera discen. VIKTIGT OM TYP: Exakt 'Putter', 'Midrange', 'Fairway Driver', 'Distance Driver'. Svara JSON: {\"Modell\": \"Namn\", \"Typ\": \"Fairway Driver\", \"Speed\": 7.0, \"Glide\": 5.0, \"Turn\": 0.0, \"Fade\": 2.0}"
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=300)
        return res.choices[0].message.content
    except: return None

def analyze_video_form(video_bytes):
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_bytes)
        vidcap = cv2.VideoCapture(tfile.name)
        frames = []; count = 0; success = True
        while success and count < 30: 
            success, image = vidcap.read()
            if success and count % 5 == 0:
                _, buffer = cv2.imencode('.jpg', image)
                frames.append(base64.b64encode(buffer).decode('utf-8'))
            count += 1
        if len(frames) >= 3:
            key_frames = [frames[0], frames[len(frames)//2], frames[-1]]
            client = OpenAI(api_key=st.secrets["openai_key"])
            content = [{"type": "text", "text": "Analysera denna discgolf-kast teknik. Ge 3 tips. Fokus: Reach back, Power pocket, Follow through."}]
            for f in key_frames: content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
            res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": content}], max_tokens=500)
            return res.choices[0].message.content
        return "Kunde inte läsa videon."
    except Exception as e: return f"Video Error: {e}"

def get_tactical_advice(player, bag_df, dist, weather, situation, obstacles, image_bytes=None):
    bag_str = ", ".join([f"{r['Modell']} ({r['Speed']}/{r['Glide']}/{r['Turn']}/{r['Fade']})" for i, r in bag_df.iterrows()])
    obs_str = ', '.join(obstacles)
    prompt = f"Caddy-råd: Spelare {player}, Bag: {bag_str}. Läge: {situation} ({dist}m till korg). Hinder: {obs_str}. Vind: {weather['wind']}m/s. Prioritera precision. Svara kort: Disc, Linje, Tanke."
    msgs = [{"role": "system", "content": "Elit-caddy."}]
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
    else: content = prompt
    msgs.append({"role": "user", "content": content})
    return ask_ai(msgs)

def get_ai_caddy_advice(player, bag_df, hole_info, weather, situation, obstacles, form_factor=1.0):
    race_bag = bag_df[bag_df["Status"] == "Bag"]
    if race_bag.empty: race_bag = bag_df
    bag_str = ", ".join([f"{r['Modell']} ({r['Speed']}/{r['Glide']}/{r['Turn']}/{r['Fade']})" for i, r in race_bag.iterrows()])
    obs_str = ', '.join(obstacles)
    prompt = f"Elit-Caddy. SPELARE: {player} (Form: {int(form_factor*100)}%). HÅL: {hole_info['l']}m, Par {hole_info['p']}. LÄGE: {situation}, HINDER: {obs_str}. VÄDER: {weather['wind']}m/s. BAG: {bag_str}. UPPGIFT: Välj BÄSTA disc. Rekommendera kasttyp (BH/FH/Roller) och linje. Motivera."
    msgs = [{"role": "system", "content": "Professionell caddy."}]
    msgs.append({"role": "user", "content": prompt})
    return ask_ai(msgs)

def generate_smart_bag(inventory, player, course_name):
    holes = st.session_state.courses[course_name]["holes"]
    lengths = [h["l"] for h in holes.values()]; max_len = max(lengths)
    all_discs = inventory[inventory["Owner"] == player].copy()
    for col in ["Speed", "Glide", "Turn", "Fade"]: all_discs[col] = pd.to_numeric(all_discs[col], errors='coerce').fillna(0)
    all_discs["Stability"] = all_discs["Turn"] + all_discs["Fade"]
    pack_indices = []
    
    def add_best_of(df, count=1, sort_col="Speed", asc=True):
        if df.empty: return
        sorted_df = df.sort_values(sort_col, ascending=asc)
        for i in range(min(count, len(sorted_df))): pack_indices.append(sorted_df.iloc[i].name)

    putters = all_discs[all_discs["Typ"] == "Putter"]; add_best_of(putters, 2, "Speed", True) 
    mids = all_discs[all_discs["Typ"] == "Midrange"]
    if not mids.empty:
        pack_indices.append(mids.sort_values("Stability", ascending=False).iloc[0].name)
        pack_indices.append(mids.sort_values("Stability", ascending=True).iloc[0].name)
    fairways = all_discs[all_discs["Typ"] == "Fairway Driver"]
    if not fairways.empty:
        pack_indices.append(fairways.sort_values("Stability", ascending=False).iloc[0].name)
        pack_indices.append(fairways.sort_values("Stability", ascending=True).iloc[0].name)
    if max_len > 90:
        drivers = all_discs[all_discs["Typ"] == "Distance Driver"]
        if not drivers.empty:
            pack_indices.append(drivers.sort_values("Glide", ascending=False).iloc[0].name)
            pack_indices.append(drivers.sort_values("Fade", ascending=False).iloc[0].name)
    return list(set(pack_indices))

def simulate_flight(speed, glide, turn, fade, power_factor=1.0):
    eff_turn = turn - (power_factor - 1.0) * 2; eff_fade = fade + (1.0 - power_factor) * 2
    x = [0]; y = [0]
    dist_turn = speed * 10 * power_factor * 0.7; side_turn = eff_turn * 3 * power_factor
    x.append(side_turn); y.append(dist_turn)
    dist_total = speed * 10 * power_factor; side_fade = side_turn - (eff_fade * 4)
    x.append(side_fade); y.append(dist_total)
    x = np.array(x); y = np.array(y)
    if make_interp_spline:
        try:
            X_Y_Spline = make_interp_spline(y, x)
            Y_smooth = np.linspace(y.min(), y.max(), 50)
            X_smooth = X_Y_Spline(Y_smooth)
            return X_smooth, Y_smooth
        except: return x, y
    return x, y

# --- STATE INIT ---
if 'data_loaded' not in st.session_state:
    with st.spinner("Startar system..."):
        i, h, c, u = load_data_from_sheet()
        st.session_state.inventory = i
        st.session_state.history = h
        st.session_state.courses = c
        st.session_state.users = u
    st.session_state.data_loaded = True

# Safety Check
if st.session_state.get('logged_in') and not st.session_state.users.empty:
    if st.session_state.current_user not in st.session_state.users["Username"].values:
        st.session_state.logged_in = False
        st.session_state.current_user = None

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'current_user' not in st.session_state: st.session_state.current_user = None
if 'user_role' not in st.session_state: st.session_state.user_role = None
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
if 'putt_session' not in st.session_state: st.session_state.putt_session = []
if 'found_courses' not in st.session_state: st.session_state.found_courses = []
if 'managed_user' not in st.session_state: st.session_state.managed_user = None # För Admin

# --- LOGIN SCREEN ---
if not st.session_state.logged_in:
    st.title("🔐 SCUDERIA PADDOCK")
    c_login = st.container()
    with c_login:
        users = st.session_state.users
        if not users.empty:
            user_list = users["Username"].tolist()
            sel_user = st.selectbox("Välj Förare", user_list)
            pin_in = st.text_input("PIN", type="password")
            if st.button("Lås Upp 🔓", type="primary"):
                user_row = users[users["Username"] == sel_user].iloc[0]
                if str(user_row["PIN"]) == str(pin_in):
                    st.session_state.logged_in = True
                    st.session_state.current_user = sel_user
                    st.session_state.user_role = user_row["Role"]
                    st.session_state.active_players = [sel_user] 
                    st.success("Access Granted"); st.rerun()
                else: st.error("Fel PIN.")
        else: st.error("Inga användare.")
    st.stop()

# --- MAIN APP ---
with st.sidebar:
    st.title("🏎️ SCUDERIA CLOUD")
    st.caption(f"👤 {st.session_state.current_user} | 🟢 v55.0 UAT Ready")
    
    if st.button("Logga Ut"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.divider()
    
    # --- ADMIN: IMPERSONATION TOOL ---
    if st.session_state.user_role == "Admin":
        all_owners = st.session_state.inventory["Owner"].unique().tolist()
        # Default to self if not set
        if not st.session_state.managed_user: st.session_state.managed_user = st.session_state.current_user
        
        # Admin selects who to manage (Inventory/Stats focus)
        managed = st.selectbox("🛠️ Hantera Profil (Admin)", all_owners, index=all_owners.index(st.session_state.managed_user) if st.session_state.managed_user in all_owners else 0)
        st.session_state.managed_user = managed
    else:
        # Regular user always manages self
        st.session_state.managed_user = st.session_state.current_user

    # --- EVERYONE: TEAM SELECTION (FOR RACE/WARMUP) ---
    all_owners = st.session_state.inventory["Owner"].unique().tolist()
    # Remove ghosts
    valid_defaults = [p for p in st.session_state.active_players if p in all_owners]
    
    st.markdown("👥 **Team för dagen (Race/Warmup)**")
    # Multiselect for grouping up
    active_team = st.multiselect("Lägg till kompisar:", all_owners, default=valid_defaults)
    
    if active_team != st.session_state.active_players:
        st.session_state.active_players = active_team
        st.rerun()

    st.divider()
    
    # 1. BANA & VÄDER
    with st.expander("📍 Välj / Lägg till Bana", expanded=True):
        course_names = list(st.session_state.courses.keys())
        sel_course = st.selectbox("Aktiv Bana", course_names, key="course_selector")
        
        if st.session_state.user_role == "Admin" or True: # Allow everyone to find courses
            st.caption("🌍 Hitta ny bana (OSM)")
            search_q = st.text_input("Sök stad/plats (t.ex. Växjö)")
            if st.button("🔍 Sök Banor"):
                if search_q:
                    with st.spinner("Skannar satelliter..."):
                        lat, lon = get_lat_lon_from_query(search_q)
                        if lat: st.session_state.found_courses = find_courses_via_osm_api(lat, lon)
                        else: st.error("Kunde inte hitta platsen.")
            
            if st.session_state.found_courses:
                c_opts = [c["name"] for c in st.session_state.found_courses]
                sel_new_c = st.selectbox("Hittade banor:", c_opts)
                if st.button("➕ Lägg till i Databas"):
                    selected_data = next((item for item in st.session_state.found_courses if item["name"] == sel_new_c), None)
                    if selected_data:
                        std_holes = {str(x): {"l": 100, "p": 3, "shape": "Rak"} for x in range(1, 19)}
                        add_course_to_sheet(selected_data["name"], selected_data["lat"], selected_data["lon"], std_holes)
                        st.success(f"{sel_new_c} tillagd!")
                        st.session_state.found_courses = []
                        st.cache_resource.clear(); st.rerun()

    if 'selected_course' not in st.session_state or sel_course != st.session_state.selected_course:
        st.session_state.selected_course = sel_course
        c_loc = st.session_state.courses[sel_course]
        w = get_live_weather(c_loc["lat"], c_loc["lon"])
        if w: st.session_state.weather_data = {"temp": w["temperature"], "wind": w["windspeed"], "dir": w["winddirection"]}
    
    wd = st.session_state.weather_data
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.metric("Temp", f"{wd['temp']}°C")
        c2.metric("Vind", f"{wd['wind']} m/s")
        hole_wind = st.radio("Vind på tee:", ["Stilla", "Mot", "Med", "Sida"], horizontal=True)

    if st.button("🔄 Synka Databas"): st.cache_resource.clear(); st.rerun()

# --- TABS ---
tabs = ["🔥 WARM-UP", "🏁 RACE", "🤖 AI-CADDY", "🧳 UTRUSTNING", "📈 TELEMETRY", "🎓 ACADEMY"]
if st.session_state.user_role == "Admin": tabs.append("⚙️ HQ")
current_tab = st.tabs(tabs)

# TAB 1: WARM-UP (MULTI-PLAYER)
with current_tab[0]:
    st.header("🔥 Driving Range")
    
    if st.session_state.active_players:
        # SELECT PLAYER FOR THIS THROW
        curr_thrower = st.selectbox("Vem kastar?", st.session_state.active_players)
        
        # Get inv for thrower
        p_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == curr_thrower]
        bag_discs = p_inv[p_inv["Status"]=="Bag"]["Modell"].tolist()
        shelf_discs = p_inv[p_inv["Status"]=="Shelf"]["Modell"].tolist()
        disc_options = ["Välj Disc"] + bag_discs + ["--- HYLLAN ---"] + shelf_discs
        
        c_in, c_list = st.columns([1, 1])
        with c_in:
            with st.container(border=True):
                st.subheader(f"Registrera ({curr_thrower})")
                sel_disc_name = st.selectbox("Disc", disc_options)
                style = st.radio("Stil", ["Backhand (RHBH)", "Forehand (RHFH)"], horizontal=True)
                c_d, c_s = st.columns(2)
                kast_len = c_d.number_input("Längd (m)", 0, 200, 50, step=5)
                kast_sida = c_s.number_input("Sida (m)", -50, 50, 0, step=1, help="-Vä / +Hö")
                if st.button("➕ Spara Kast", type="primary"):
                    if sel_disc_name != "Välj Disc" and "---" not in sel_disc_name and kast_len > 0:
                        d_data = p_inv[p_inv["Modell"]==sel_disc_name].iloc[0]
                        st.session_state.warmup_shots.append({
                            "player": curr_thrower,
                            "disc": sel_disc_name, 
                            "style": style, 
                            "len": kast_len, 
                            "side": kast_sida, 
                            "speed": float(d_data["Speed"])
                        })
                        st.success("Sparat!")
    with c_list:
        if st.session_state.warmup_shots:
            st.dataframe(pd.DataFrame(st.session_state.warmup_shots)[["player", "disc", "len", "side"]], hide_index=True, height=200)
            if st.button("Rensa Session"): st.session_state.warmup_shots = []; st.rerun()
            
    if st.session_state.warmup_shots:
        st.divider()
        # Calculate form for ALL players in session
        for p in st.session_state.active_players:
            p_shots = [s for s in st.session_state.warmup_shots if s['player'] == p]
            if p_shots:
                tot_pot = 0
                for s in p_shots: opt_dist = max(s["speed"] * 10.0, 40.0); tot_pot += (s["len"] / opt_dist)
                avg_form = tot_pot / len(p_shots)
                st.session_state.daily_forms[p] = avg_form
        
        # Display Stats
        st.subheader("📊 Session Stats")
        cols = st.columns(len(st.session_state.active_players))
        for i, p in enumerate(st.session_state.active_players):
            f = st.session_state.daily_forms.get(p, 0)
            if f > 0: cols[i].metric(p, f"Form: {int(f*100)}%")

        # Shared Graph
        fig, ax = plt.subplots(figsize=(6,3))
        shots = st.session_state.warmup_shots
        # Color map
        colors = plt.cm.rainbow(np.linspace(0, 1, len(st.session_state.active_players)))
        p_map = {p: c for p, c in zip(st.session_state.active_players, colors)}
        
        for s in shots:
            ax.scatter(s["side"], s["len"], color=p_map.get(s["player"], 'white'), s=100, alpha=0.8, label=s["player"])
        
        ax.axvline(0, c='white', ls='--')
        ax.set_facecolor('#1a1a1a'); fig.patch.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white'); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
        
        # Fix legend (remove duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), facecolor='#1a1a1a', labelcolor='white')
        
        c2.pyplot(fig)
    else: st.info("Lägg till spelare i menyn till vänster.")

# TAB 2: RACE
with current_tab[1]:
    bana = st.session_state.selected_course
    c_data = st.session_state.courses[bana]
    
    st.subheader("🏁 Race Day")
    # Uses active_players from sidebar
    active_racers = st.session_state.active_players
    
    col_n, col_s = st.columns([1, 2])
    with col_n:
        holes = sorted(list(c_data["holes"].keys()), key=lambda x: int(x) if x.isdigit() else x)
        hole = st.selectbox("Hål", holes)
        inf = c_data["holes"][hole]
        st.metric(f"Hål {hole}", f"{inf['l']}m", f"Par {inf['p']}"); st.caption(inf.get('shape', 'Rak'))
    with col_s:
        if hole not in st.session_state.current_scores: st.session_state.current_scores[hole] = {}
        if hole not in st.session_state.selected_discs: st.session_state.selected_discs[hole] = {}
        
        for p in active_racers:
            if p not in st.session_state.current_scores[hole]: st.session_state.current_scores[hole][p] = inf['p']
            if p not in st.session_state.selected_discs[hole]: st.session_state.selected_discs[hole][p] = None

        for p in active_racers:
            with st.expander(f"🏎️ {p} (Score: {st.session_state.current_scores[hole][p]})", expanded=True):
                # Caddy Advice Button
                if st.button(f"🧠 AI ({p})", key=f"ai_btn_{hole}_{p}"):
                    p_bag = st.session_state.inventory[st.session_state.inventory["Owner"]==p]
                    form = st.session_state.daily_forms.get(p, 1.0)
                    # Simplified context for button click speed
                    with st.spinner("..."):
                        advice = get_ai_caddy_advice(p, p_bag, inf, st.session_state.weather_data, "Tee", [], form)
                        st.session_state.hole_advice[f"{hole}_{p}"] = advice
                
                if f"{hole}_{p}" in st.session_state.hole_advice: st.info(st.session_state.hole_advice[f"{hole}_{p}"])

                c1, c2, c3 = st.columns([1,2,1])
                if c1.button("➖", key=f"m_{hole}_{p}"): st.session_state.current_scores[hole][p] -= 1; st.rerun()
                c2.markdown(f"<h2 style='text-align:center'>{st.session_state.current_scores[hole][p]}</h2>", unsafe_allow_html=True)
                if c3.button("➕", key=f"p_{hole}_{p}"): st.session_state.current_scores[hole][p] += 1; st.rerun()
                
                p_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == p]
                bag_discs = p_inv[p_inv["Status"]=="Bag"]["Modell"].tolist()
                all_discs = p_inv["Modell"].tolist()
                opts = ["Välj Disc"] + (bag_discs if bag_discs else all_discs)
                st.session_state.selected_discs[hole][p] = st.selectbox("Vald Disc", opts, key=f"ds_{hole}_{p}")

    if st.button("🏁 SPARA RUNDA", type="primary"):
        new_rows = []
        d = datetime.now().strftime("%Y-%m-%d")
        for h, scores in st.session_state.current_scores.items():
            for p, s in scores.items():
                disc = st.session_state.selected_discs[h].get(p, "Unknown")
                new_rows.append({"Datum": d, "Bana": bana, "Spelare": p, "Hål": h, "Resultat": s, "Par": c_data["holes"][h]["p"], "Disc_Used": disc})
        new_df = pd.DataFrame(new_rows)
        st.session_state.history = pd.concat([st.session_state.history, new_df], ignore_index=True)
        save_to_sheet(st.session_state.history, "History")
        st.balloons(); st.success("Sparat!"); st.session_state.current_scores = {}

# TAB 3: AI-CADDY
with current_tab[2]:
    st.header("🤖 AI-Chatt")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("Fråga..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        p = st.session_state.current_user
        my_discs = st.session_state.inventory[st.session_state.inventory["Owner"]==p]["Modell"].tolist()
        bag_info = f"Min väska: {', '.join(my_discs)}."
        context = f"Du är en elit-discgolf caddy. {bag_info}. Svara kort."
        messages = [{"role": "system", "content": context}] + st.session_state.chat_history
        with st.chat_message("assistant"):
            with st.spinner("..."):
                reply = ask_ai(messages)
                st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

# TAB 4: UTRUSTNING
with current_tab[3]:
    # USE MANAGED USER (Set in Sidebar)
    target_p = st.session_state.managed_user
    st.header(f"🧳 Logistik: {target_p}")
    
    with st.container(border=True):
        st.markdown("#### 🤖 Strategen")
        c1, c2, c3 = st.columns([2, 1, 1])
        tc = c1.selectbox("Bana:", list(st.session_state.courses.keys()), key="strat_course")
        if c2.button("Generera"): st.session_state.suggested_pack = generate_smart_bag(st.session_state.inventory, target_p, tc); st.rerun()
        if st.session_state.suggested_pack:
            pack_names = st.session_state.inventory.loc[st.session_state.suggested_pack, "Modell"].tolist()
            c1.info(f"Föreslår: {', '.join(pack_names)}")
            if c3.button("Verkställ", type="primary"):
                st.session_state.inventory.loc[st.session_state.inventory["Owner"]==target_p, "Status"] = "Shelf"
                st.session_state.inventory.loc[st.session_state.suggested_pack, "Status"] = "Bag"
                save_to_sheet(st.session_state.inventory, "Inventory"); st.session_state.suggested_pack = []; st.success("Packat!"); st.rerun()
    
    st.markdown("---")
    sort_mode = st.radio("Sortera på:", ["Speed", "Modell", "Typ"], horizontal=True)
    my_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == target_p]
    c_shelf = st.container(border=True); c_bag = st.container(border=True)
    with c_shelf:
        st.subheader("🏠 Hyllan")
        shelf = my_inv[my_inv["Status"] == "Shelf"].sort_values(sort_mode)
        if shelf.empty: st.caption("Tomt.")
        else:
            for idx, row in shelf.iterrows():
                c_txt, c_btn, c_del = st.columns([3, 1, 0.5])
                c_txt.text(f"{row['Modell']} ({int(row['Speed'])})")
                if c_btn.button("➡️", key=f"s2b_{idx}"):
                    st.session_state.inventory.at[idx, "Status"] = "Bag"; save_to_sheet(st.session_state.inventory, "Inventory"); st.rerun()
                if c_del.button("🗑️", key=f"del_s_{idx}"):
                    st.session_state.inventory = st.session_state.inventory.drop(idx); save_to_sheet(st.session_state.inventory, "Inventory"); st.rerun()
    with c_bag:
        st.subheader("🎒 Bagen")
        bag = my_inv[my_inv["Status"] == "Bag"].sort_values(sort_mode)
        if bag.empty: st.caption("Tomt.")
        else:
            for idx, row in bag.iterrows():
                c_btn, c_txt, c_del = st.columns([1, 3, 0.5])
                if c_btn.button("⬅️", key=f"b2s_{idx}"):
                    st.session_state.inventory.at[idx, "Status"] = "Shelf"; save_to_sheet(st.session_state.inventory, "Inventory"); st.rerun()
                c_txt.text(f"{row['Modell']} ({int(row['Speed'])})")
                if c_del.button("🗑️", key=f"del_b_{idx}"):
                    st.session_state.inventory = st.session_state.inventory.drop(idx); save_to_sheet(st.session_state.inventory, "Inventory"); st.rerun()
    st.markdown("---")
    with st.expander(f"➕ Lägg till ny disc för {target_p}"):
        if st.checkbox("Visa Kamera"):
            img_file = st.camera_input("Fota discen")
            if img_file:
                if st.button("🔍 Analysera"):
                    with st.spinner("AI jobbar..."):
                        b_data = img_file.getvalue()
                        json_str = analyze_image(b_data)
                        try:
                            json_str = json_str.replace("```json", "").replace("```", "").strip()
                            st.session_state.ai_disc_data = json.loads(json_str); st.success("Hittad!")
                        except: st.error("Försök igen.")
        with st.form("add_cloud"):
            ai_d = st.session_state.ai_disc_data if st.session_state.ai_disc_data else {}
            c1, c2 = st.columns(2)
            mn = c1.text_input("Modell", value=ai_d.get("Modell", ""))
            v_types = ["Putter", "Midrange", "Fairway Driver", "Distance Driver"]
            r_type = ai_d.get("Typ", "Putter"); f_idx = 0
            for i, vt in enumerate(v_types):
                if vt.lower() in r_type.lower(): f_idx = i; break
            ty = c2.selectbox("Typ", v_types, index=f_idx)
            c3, c4, c5, c6 = st.columns(4)
            sp = c3.number_input("Speed", 0.0, 15.0, float(ai_d.get("Speed", 7.0)))
            gl = c4.number_input("Glide", 0.0, 7.0, float(ai_d.get("Glide", 5.0)))
            tu = c5.number_input("Turn", -5.0, 1.0, float(ai_d.get("Turn", 0.0)))
            fa = c6.number_input("Fade", 0.0, 6.0, float(ai_d.get("Fade", 2.0)))
            if st.form_submit_button("Spara till Hyllan"):
                nw = {"Owner": target_p, "Modell": mn, "Typ": ty, "Speed": sp, "Glide": gl, "Turn": tu, "Fade": fa, "Status": "Shelf"}
                st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame([nw])], ignore_index=True)
                save_to_sheet(st.session_state.inventory, "Inventory")
                st.success(f"{mn} sparad för {target_p}!"); st.session_state.ai_disc_data = None; st.rerun()

# TAB 5: TELEMETRY
with current_tab[4]:
    st.header("📈 SCUDERIA TELEMETRY")
    st1, st2, st3 = st.tabs(["✈️ Aero Lab", "🏎️ Race Performance", "🧩 Sector Analysis"])
    df = st.session_state.history
    
    with st1:
        st.subheader(f"Aerodynamic Wind Tunnel: {st.session_state.managed_user}")
        p = st.session_state.managed_user
        my_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == p]
        c_sim1, c_sim2 = st.columns([1, 2])
        with c_sim1:
            power = st.slider("Power (%)", 50, 150, 100, step=10)
            selected_sim_discs = st.multiselect("Välj Discar", my_inv["Modell"].unique())
        with c_sim2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.axvline(0, color='white', linestyle='--', alpha=0.3)
            if selected_sim_discs:
                for d_name in selected_sim_discs:
                    d_row = my_inv[my_inv["Modell"] == d_name].iloc[0]
                    xs, ys = simulate_flight(d_row["Speed"], d_row["Glide"], d_row["Turn"], d_row["Fade"], power/100.0)
                    stab = d_row["Turn"] + d_row["Fade"]
                    col = '#ff2800' if stab < 0 else '#0066cc' if stab > 2 else '#ffffff'
                    ax.plot(xs, ys, label=d_name, color=col, linewidth=2)
            ax.set_facecolor('#1a1a1a'); fig.patch.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white'); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
            ax.set_xlim(-50, 50); ax.set_ylim(0, 150)
            ax.grid(color='gray', linestyle=':', alpha=0.3)
            if selected_sim_discs: ax.legend(facecolor='#1a1a1a', labelcolor='white')
            st.pyplot(fig)

    with st2:
        if not df.empty:
            c1, c2 = st.columns(2)
            valid_p = [p for p in st.session_state.active_players if p in df["Spelare"].unique()]
            sel_p_stats = c1.multiselect("Förare (Jämför)", df["Spelare"].unique(), default=valid_p)
            sel_c_stats = c2.selectbox("Grand Prix", df["Bana"].unique())
            dff = df[(df["Spelare"].isin(sel_p_stats)) & (df["Bana"]==sel_c_stats)]
            if not dff.empty:
                st.markdown("**Race Pace Trend**")
                trend_data = dff.groupby(["Datum", "Spelare"])["Resultat"].mean().reset_index()
                chart = alt.Chart(trend_data).mark_line(point=True).encode(x='Datum:T', y='Resultat', color='Spelare', tooltip=['Datum', 'Spelare', 'Resultat']).interactive()
                st.altair_chart(chart, use_container_width=True)
            else: st.info("Ingen data.")
        else: st.info("Ingen historik.")

    with st3:
        if not df.empty:
            sel_b_sec = st.selectbox("Analysera Bana", df["Bana"].unique(), key="sec_bana")
            valid_p_sec = [p for p in st.session_state.active_players if p in df["Spelare"].unique()]
            sel_p_sec = st.multiselect("Analysera Förare", df["Spelare"].unique(), key="sec_driver", default=valid_p_sec)
            hdf = df[(df["Bana"]==sel_b_sec) & (df["Spelare"].isin(sel_p_sec))]
            if not hdf.empty:
                hdf['Hål_Int'] = pd.to_numeric(hdf['Hål'], errors='coerce')
                hole_summary = hdf.groupby(["Hål_Int", "Spelare"])["Resultat"].agg(['mean', 'min']).reset_index()
                hole_summary.columns = ['Hål', 'Spelare', 'Snitt', 'Bästa']
                with st.expander("📊 Sektor-Data (Tabell)", expanded=True): st.dataframe(hole_summary, hide_index=True)
                base = alt.Chart(hole_summary).encode(x=alt.X('Hål:O', title="Hål"))
                bar = base.mark_bar(opacity=0.7).encode(y=alt.Y('Snitt', title='Score'), color=alt.Color('Spelare'), xOffset='Spelare', tooltip=['Spelare', 'Hål', 'Snitt', 'Bästa'])
                point = base.mark_point(color='white', size=50, shape='diamond', filled=True).encode(y='Bästa', xOffset='Spelare')
                st.altair_chart((bar + point).interactive(), use_container_width=True)
            else: st.info("Ingen data.")

# TAB 6: ACADEMY
with current_tab[5]:
    st.header("🎓 SCUDERIA ACADEMY")
    st1, st2 = st.tabs(["🎯 Putt-Coach", "📹 Video Scout"])
    with st1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### 🎲 Generera Pass")
            game_mode = st.selectbox("Välj Spel", ["JYLY (Classic)", "Jorden Runt", "Ladder", "Random Pressure"])
            if st.button("Starta Nytt Pass", type="primary"):
                st.session_state.putt_session = []
                if game_mode == "JYLY (Classic)":
                    for d in [5, 6, 7, 8, 9, 10]: st.session_state.putt_session.append({"Dist": d, "Kast": 5, "Träff": 0})
                elif game_mode == "Jorden Runt":
                    for d in [4, 5, 6, 7, 8, 9, 10]: st.session_state.putt_session.append({"Dist": d, "Kast": 3, "Träff": 0})
                elif game_mode == "Ladder":
                    for d in range(3, 11): st.session_state.putt_session.append({"Dist": d, "Kast": 1, "Träff": 0})
                else:
                    for i in range(5): d = random.randint(4, 12); k = random.randint(3, 10); st.session_state.putt_session.append({"Dist": d, "Kast": k, "Träff": 0})
                st.rerun()
        with c2:
            if st.session_state.putt_session:
                st.markdown(f"### 📋 Pågående: {game_mode}")
                total_hits = 0; total_throws = 0
                for i, station in enumerate(st.session_state.putt_session):
                    with st.container(border=True):
                        cols = st.columns([2, 2, 1])
                        cols[0].metric(f"Station {i+1}", f"{station['Dist']}m")
                        res = cols[1].slider(f"Träffar (av {station['Kast']})", 0, station['Kast'], station['Träff'], key=f"putt_{i}")
                        st.session_state.putt_session[i]["Träff"] = res
                        total_hits += res; total_throws += station['Kast']
                st.divider()
                score_col, chart_col = st.columns(2)
                pct = int((total_hits/total_throws)*100) if total_throws > 0 else 0
                score_col.metric("Total Score", f"{total_hits}/{total_throws}", f"{pct}%")
                if st.button("🏁 Avsluta & Spara Pass"):
                    st.balloons(); st.success("Bra jobbat! Vila armen."); st.session_state.putt_session = []; st.rerun()
            else: st.info("Starta ett pass till vänster.")
    with st2:
        st.subheader("📹 Video Form Check")
        st.info("På mobil: Klicka 'Browse files' -> Välj 'Ta Video' eller 'Kamera' för att spela in direkt.")
        vid_file = st.file_uploader("📹 Spela in / Ladda upp Video", type=['mp4', 'mov'])
        if vid_file:
            st.video(vid_file)
            if st.button("🧠 Analysera Teknik"):
                with st.spinner("AI-ögat granskar din sving..."):
                    advice = analyze_video_form(vid_file.read())
                    st.markdown(advice)

# TAB 7: HQ (ADMIN ONLY)
if st.session_state.user_role == "Admin":
    with current_tab[6]:
        st.header("⚙️ SCUDERIA HEADQUARTERS")
        
        st.subheader("👥 Crew Management")
        users = st.session_state.users
        st.dataframe(users, hide_index=True)
        
        c_u1, c_u2 = st.columns(2)
        with c_u1:
            with st.form("new_user_hq"):
                st.markdown("**Skapa Nytt Konto**")
                nu_name = st.text_input("Namn")
                nu_pin = st.text_input("PIN (4 siffror)", max_chars=4)
                nu_role = st.selectbox("Roll", ["Player", "Admin"])
                nu_mun = st.text_input("Hemkommun (t.ex. Kungsbacka)")
                
                if st.form_submit_button("Skapa Användare & Scanna"):
                    client = get_gsheet_client()
                    ws = client.open("DiscCaddy_DB").worksheet("Users")
                    ws.append_row([nu_name, nu_pin, nu_role, "True", nu_mun])
                    
                    start_kit = [{"Owner": nu_name, "Modell": "Start Putter", "Typ": "Putter", "Speed": 3, "Glide": 3, "Turn": 0, "Fade": 0, "Status": "Bag"}]
                    st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame(start_kit)], ignore_index=True)
                    save_to_sheet(st.session_state.inventory, "Inventory")
                    
                    if nu_mun:
                        lat, lon = get_lat_lon_from_query(nu_mun)
                        if lat:
                            new_courses = find_courses_via_osm_api(lat, lon)
                            for nc in new_courses:
                                std_holes = {str(x): {"l": 100, "p": 3, "shape": "Rak"} for x in range(1, 19)}
                                add_course_to_sheet(nc["name"], nc["lat"], nc["lon"], std_holes)
                            st.success(f"Användare {nu_name} skapad! Hittade {len(new_courses)} banor i {nu_mun}.")
                        else: st.warning("Användare skapad, men kunde inte hitta kommunen för bankartläggning.")
                    
                    st.cache_resource.clear(); st.rerun()
        
        with c_u2:
            del_user = st.selectbox("Ta bort användare", users["Username"].tolist())
            if st.button("🗑️ Radera Användare"):
                client = get_gsheet_client()
                ws = client.open("DiscCaddy_DB").worksheet("Users")
                try:
                    cell = ws.find(del_user)
                    ws.delete_rows(cell.row)
                    st.success("Raderad!")
                    st.cache_resource.clear(); st.rerun()
                except: st.error("Kunde inte hitta användaren.")

        st.divider()
        st.subheader("📥 Importera Data")
        up = st.file_uploader("Ladda upp CSV", type=['csv'])
        if up and st.button("Kör Import"):
            try:
                udf = pd.read_csv(up); nd = []
                for i, r in udf.iterrows():
                    if r.get('PlayerName')=='Par': continue
                    mn = r.get('PlayerName')
                    raw_date = str(r.get('StartDate', r.get('Date', datetime.now())))[:10]
                    for hi in range(1, 19):
                        h_score = r.get(f"Hole{hi}")
                        if pd.notna(h_score):
                            nd.append({"Datum": raw_date, "Bana": r.get('CourseName', 'Unknown'), "Spelare": mn, "Hål": str(hi), "Resultat": int(h_score), "Par": 3, "Disc_Used": "Unknown"})
                if nd:
                    new_hist = pd.concat([st.session_state.history, pd.DataFrame(nd)], ignore_index=True)
                    st.session_state.history = new_hist; save_to_sheet(new_hist, "History")
                    st.success(f"Importerade {len(nd)} rader!")
            except Exception as e: st.error(f"Fel: {e}")
