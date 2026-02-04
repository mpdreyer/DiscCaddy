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

@st.cache_resource
def get_gsheet_client():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        return gspread.authorize(creds)
    except Exception as e: return None

def load_data_from_sheet():
    client = get_gsheet_client()
    if not client: return pd.DataFrame(), pd.DataFrame()
    try:
        sheet = client.open("DiscCaddy_DB")
        try: ws_inv = sheet.worksheet("Inventory")
        except: ws_inv = sheet.add_worksheet("Inventory", 100, 10); ws_inv.append_row(["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"])
        inv_data = ws_inv.get_all_records()
        df_inv = pd.DataFrame(inv_data)
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
        hist_data = ws_hist.get_all_records()
        df_hist = pd.DataFrame(hist_data)
        if df_hist.empty: df_hist = pd.DataFrame(columns=["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
        return df_inv, df_hist
    except Exception as e: st.error(f"DB Error: {e}"); return pd.DataFrame(), pd.DataFrame()

def save_to_sheet(df, worksheet_name):
    client = get_gsheet_client()
    if not client: return
    try:
        sheet = client.open("DiscCaddy_DB")
        try: ws = sheet.worksheet(worksheet_name)
        except: ws = sheet.add_worksheet(worksheet_name, 100, 10)
        # SÄKERHET: Rensa inte om dataframe är helt tom (förutom headers) om det verkar fel
        if df.empty and worksheet_name == "Inventory":
             # Om vi försöker spara en tom inventory, varna hellre än radera allt
             pass 
        ws.clear(); ws.update([df.columns.values.tolist()] + df.values.tolist())
    except Exception as e: st.error(f"Save Error: {e}")

# --- COURSE DATABASE ---
DEFAULT_COURSES = {
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

# --- LOGIC ---
def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&windspeed_unit=ms"
        res = requests.get(url, timeout=3); data = res.json()
        if "current_weather" in data: return data["current_weather"]
    except: pass
    return None

def find_courses_via_osm(lat, lon, radius=5000):
    try:
        url = "http://overpass-api.de/api/interpreter"
        q = f"[out:json];(node['sport'='disc_golf'](around:{radius},{lat},{lon});way['sport'='disc_golf'](around:{radius},{lat},{lon}););out center;"
        r = requests.get(url, params={'data': q}, timeout=10); d = r.json()
        found = []
        for e in d.get('elements', []):
            name = e.get('tags', {}).get('name', 'Okänd (OSM)')
            lat = e['lat'] if 'lat' in e else e.get('center',{}).get('lat')
            lon = e['lon'] if 'lon' in e else e.get('center',{}).get('lon')
            found.append({"name": name, "lat": lat, "lon": lon})
        return found
    except: return []

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
        frames = []
        count = 0
        success = True
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

def suggest_disc(bag, player, dist, shape, form=1.0, wind_str=0, wind_type="Stilla"):
    pb = bag[(bag["Owner"]==player) & (bag["Status"]=="Bag")]
    if pb.empty: return None, "Tom väska"
    eff_dist = dist / max(form, 0.5); target_speed = eff_dist / 10.0
    pb = pb.copy()
    for c in ["Speed", "Turn", "Fade"]: pb[c] = pd.to_numeric(pb[c], errors='coerce').fillna(0)
    
    if "Motvind" in wind_type: pb["Eff_Turn"] = pb["Turn"] - (wind_str * 0.4); advice_suffix = " (Motvind)"
    elif "Medvind" in wind_type: pb["Eff_Turn"] = pb["Turn"] + (wind_str * 0.3); advice_suffix = " (Medvind)"
    else: pb["Eff_Turn"] = pb["Turn"]; advice_suffix = ""

    pb["Speed_Diff"] = abs(pb["Speed"] - target_speed)
    candidates = pb.copy()
    if eff_dist < 45: candidates = candidates[candidates["Typ"]=="Putter"]
    elif eff_dist < 85: candidates = candidates[candidates["Typ"].isin(["Putter","Midrange"])]
    elif eff_dist < 110: candidates = candidates[candidates["Typ"].isin(["Midrange", "Fairway Driver"])]
    if candidates.empty: candidates = pb
    
    if form < 0.9: candidates["Score"] = candidates["Speed_Diff"] + (candidates["Eff_Turn"] * 0.5)
    else: candidates["Score"] = candidates["Speed_Diff"]
    
    if shape == "Höger": best = candidates.sort_values(by=["Score", "Fade"], ascending=[True, False]).iloc[0]; reason="Forehand" + advice_suffix
    elif shape == "Vänster": best = candidates.sort_values(by=["Score", "Fade"], ascending=[True, False]).iloc[0]; reason="Hyzer" + advice_suffix
    else: best = candidates.sort_values(by=["Score", "Eff_Turn"], ascending=[True, True]).iloc[0]; reason="Rakt" + advice_suffix
    return best, reason

def generate_smart_bag(inventory, player, course_name):
    holes = st.session_state.courses[course_name]["holes"]
    lengths = [h["l"] for h in holes.values()]
    max_len = max(lengths)
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
if 'putt_session' not in st.session_state: st.session_state.putt_session = []

# --- UI LOGIC ---
with st.sidebar:
    st.title("🏎️ SCUDERIA CLOUD")
    st.caption("🟢 v49.0 Safety Car Update")
    
    with st.expander("📍 Plats & Väder", expanded=True):
        loc_presets = {"Kungsbacka": (57.492, 12.075), "Göteborg": (57.704, 12.036), "Borås": (57.721, 12.940), "Ale": (57.947, 12.134), "Lund": (55.704, 13.191)}
        sel_loc = st.selectbox("Område", list(loc_presets.keys()))
        st.session_state.user_location = {"lat": loc_presets[sel_loc][0], "lon": loc_presets[sel_loc][1], "name": sel_loc}
        
        course_list = []
        for name, data in st.session_state.courses.items():
            dist = geodesic(loc_presets[sel_loc], (data["lat"], data["lon"])).km
            course_list.append((name, dist))
        course_list.sort(key=lambda x: x[1])
        sorted_names = [x[0] for x in course_list]
        st.caption(f"Närmast: {course_list[0][0]}")
        
        if st.button("🔍 Scanna (OSM)"):
            with st.spinner("Söker..."):
                nc = find_courses_via_osm(loc_presets[sel_loc][0], loc_presets[sel_loc][1])
                for n in nc:
                    if n["name"] not in st.session_state.courses:
                        st.session_state.courses[n["name"]] = {"lat": n["lat"], "lon": n["lon"], "holes": {str(h): {"l": 100, "p": 3, "shape": "Rak"} for h in range(1, 19)}}
                st.success("Klar!"); st.rerun()

    sel_course = st.selectbox("Välj Bana", sorted_names, key="course_selector")
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

    st.divider()
    
    # 2. SPELARE (Updated for Robustness)
    all_owners = st.session_state.inventory["Owner"].unique().tolist() if not st.session_state.inventory.empty else []
    
    new_p = st.text_input("Ny spelare / Återställ:", placeholder="Namn")
    if st.button("Lägg till") and new_p:
        # Create start kit row
        start_kit = [{"Owner": new_p, "Modell": "Start Putter", "Typ": "Putter", "Speed": 3, "Glide": 3, "Turn": 0, "Fade": 0, "Status": "Bag"}]
        # Append safe
        st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame(start_kit)], ignore_index=True)
        # Force save
        save_to_sheet(st.session_state.inventory, "Inventory")
        # Clear cache to force reload next time
        st.cache_resource.clear()
        st.success("Spelare skapad/återställd!"); st.rerun()

    active = st.multiselect("Spelare", all_owners, default=st.session_state.active_players)
    if active != st.session_state.active_players:
        st.session_state.active_players = active
        st.rerun()
        
    if st.button("🔄 Synka Databas"): st.cache_resource.clear(); st.rerun()

t1, t2, t3, t4, t5, t6, t7 = st.tabs(["🔥 WARM-UP", "🏁 RACE", "🤖 AI-CADDY", "🧳 UTRUSTNING", "📈 TELEMETRY", "⚙️ ADMIN", "🎓 ACADEMY"])

# TAB 1: WARM-UP
with t1:
    st.header("🔥 Driving Range")
    if st.session_state.active_players:
        curr_p = st.selectbox("Kalibrera:", st.session_state.active_players)
        p_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == curr_p]
        disc_options = ["Välj Disc"] + p_inv["Modell"].unique().tolist()
        c_in, c_list = st.columns([1, 1])
        with c_in:
            with st.container(border=True):
                st.subheader("Registrera")
                sel_disc_name = st.selectbox("Disc", disc_options)
                style = st.radio("Stil", ["Backhand (RHBH)", "Forehand (RHFH)"], horizontal=True)
                c_d, c_s = st.columns(2)
                kast_len = c_d.number_input("Längd (m)", 0, 200, 50, step=5)
                kast_sida = c_s.number_input("Sida (m)", -50, 50, 0, step=1, help="-Vä / +Hö")
                if st.button("➕ Spara Kast", type="primary"):
                    if sel_disc_name != "Välj Disc" and kast_len > 0:
                        d_data = p_inv[p_inv["Modell"]==sel_disc_name].iloc[0]
                        st.session_state.warmup_shots.append({"disc": sel_disc_name, "style": style, "len": kast_len, "side": kast_sida, "speed": float(d_data["Speed"]), "turn": float(d_data["Turn"]), "fade": float(d_data["Fade"])})
                        st.success("Sparat!")
        with c_list:
            if st.session_state.warmup_shots:
                st.dataframe(pd.DataFrame(st.session_state.warmup_shots)[["disc","style","len","side"]], hide_index=True, height=200)
                if st.button("Rensa"): st.session_state.warmup_shots = []; st.rerun()
        if st.session_state.warmup_shots:
            st.divider()
            shots = st.session_state.warmup_shots
            tot_pot = 0; tot_tech = 0
            for s in shots:
                opt_dist = max(s["speed"] * 10.0, 40.0); tot_pot += (s["len"] / opt_dist)
                nat_side = 0 
                tot_tech += (s["side"] - nat_side)
            avg_form = tot_pot / len(shots)
            st.session_state.daily_forms[curr_p] = avg_form
            c1, c2 = st.columns(2)
            c1.metric("Potential", f"{int(avg_form*100)}%")
            fig, ax = plt.subplots(figsize=(4,3))
            x=[s["side"] for s in shots]; y=[s["len"] for s in shots]
            c=['#fff200' if "Backhand" in s["style"] else '#ffffff' for s in shots]
            ax.scatter(x,y,c=c,s=80,alpha=0.7); ax.axvline(0,c='white',ls='--')
            ax.set_xlim(-40,40); ax.set_ylim(0, max(y)*1.2)
            ax.set_facecolor('#1a1a1a'); fig.patch.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white'); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white'); ax.title.set_color('white')
            c2.pyplot(fig)
    else: st.info("Välj spelare.")

# TAB 2: RACE
with t2:
    bana = st.session_state.selected_course
    c_data = st.session_state.courses[bana]
    col_n, col_s = st.columns([1, 2])
    with col_n:
        holes = sorted(list(c_data["holes"].keys()), key=lambda x: int(x) if x.isdigit() else x)
        hole = st.selectbox("Hål", holes)
        inf = c_data["holes"][hole]
        st.metric(f"Hål {hole}", f"{inf['l']}m", f"Par {inf['p']}"); st.caption(inf.get('shape', 'Rak'))
    with col_s:
        if hole not in st.session_state.current_scores: st.session_state.current_scores[hole] = {}
        if hole not in st.session_state.selected_discs: st.session_state.selected_discs[hole] = {}
        for p in st.session_state.active_players:
            if p not in st.session_state.current_scores[hole]: st.session_state.current_scores[hole][p] = inf['p']
            if p not in st.session_state.selected_discs[hole]: st.session_state.selected_discs[hole][p] = None
        for p in st.session_state.active_players:
            with st.expander(f"🏎️ {p} (Score: {st.session_state.current_scores[hole][p]})", expanded=True):
                with st.container(border=True):
                    st.markdown(f"**Taktik {p}**")
                    c_sit, c_obs = st.columns([1, 1])
                    situation = c_sit.radio("Läge", ["Tee", "Fairway", "Ruff", "Putt"], key=f"sit_{hole}_{p}", horizontal=True)
                    dist_left = c_sit.slider("Avstånd (m)", 0, 200, int(inf['l']) if situation=="Tee" else 50, key=f"d_{hole}_{p}")
                    base_obs = ["Träd Vänster", "Träd Höger", "Smal Korridor", "Port/Gap", "Lågt Tak", "Vatten", "Uppför", "Nedför"]
                    obstacles = c_obs.multiselect("Hinder", base_obs, key=f"obs_{hole}_{p}")
                    gap_info = ""
                    if "Port/Gap" in obstacles:
                        cg1, cg2 = st.columns(2)
                        g_dist = cg1.number_input("Avstånd Port", 0, 150, 30, key=f"gd_{hole}_{p}")
                        g_width = cg2.number_input("Bredd Port", 1, 20, 3, key=f"gw_{hole}_{p}")
                        gap_info = f" | Måste träffa {g_width}m lucka {g_dist}m bort."
                    use_cam = st.checkbox("📸 Fota", key=f"cam_tog_{hole}_{p}")
                    img_data = None
                    if use_cam:
                        img_file = st.camera_input("Fota", key=f"ci_{hole}_{p}")
                        if img_file: img_data = img_file.getvalue()
                    if st.button(f"🧠 Team Radio ({p})", key=f"ai_btn_{hole}_{p}"):
                        p_bag = st.session_state.inventory[(st.session_state.inventory["Owner"]==p) & (st.session_state.inventory["Status"]=="Bag")]
                        final_obs = obstacles.copy()
                        if gap_info: final_obs.append(gap_info)
                        with st.spinner("Beräknar..."):
                            advice = get_tactical_advice(p, p_bag, dist_left, st.session_state.weather_data, situation, final_obs, img_data)
                            st.session_state.hole_advice[f"{hole}_{p}"] = advice
                    if f"{hole}_{p}" in st.session_state.hole_advice: st.info(st.session_state.hole_advice[f"{hole}_{p}"])
                c1, c2, c3 = st.columns([1,2,1])
                if c1.button("➖", key=f"m_{hole}_{p}"): st.session_state.current_scores[hole][p] -= 1; st.rerun()
                c2.markdown(f"<h2 style='text-align:center'>{st.session_state.current_scores[hole][p]}</h2>", unsafe_allow_html=True)
                if c3.button("➕", key=f"p_{hole}_{p}"): st.session_state.current_scores[hole][p] += 1; st.rerun()
                p_bag = st.session_state.inventory[(st.session_state.inventory["Owner"]==p) & (st.session_state.inventory["Status"]=="Bag")]
                opts = ["Välj Disc"] + p_bag["Modell"].tolist()
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
with t3:
    st.header("🤖 AI-Chatt")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("Fråga..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        bag_info = ""
        if st.session_state.active_players:
            p = st.session_state.active_players[0]
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
with t4:
    st.header("🧳 Logistik-Center")
    owner = st.selectbox("Hantera", st.session_state.active_players, index=0) if st.session_state.active_players else None
    with st.container(border=True):
        st.markdown("#### 🤖 Strategen")
        c1, c2, c3 = st.columns([2, 1, 1])
        tc = c1.selectbox("Bana:", list(st.session_state.courses.keys()), key="strat_course")
        if c2.button("Generera"): st.session_state.suggested_pack = generate_smart_bag(st.session_state.inventory, owner, tc); st.rerun()
        if st.session_state.suggested_pack:
            pack_names = st.session_state.inventory.loc[st.session_state.suggested_pack, "Modell"].tolist()
            c1.info(f"Föreslår: {', '.join(pack_names)}")
            if c3.button("Verkställ", type="primary"):
                st.session_state.inventory.loc[st.session_state.inventory["Owner"]==owner, "Status"] = "Shelf"
                st.session_state.inventory.loc[st.session_state.suggested_pack, "Status"] = "Bag"
                save_to_sheet(st.session_state.inventory, "Inventory"); st.session_state.suggested_pack = []; st.success("Packat!"); st.rerun()
    if owner:
        st.markdown("---")
        sort_mode = st.radio("Sortera på:", ["Speed", "Modell", "Typ"], horizontal=True)
        my_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == owner]
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
    with st.expander("➕ Lägg till ny disc"):
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
                nw = {"Owner": owner, "Modell": mn, "Typ": ty, "Speed": sp, "Glide": gl, "Turn": tu, "Fade": fa, "Status": "Shelf"}
                st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame([nw])], ignore_index=True)
                save_to_sheet(st.session_state.inventory, "Inventory")
                st.success(f"{mn} sparad!"); st.session_state.ai_disc_data = None; st.rerun()

# TAB 5: TELEMETRY
with t5:
    st.header("📈 SCUDERIA TELEMETRY")
    st1, st2, st3 = st.tabs(["✈️ Aero Lab", "🏎️ Race Performance", "🧩 Sector Analysis"])
    df = st.session_state.history
    
    with st1:
        st.subheader("Aerodynamic Wind Tunnel")
        if st.session_state.active_players:
            p = st.session_state.active_players[0]
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
        else: st.info("Välj spelare.")

    with st2:
        if not df.empty:
            c1, c2 = st.columns(2)
            sel_p_stats = c1.multiselect("Förare (Jämför)", df["Spelare"].unique(), default=df["Spelare"].unique())
            sel_c_stats = c2.selectbox("Grand Prix", df["Bana"].unique())
            dff = df[(df["Spelare"].isin(sel_p_stats)) & (df["Bana"]==sel_c_stats)]
            if not dff.empty:
                st.markdown("**Race Pace Trend**")
                trend_data = dff.groupby(["Datum", "Spelare"])["Resultat"].mean().reset_index()
                chart = alt.Chart(trend_data).mark_line(point=True).encode(
                    x='Datum:T', y='Resultat', color='Spelare', tooltip=['Datum', 'Spelare', 'Resultat']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else: st.info("Ingen data.")
        else: st.info("Ingen historik.")

    with st3:
        if not df.empty:
            sel_b_sec = st.selectbox("Analysera Bana", df["Bana"].unique(), key="sec_bana")
            sel_p_sec = st.multiselect("Analysera Förare", df["Spelare"].unique(), key="sec_driver", default=df["Spelare"].unique())
            
            hdf = df[(df["Bana"]==sel_b_sec) & (df["Spelare"].isin(sel_p_sec))]
            
            if not hdf.empty:
                hdf['Hål_Int'] = pd.to_numeric(hdf['Hål'], errors='coerce')
                
                # Sektor Analys 2.0 (Dual Stats)
                hole_summary = hdf.groupby(["Hål_Int", "Spelare"])["Resultat"].agg(['mean', 'min']).reset_index()
                hole_summary.columns = ['Hål', 'Spelare', 'Snitt', 'Bästa']
                
                # Tabellvy
                with st.expander("📊 Sektor-Data (Tabell)", expanded=True):
                    st.dataframe(hole_summary, hide_index=True)

                # Graf: Kombinerad Stapel (Snitt) + Prick (Bästa)
                base = alt.Chart(hole_summary).encode(x=alt.X('Hål:O', title="Hål"))
                
                bar = base.mark_bar(opacity=0.7).encode(
                    y=alt.Y('Snitt', title='Score'),
                    color=alt.Color('Spelare'),
                    xOffset='Spelare',
                    tooltip=['Spelare', 'Hål', 'Snitt', 'Bästa']
                )
                
                point = base.mark_point(color='white', size=50, shape='diamond', filled=True).encode(
                    y='Bästa',
                    xOffset='Spelare'
                )
                
                st.altair_chart((bar + point).interactive(), use_container_width=True)
            else: st.info("Ingen data.")

# TAB 6: ADMIN
with t6:
    st.subheader("⚙️ Admin")
    up = st.file_uploader("Ladda upp CSV", type=['csv'])
    if up and st.button("Kör Import"):
        try:
            udf = pd.read_csv(up); nd = []
            for i, r in udf.iterrows():
                if r.get('PlayerName')=='Par': continue
                mn = "Mattias" if "Mattias" in r.get('PlayerName','') else "Jenny" if "Jenny" in r.get('PlayerName','') else r.get('PlayerName')
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

# TAB 7: ACADEMY
with t7:
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
