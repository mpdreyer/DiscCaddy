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
    # Bygg kontext för AI
    bag_str = ", ".join([f"{r['Modell']} ({r['Speed']}/{r['Glide']}/{r['Turn']}/{r['Fade']})" for i, r in bag_df.iterrows()])
    
    prompt = f"""
    Du är en elit-discgolf caddy. Ge ett EXAKT, TAKTISKT råd för nästa kast.
    
    SPELARE: {player}
    VÄSKA: {bag_str}
    LÄGE: {situation} (Avstånd till korg: {dist}m)
    HINDER: {', '.join(obstacles)}
    VÄDER: {weather['wind']} m/s, {weather['temp']} grader.
    
    Uppgift:
    1. Rekommendera BÄSTA discen ur väskan.
    2. Rekommendera KASTTYP (Backhand/Forehand).
    3. Rekommendera LINJE (Hyzer, Flat, Anhyzer/Flex, Roller).
    4. Ge ett kort tips om höjd och kraft.
    
    Svara kort och peppande. Format: "**Disc:** [Val] \n**Kast:** [BH/FH] [Linje] \n**Tanke:** [Förklaring]"
    """
    
    messages = [{"role": "system", "content": "Du är en professionell discgolf-caddy."}]
    
    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        user_content = [
            {"type": "text", "text": prompt + " \n(Se bifogad bild på hålet/läget)"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
        ]
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
if 'hole_advice' not in st.session_state: st.session_state.hole_advice = {} # Spara råd per hål/spelare

# --- UI LOGIC ---
with st.sidebar:
    st.title("🏎️ SCUDERIA CLOUD")
    st.caption("🟢 v38.0 Tactical Caddy")
    
    # 1. GPS & VÄDER
    with st.expander("📍 Plats & Väder", expanded=True):
        loc_presets = {"Kungsbacka": (57.492, 12.075), "Göteborg": (57.704, 12.036), "Borås": (57.721, 12.940), "Ale": (57.947, 12.134)}
        sel_loc = st.selectbox("Område", list(loc_presets.keys()))
        st.session_state.user_location = {"lat": loc_presets[sel_loc][0], "lon": loc_presets[sel_loc][1], "name": sel_loc}
        
        # Sortera banor
        course_list = []
        for name, data in st.session_state.courses.items():
            dist = geodesic(loc_presets[sel_loc], (data["lat"], data["lon"])).km
            course_list.append((name, dist))
        course_list.sort(key=lambda x: x[1])
        sorted_names = [x[0] for x in course_list]
        st.caption(f"Närmast: {course_list[0][0]}")
        
        if st.button("🔍 Scanna nya banor (OSM)"):
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
        hole_wind = st.radio("Vindriktning (Tee)", ["Stilla", "Mot", "Med", "Sida"], horizontal=True)

    st.divider()
    
    # SPELARE & SYNK
    all_owners = st.session_state.inventory["Owner"].unique().tolist()
    active = st.multiselect("Spelare", all_owners, default=st.session_state.active_players)
    if active != st.session_state.active_players:
        st.session_state.active_players = active
        st.rerun()
    if st.button("🔄 Synka Databas"): st.cache_resource.clear(); st.rerun()

t1, t2, t3, t4, t5, t6 = st.tabs(["🔥 WARM-UP", "🏁 RACE", "🤖 AI-CADDY", "🧳 UTRUSTNING", "📊 STATS", "⚙️ ADMIN"])

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
                req = s["speed"]*10.0; p_rat = s["len"]/req if req>0 else 1.0
                f_dir = -1 if "Backhand" in s["style"] else 1
                t_dir = -f_dir
                nat_side = (s["fade"]*3*f_dir) if p_rat<0.8 else (s["turn"]*2*t_dir + s["fade"]*2*f_dir)
                tot_tech += (s["side"] - nat_side)
            
            avg_form = tot_pot / len(shots)
            st.session_state.daily_forms[curr_p] = avg_form
            
            c1, c2 = st.columns(2)
            c1.metric("Potential", f"{int(avg_form*100)}%")
            if abs(tot_tech/len(shots)) < 7: c1.success("✅ Bra linjer!")
            else: c1.warning("⚠️ Teknikavvikelse")
            
            fig, ax = plt.subplots(figsize=(4,3))
            x=[s["side"] for s in shots]; y=[s["len"] for s in shots]
            c=['#cc0000' if "Backhand" in s["style"] else '#0066cc' for s in shots]
            ax.scatter(x,y,c=c,s=80,alpha=0.7); ax.axvline(0,c='gray',ls='--')
            ax.set_xlim(-40,40); ax.set_ylim(0, max(y)*1.2); c2.pyplot(fig)
    else: st.info("Välj spelare.")

# TAB 2: RACE (TACTICAL CADDY)
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
        # Init scores
        if hole not in st.session_state.current_scores: st.session_state.current_scores[hole] = {}
        if hole not in st.session_state.selected_discs: st.session_state.selected_discs[hole] = {}
        for p in st.session_state.active_players:
            if p not in st.session_state.current_scores[hole]: st.session_state.current_scores[hole][p] = inf['p']
            if p not in st.session_state.selected_discs[hole]: st.session_state.selected_discs[hole][p] = None

        # SPELAR-KORT
        for p in st.session_state.active_players:
            with st.expander(f"🏎️ {p} (Score: {st.session_state.current_scores[hole][p]})", expanded=True):
                
                # --- TAKTIK-CENTRALEN ---
                with st.container(border=True):
                    st.markdown(f"**Taktik {p}**")
                    
                    # 1. Situations-input
                    c_sit, c_obs = st.columns([1, 1])
                    situation = c_sit.radio("Läge", ["Tee", "Fairway/Inspel", "Ruff/Trubbel", "Putt"], key=f"sit_{hole}_{p}", horizontal=True)
                    dist_left = c_sit.slider("Avstånd kvar (m)", 0, 200, int(inf['l']) if situation=="Tee" else 50, key=f"d_{hole}_{p}")
                    obstacles = c_obs.multiselect("Hinder", ["Träd Vänster", "Träd Höger", "Lågt Tak", "Vatten", "Uppför", "Nedför"], key=f"obs_{hole}_{p}")
                    
                    # 2. Kamera (Valfritt)
                    use_cam = st.checkbox("📸 Fota Läge/Skylt", key=f"cam_tog_{hole}_{p}")
                    img_data = None
                    if use_cam:
                        img_file = st.camera_input("Fota", key=f"ci_{hole}_{p}")
                        if img_file: img_data = img_file.getvalue()

                    # 3. AI-ANALYS KNAPP
                    if st.button(f"🧠 Fråga Caddyn ({p})", key=f"ai_btn_{hole}_{p}"):
                        p_bag = st.session_state.inventory[(st.session_state.inventory["Owner"]==p) & (st.session_state.inventory["Status"]=="Bag")]
                        with st.spinner("Analyserar vind, fysik och bag..."):
                            advice = get_tactical_advice(p, p_bag, dist_left, st.session_state.weather_data, situation, obstacles, img_data)
                            st.session_state.hole_advice[f"{hole}_{p}"] = advice
                    
                    # 4. VISA RÅD
                    if f"{hole}_{p}" in st.session_state.hole_advice:
                        st.info(st.session_state.hole_advice[f"{hole}_{p}"])

                # --- SCORE & STATS ---
                c1, c2, c3 = st.columns([1,2,1])
                if c1.button("➖", key=f"m_{hole}_{p}"): st.session_state.current_scores[hole][p] -= 1; st.rerun()
                c2.markdown(f"<h2 style='text-align:center'>{st.session_state.current_scores[hole][p]}</h2>", unsafe_allow_html=True)
                if c3.button("➕", key=f"p_{hole}_{p}"): st.session_state.current_scores[hole][p] += 1; st.rerun()
                
                # Välj disc för historik
                p_bag = st.session_state.inventory[(st.session_state.inventory["Owner"]==p) & (st.session_state.inventory["Status"]=="Bag")]
                opts = ["Välj Disc"] + p_bag["Modell"].tolist()
                st.session_state.selected_discs[hole][p] = st.selectbox("Disc (Utslag)", opts, key=f"ds_{hole}_{p}")

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

# TAB 5: STATS
with t5:
    st.header("📊 Live Moln-Stats")
    df = st.session_state.history
    if not df.empty:
        c1, c2 = st.columns(2)
        sp = c1.multiselect("Jämför", df["Spelare"].unique())
        if sp:
            dff = df[df["Spelare"].isin(sp)]
            st.bar_chart(dff.groupby("Spelare")["Resultat"].mean())
            try:
                dff['Hål_Int'] = dff['Hål'].astype(int)
                chart = alt.Chart(dff).mark_bar().encode(x='Hål_Int:O', y='mean(Resultat)', color='Spelare', xOffset='Spelare').interactive()
                st.altair_chart(chart, use_container_width=True)
            except: st.bar_chart(dff.groupby("Hål")["Resultat"].mean())
    else: st.info("Databasen är tom.")

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
