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
from io import BytesIO
from PIL import Image

# --- 1. KONFIGURATION & SETUP ---
st.set_page_config(page_title="Scuderia Wonka Caddy", page_icon="🏎️", layout="wide")

# Google Sheets Setup
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

@st.cache_resource
def get_gsheet_client():
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
    return gspread.authorize(creds)

def load_data_from_sheet():
    client = get_gsheet_client()
    try:
        sheet = client.open("DiscCaddy_DB")
        
        # Inventory
        ws_inv = sheet.worksheet("Inventory")
        inv_data = ws_inv.get_all_records()
        df_inv = pd.DataFrame(inv_data)
        if df_inv.empty: df_inv = pd.DataFrame(columns=["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"])
        
        # History
        ws_hist = sheet.worksheet("History")
        hist_data = ws_hist.get_all_records()
        df_hist = pd.DataFrame(hist_data)
        if df_hist.empty: df_hist = pd.DataFrame(columns=["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
        
        return df_inv, df_hist
    except Exception as e:
        st.error(f"Databas-fel: {e}. Kontrollera att flikarna 'Inventory' och 'History' finns i ditt Google Sheet och har rubriker på rad 1.")
        return pd.DataFrame(), pd.DataFrame()

def save_to_sheet(df, worksheet_name):
    client = get_gsheet_client()
    sheet = client.open("DiscCaddy_DB")
    ws = sheet.worksheet(worksheet_name)
    ws.clear()
    ws.update([df.columns.values.tolist()] + df.values.tolist())

# AI & VISION SETUP
def analyze_image(image_bytes):
    # Konvertera bild till base64 för OpenAI
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    client = OpenAI(api_key=st.secrets["openai_key"])
    
    prompt = """
    Titta på denna discgolf-disc. Identifiera:
    1. Tillverkare och Modell (t.ex. Innova Destroyer).
    2. Flight numbers (Speed, Glide, Turn, Fade).
    3. Typ av disc (Putter, Midrange, Fairway Driver, Distance Driver).
    
    Svara EXAKT i detta JSON-format, inget annat prat:
    {
        "Modell": "Tillverkare Modell",
        "Typ": "Typ",
        "Speed": 0.0,
        "Glide": 0.0,
        "Turn": 0.0,
        "Fade": 0.0
    }
    Om du inte ser tydligt, gissa baserat på färg/form eller svara null.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return None

def ask_ai(messages):
    try:
        client = OpenAI(api_key=st.secrets["openai_key"])
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        return response.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# --- 2. STATE ---
if 'data_loaded' not in st.session_state:
    i, h = load_data_from_sheet()
    st.session_state.inventory = i
    st.session_state.history = h
    # Default Courses
    st.session_state.courses = {
        "Kungsbackaskogen": {"lat": 57.492, "lon": 12.075, "holes": {str(x):{"l": y, "p": 3, "shape": "Rak"} for x,y in zip(range(1,10), [63,81,48,65,75,55,62,78,52])}},
        "Lygnevi Sätila": {"lat": 57.545, "lon": 12.433, "holes": {str(x):{"l": 100, "p": 3, "shape": "Rak"} for x in range(1,19)}},
        "Åbyvallen": {"lat": 57.480, "lon": 12.070, "holes": {str(x):{"l": 70, "p": 3, "shape": "Vänster"} for x in range(1,9)}}
    }
    st.session_state.data_loaded = True

if 'active_players' not in st.session_state: st.session_state.active_players = []
if 'current_scores' not in st.session_state: st.session_state.current_scores = {}
if 'selected_discs' not in st.session_state: st.session_state.selected_discs = {}
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'ai_disc_data' not in st.session_state: st.session_state.ai_disc_data = None # För Vision resultatet

# --- 3. LOGIK ---
def suggest_disc(bag, player, dist, shape):
    pb = bag[(bag["Owner"]==player) & (bag["Status"]=="Bag")]
    if pb.empty: return None, "Tom väska"
    
    eff_dist = dist 
    target_speed = eff_dist / 10.0
    candidates = pb.copy()
    candidates["Score"] = abs(candidates["Speed"] - target_speed)
    
    if eff_dist < 40: candidates = candidates[candidates["Typ"]=="Putter"]
    elif eff_dist < 80: candidates = candidates[candidates["Typ"].isin(["Putter","Midrange"])]
    
    if candidates.empty: candidates = pb
    
    if shape == "Höger": best = candidates.sort_values("Fade", ascending=False).iloc[0]; reason="Forehand"
    elif shape == "Vänster": best = candidates.sort_values("Fade", ascending=False).iloc[0]; reason="Hyzer"
    else: best = candidates.sort_values("Turn", ascending=False).iloc[0]; reason="Rakt"
    
    return best, reason

# --- 4. UI ---
with st.sidebar:
    st.title("🏎️ SCUDERIA CLOUD")
    st.caption("🟢 Online & AI-Powered")
    
    all_owners = st.session_state.inventory["Owner"].unique().tolist() if not st.session_state.inventory.empty else []
    active = st.multiselect("Spelare", all_owners, default=st.session_state.active_players)
    if active != st.session_state.active_players:
        st.session_state.active_players = active
        st.rerun()
        
    if st.button("🔄 Synka Databas"):
        i, h = load_data_from_sheet()
        st.session_state.inventory = i; st.session_state.history = h
        st.success("Synkad!")

t1, t2, t3, t4, t5 = st.tabs(["🔥 WARM-UP", "🏁 RACE", "🤖 AI-CADDY", "🧳 UTRUSTNING", "📊 STATS"])

# TAB 1: WARM-UP
with t1:
    st.header("🔥 Kalibrering")
    if st.session_state.active_players:
        p = st.selectbox("Spelare", st.session_state.active_players)
        form = st.slider(f"Dagsform {p} (%)", 50, 150, 100)
        st.metric("Effektiv kraft", f"{form}%")
    else: st.info("Välj spelare i menyn.")

# TAB 2: RACE
with t2:
    courses = list(st.session_state.courses.keys())
    bana = st.selectbox("Bana", courses)
    c_data = st.session_state.courses[bana]
    
    col_n, col_s = st.columns([1, 2])
    with col_n:
        holes = sorted(list(c_data["holes"].keys()), key=lambda x: int(x) if x.isdigit() else x)
        hole = st.selectbox("Hål", holes)
        inf = c_data["holes"][hole]
        st.metric(f"Hål {hole}", f"{inf['l']}m", f"Par {inf['p']}")
        st.caption(inf.get('shape', 'Rak'))

    with col_s:
        if hole not in st.session_state.current_scores: st.session_state.current_scores[hole] = {p: inf['p'] for p in st.session_state.active_players}
        if hole not in st.session_state.selected_discs: st.session_state.selected_discs[hole] = {p: None for p in st.session_state.active_players}
        
        for p in st.session_state.active_players:
            with st.expander(f"{p} - {st.session_state.current_scores[hole][p]}", expanded=True):
                rec, reason = suggest_disc(st.session_state.inventory, p, inf['l'], inf.get('shape', 'Rak'))
                if rec is not None: st.success(f"Caddy: {rec['Modell']} ({reason})")
                
                c1, c2, c3 = st.columns([1,2,1])
                if c1.button("➖", key=f"m_{hole}_{p}"): st.session_state.current_scores[hole][p] -= 1; st.rerun()
                c2.markdown(f"<h2 style='text-align:center'>{st.session_state.current_scores[hole][p]}</h2>", unsafe_allow_html=True)
                if c3.button("➕", key=f"p_{hole}_{p}"): st.session_state.current_scores[hole][p] += 1; st.rerun()
                
                p_bag = st.session_state.inventory[(st.session_state.inventory["Owner"]==p) & (st.session_state.inventory["Status"]=="Bag")]
                opts = ["Välj"] + p_bag["Modell"].tolist()
                st.session_state.selected_discs[hole][p] = st.selectbox("Disc", opts, key=f"d_{hole}_{p}")

    if st.button("🏁 SPARA TILL MOLNET", type="primary"):
        new_rows = []
        d = datetime.now().strftime("%Y-%m-%d")
        for h, scores in st.session_state.current_scores.items():
            for p, s in scores.items():
                disc = st.session_state.selected_discs[h].get(p, "Unknown")
                new_rows.append({"Datum": d, "Bana": bana, "Spelare": p, "Hål": h, "Resultat": s, "Par": c_data["holes"][h]["p"], "Disc_Used": disc})
        
        new_df = pd.DataFrame(new_rows)
        st.session_state.history = pd.concat([st.session_state.history, new_df], ignore_index=True)
        save_to_sheet(st.session_state.history, "History")
        st.balloons(); st.success("Rundan sparad i Google Sheets!"); st.session_state.current_scores = {}

# TAB 3: AI-CADDY
with t3:
    st.header("🤖 Chatta med Scuderia AI")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Fråga caddyn..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        bag_info = ""
        if st.session_state.active_players:
            p = st.session_state.active_players[0]
            my_discs = st.session_state.inventory[st.session_state.inventory["Owner"]==p]["Modell"].tolist()
            bag_info = f"Min väska innehåller: {', '.join(my_discs)}."

        context = f"Du är en elit-discgolf caddy. {bag_info}. Svara kort och strategiskt."
        messages = [{"role": "system", "content": context}] + st.session_state.chat_history
        with st.chat_message("assistant"):
            with st.spinner("Tänker..."):
                reply = ask_ai(messages)
                st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

# TAB 4: UTRUSTNING (MED AI VISION)
with t4:
    st.header("🧳 Moln-Bag")
    owner = st.selectbox("Hantera", st.session_state.active_players) if st.session_state.active_players else None
    
    st.subheader("📸 Lägg till med AI")
    img_file = st.camera_input("Fota discen")
    
    if img_file:
        if st.button("🔍 Analysera Bild"):
            with st.spinner("AI-ögat skannar discen..."):
                bytes_data = img_file.getvalue()
                json_str = analyze_image(bytes_data)
                try:
                    # Rensa bort eventuell markdown ```json ... ```
                    json_str = json_str.replace("```json", "").replace("```", "").strip()
                    data = json.loads(json_str)
                    st.session_state.ai_disc_data = data
                    st.success("Identifierad!")
                except:
                    st.error("Kunde inte läsa av discen. Försök igen med bättre ljus.")

    # Formulär (Fylls i av AI om data finns)
    with st.form("add_cloud"):
        ai_d = st.session_state.ai_disc_data if st.session_state.ai_disc_data else {}
        
        c1, c2 = st.columns(2)
        mn = c1.text_input("Modell", value=ai_d.get("Modell", ""))
        ty = c2.selectbox("Typ", ["Putter", "Midrange", "Fairway Driver", "Distance Driver"], index=["Putter", "Midrange", "Fairway Driver", "Distance Driver"].index(ai_d.get("Typ", "Putter")) if ai_d.get("Typ") in ["Putter", "Midrange", "Fairway Driver", "Distance Driver"] else 0)
        
        c3, c4, c5, c6 = st.columns(4)
        sp = c3.number_input("Speed", 0.0, 15.0, float(ai_d.get("Speed", 7.0)))
        gl = c4.number_input("Glide", 0.0, 7.0, float(ai_d.get("Glide", 5.0)))
        tu = c5.number_input("Turn", -5.0, 1.0, float(ai_d.get("Turn", 0.0)))
        fa = c6.number_input("Fade", 0.0, 6.0, float(ai_d.get("Fade", 2.0)))
        
        if st.form_submit_button("Spara till Databas"):
            nw = {"Owner": owner, "Modell": mn, "Typ": ty, "Speed": sp, "Glide": gl, "Turn": tu, "Fade": fa, "Status": "Bag"}
            st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame([nw])], ignore_index=True)
            save_to_sheet(st.session_state.inventory, "Inventory")
            st.success(f"{mn} sparad i molnet!")
            st.session_state.ai_disc_data = None # Rensa efter sparning
            st.rerun()
            
    if owner:
        st.subheader("Din Bag")
        st.dataframe(st.session_state.inventory[st.session_state.inventory["Owner"]==owner])

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
            
            st.subheader("Hål-analys")
            try:
                dff['Hål_Int'] = dff['Hål'].astype(int)
                chart = alt.Chart(dff).mark_bar().encode(
                    x='Hål_Int:O', y='mean(Resultat)', color='Spelare', xOffset='Spelare'
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            except: st.bar_chart(dff.groupby("Hål")["Resultat"].mean())
    else: st.info("Databasen är tom.")
