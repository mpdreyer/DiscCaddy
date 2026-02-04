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

# --- 1. KONFIGURATION & SETUP ---
st.set_page_config(page_title="Scuderia Wonka Caddy", page_icon="🏎️", layout="wide")

# Google Sheets Setup
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

@st.cache_resource
def get_gsheet_client():
    # Hämtar nycklar från Streamlit Secrets
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
        
        # --- INVENTORY ---
        try:
            ws_inv = sheet.worksheet("Inventory")
        except:
            ws_inv = sheet.add_worksheet("Inventory", 100, 10)
            ws_inv.append_row(["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"])
            
        inv_data = ws_inv.get_all_records()
        df_inv = pd.DataFrame(inv_data)
        # Säkra kolumner om arket är tomt men har rubriker
        expected_inv = ["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"]
        if df_inv.empty: df_inv = pd.DataFrame(columns=expected_inv)
        
        # --- HISTORY ---
        try:
            ws_hist = sheet.worksheet("History")
        except:
            ws_hist = sheet.add_worksheet("History", 100, 10)
            ws_hist.append_row(["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
            
        hist_data = ws_hist.get_all_records()
        df_hist = pd.DataFrame(hist_data)
        expected_hist = ["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"]
        if df_hist.empty: df_hist = pd.DataFrame(columns=expected_hist)
        
        return df_inv, df_hist

    except Exception as e:
        st.error(f"Kunde inte ladda databasen: {e}")
        return pd.DataFrame(), pd.DataFrame()

def save_to_sheet(df, worksheet_name):
    client = get_gsheet_client()
    if not client: return
    try:
        sheet = client.open("DiscCaddy_DB")
        try: ws = sheet.worksheet(worksheet_name)
        except: ws = sheet.add_worksheet(worksheet_name, 100, 10)
        
        ws.clear()
        ws.update([df.columns.values.tolist()] + df.values.tolist())
    except Exception as e:
        st.error(f"Kunde inte spara: {e}")

# AI Setup
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
        prompt = "Identifiera discen. Svara EXAKT JSON: {\"Modell\": \"Namn\", \"Typ\": \"Putter/Midrange/Fairway/Distance\", \"Speed\": 0.0, \"Glide\": 0.0, \"Turn\": 0.0, \"Fade\": 0.0}. Gissa om otydligt."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except: return None

# --- 2. STATE ---
if 'data_loaded' not in st.session_state:
    with st.spinner("Kopplar upp mot molnet..."):
        i, h = load_data_from_sheet()
        st.session_state.inventory = i
        st.session_state.history = h
    
    # Banor (Static Backup)
    st.session_state.courses = {
        "Kungsbackaskogen": {"lat": 57.492, "lon": 12.075, "holes": {str(x):{"l": y, "p": 3, "shape": "Rak"} for x,y in zip(range(1,10), [63,81,48,65,75,55,62,78,52])}},
        "Lygnevi Sätila": {"lat": 57.545, "lon": 12.433, "holes": {str(x):{"l": 100, "p": 3, "shape": "Rak"} for x in range(1,19)}},
        "Åbyvallen": {"lat": 57.480, "lon": 12.070, "holes": {str(x):{"l": 70, "p": 3, "shape": "Vänster"} for x in range(1,9)}}
    }
    st.session_state.data_loaded = True

if 'active_players' not in st.session_state: st.session_state.active_players = []
if 'current_scores' not in st.session_state: st.session_state.current_scores = {}
if 'selected_discs' not in st.session_state: st.session_state.selected_discs = {}
if 'daily_forms' not in st.session_state: st.session_state.daily_forms = {}
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'ai_disc_data' not in st.session_state: st.session_state.ai_disc_data = None

# --- 3. LOGIK (SMART CADDY v23 RESTORED) ---
def suggest_disc(bag, player, dist, shape, form=1.0):
    pb = bag[(bag["Owner"]==player) & (bag["Status"]=="Bag")]
    if pb.empty: return None, "Tom väska"
    
    # Justera för form
    eff_dist = dist / max(form, 0.5)
    target_speed = eff_dist / 10.0
    
    candidates = pb.copy()
    candidates["Speed_Diff"] = abs(candidates["Speed"] - target_speed)
    
    if eff_dist < 40: candidates = candidates[candidates["Typ"]=="Putter"]
    elif eff_dist < 75: candidates = candidates[candidates["Typ"].isin(["Putter","Midrange"])]
    elif eff_dist < 100: candidates = candidates[candidates["Typ"].isin(["Midrange", "Fairway Driver"])]
    
    if candidates.empty: candidates = pb
    
    # Form-anpassning
    if form < 0.9: candidates["Score"] = candidates["Speed_Diff"] + (candidates["Turn"] * 0.5)
    else: candidates["Score"] = candidates["Speed_Diff"]
    
    if shape == "Höger": best = candidates.sort_values(by=["Score", "Fade"], ascending=[True, False]).iloc[0]; reason="Forehand"
    elif shape == "Vänster": best = candidates.sort_values(by=["Score", "Fade"], ascending=[True, False]).iloc[0]; reason="Backhand Hyzer"
    else: 
        candidates["Neu"] = abs(candidates["Turn"] + candidates["Fade"])
        best = candidates.sort_values(by=["Score", "Neu"], ascending=[True, True]).iloc[0]; reason="Rakt"
        
    return best, reason

def simulate_flight_path_hd(speed, glide, turn, fade, power, throw_type="Backhand"):
    req_power = speed * 10.0; p_factor = min(power / req_power if req_power > 0 else 1.0, 1.2)
    eff_turn = max(turn, turn * ((p_factor - 0.7) * 2.5)) if p_factor > 0.8 else 0
    eff_fade = fade * (1.5 / max(p_factor, 0.3)) if p_factor < 0.8 else fade
    fade_start = max(0.2, p_factor - 0.2) if p_factor < 0.8 else 0.7
    steps = 200; x = []; y = []; total_dist = power * (1 + (glide/40.0)) * (0.85 if speed > 10 and power < 70 else 1.0)
    for i in range(steps):
        t = i / steps; y.append(t * total_dist)
        turn_p = np.sin(t * np.pi) * (1-t)
        fade_p = ((t - fade_start)/(1 - fade_start))**2 if t > fade_start else 0
        val = (eff_turn * turn_p * 2.5) + (eff_fade * fade_p * -4.0)
        x.append(-val)
    if throw_type == "Forehand": x = [-v for v in x]
    return x, y

# --- 4. UI ---
with st.sidebar:
    st.title("🏎️ SCUDERIA CLOUD")
    st.caption("🟢 Live")
    
    # Lobby Logic
    all_owners = st.session_state.inventory["Owner"].unique().tolist() if not st.session_state.inventory.empty else []
    
    # Lägg till spelare om listan är tom
    new_p = st.text_input("Ny spelare (Skapa)", placeholder="Namn")
    if st.button("Lägg till") and new_p:
        start_kit = [{"Owner": new_p, "Modell": "Start Putter", "Typ": "Putter", "Speed": 3, "Glide": 3, "Turn": 0, "Fade": 0, "Status": "Bag"}]
        st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame(start_kit)], ignore_index=True)
        save_to_sheet(st.session_state.inventory, "Inventory")
        st.success("Spelare skapad!"); st.rerun()

    active = st.multiselect("Välj Spelare", all_owners, default=st.session_state.active_players)
    if active != st.session_state.active_players:
        st.session_state.active_players = active
        # Nollställ form för nya spelare
        for p in active:
            if p not in st.session_state.daily_forms: st.session_state.daily_forms[p] = 1.0
        st.rerun()
        
    if st.button("🔄 Synka Databas"):
        st.cache_resource.clear()
        st.rerun()

# VIKTIGT: 6 FLIKAR NU!
t1, t2, t3, t4, t5, t6 = st.tabs(["🔥 WARM-UP", "🏁 RACE", "🤖 AI-CADDY", "🧳 UTRUSTNING", "📊 STATS", "⚙️ ADMIN"])

# TAB 1: WARM-UP
with t1:
    st.header("🔥 Kalibrering")
    if st.session_state.active_players:
        p = st.selectbox("Justera form för:", st.session_state.active_players)
        f_val = st.slider(f"Dagsform {p} (%)", 50, 150, 100, key=f"form_{p}")
        st.session_state.daily_forms[p] = f_val / 100.0
        st.metric("Effektiv kraft", f"{f_val}%")
    else: st.info("Välj spelare i sidomenyn för att börja.")

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
                # Form-aware Caddy
                curr_form = st.session_state.daily_forms.get(p, 1.0)
                rec, reason = suggest_disc(st.session_state.inventory, p, inf['l'], inf.get('shape', 'Rak'), curr_form)
                
                if rec is not None: st.success(f"Caddy: {rec['Modell']} ({reason})")
                else: st.warning("Tom väska")
                
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
        st.balloons(); st.success("Rundan sparad!"); st.session_state.current_scores = {}

# TAB 3: AI-CADDY
with t3:
    st.header("🤖 AI-Chatt")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Fråga caddyn..."):
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
    st.header("🧳 Moln-Bag")
    owner = st.selectbox("Hantera", st.session_state.active_players) if st.session_state.active_players else None
    
    # KAMERA FIX: Gömd bakom expander/checkbox så den inte autostartar
    with st.expander("📸 Lägg till Disc med Kamera"):
        if st.checkbox("Starta Kamera"):
            img_file = st.camera_input("Fota discen")
            if img_file:
                if st.button("🔍 Analysera"):
                    with st.spinner("AI jobbar..."):
                        b_data = img_file.getvalue()
                        json_str = analyze_image(b_data)
                        try:
                            json_str = json_str.replace("```json", "").replace("```", "").strip()
                            st.session_state.ai_disc_data = json.loads(json_str)
                            st.success("Hittad!")
                        except: st.error("Försök igen.")

    # Formulär
    with st.form("add_cloud"):
        ai_d = st.session_state.ai_disc_data if st.session_state.ai_disc_data else {}
        c1, c2 = st.columns(2)
        mn = c1.text_input("Modell", value=ai_d.get("Modell", ""))
        ty = c2.selectbox("Typ", ["Putter", "Midrange", "Fairway Driver", "Distance Driver"], index=0) # Index fix kan läggas till
        c3, c4, c5, c6 = st.columns(4)
        sp = c3.number_input("Speed", 0.0, 15.0, float(ai_d.get("Speed", 7.0)))
        gl = c4.number_input("Glide", 0.0, 7.0, float(ai_d.get("Glide", 5.0)))
        tu = c5.number_input("Turn", -5.0, 1.0, float(ai_d.get("Turn", 0.0)))
        fa = c6.number_input("Fade", 0.0, 6.0, float(ai_d.get("Fade", 2.0)))
        
        if st.form_submit_button("Spara till Databas"):
            nw = {"Owner": owner, "Modell": mn, "Typ": ty, "Speed": sp, "Glide": gl, "Turn": tu, "Fade": fa, "Status": "Bag"}
            st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame([nw])], ignore_index=True)
            save_to_sheet(st.session_state.inventory, "Inventory")
            st.success(f"{mn} sparad!")
            st.session_state.ai_disc_data = None
            st.rerun()
            
    if owner:
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
            try:
                dff['Hål_Int'] = dff['Hål'].astype(int)
                chart = alt.Chart(dff).mark_bar().encode(x='Hål_Int:O', y='mean(Resultat)', color='Spelare', xOffset='Spelare').interactive()
                st.altair_chart(chart, use_container_width=True)
            except: st.bar_chart(dff.groupby("Hål")["Resultat"].mean())
    else: st.info("Databasen är tom.")

# TAB 6: ADMIN (ÅTERSTÄLLD!)
with t6:
    st.subheader("⚙️ Admin & Import")
    
    st.markdown("### 📥 Importera UDisc CSV")
    up = st.file_uploader("Ladda upp CSV", type=['csv'])
    if up and st.button("Kör Import"):
        try:
            udf = pd.read_csv(up)
            nd = []
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
                st.session_state.history = new_hist
                save_to_sheet(new_hist, "History")
                st.success(f"Importerade {len(nd)} rader till molnet!")
        except Exception as e: st.error(f"Fel: {e}")
