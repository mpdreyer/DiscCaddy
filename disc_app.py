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
        
        # --- INVENTORY ---
        try: ws_inv = sheet.worksheet("Inventory")
        except:
            ws_inv = sheet.add_worksheet("Inventory", 100, 10)
            ws_inv.append_row(["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"])
            
        inv_data = ws_inv.get_all_records()
        df_inv = pd.DataFrame(inv_data)
        
        expected_inv = ["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"]
        if df_inv.empty: 
            df_inv = pd.DataFrame(columns=expected_inv)
        else:
            for col in ["Speed", "Glide", "Turn", "Fade"]:
                df_inv[col] = pd.to_numeric(df_inv[col], errors='coerce').fillna(0)
            if "Status" not in df_inv.columns: df_inv["Status"] = "Shelf"
            df_inv["Status"] = df_inv["Status"].fillna("Shelf")

        # --- HISTORY ---
        try: ws_hist = sheet.worksheet("History")
        except:
            ws_hist = sheet.add_worksheet("History", 100, 10)
            ws_hist.append_row(["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
            
        hist_data = ws_hist.get_all_records()
        df_hist = pd.DataFrame(hist_data)
        expected_hist = ["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"]
        if df_hist.empty: df_hist = pd.DataFrame(columns=expected_hist)
        
        return df_inv, df_hist

    except Exception as e:
        st.error(f"Databas-fel: {e}")
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
    except Exception as e: st.error(f"Sparfel: {e}")

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
        prompt = """
        Identifiera discen.
        VIKTIGT OM TYP: Måste vara exakt en av dessa strängar: 'Putter', 'Midrange', 'Fairway Driver', 'Distance Driver'.
        Svara EXAKT JSON: 
        {
            "Modell": "Tillverkare Modell", 
            "Typ": "Fairway Driver", 
            "Speed": 7.0, 
            "Glide": 5.0, 
            "Turn": 0.0, 
            "Fade": 2.0
        }
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except: return None

# --- 2. STATE ---
if 'data_loaded' not in st.session_state:
    with st.spinner("Laddar molndata..."):
        i, h = load_data_from_sheet()
        st.session_state.inventory = i
        st.session_state.history = h
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
if 'camera_active' not in st.session_state: st.session_state.camera_active = False
if 'suggested_pack' not in st.session_state: st.session_state.suggested_pack = []
if 'warmup_shots' not in st.session_state: st.session_state.warmup_shots = []

# --- 3. LOGIK ---
def suggest_disc(bag, player, dist, shape, form=1.0, wind_str=0, wind_dir="Stilla"):
    pb = bag[(bag["Owner"]==player) & (bag["Status"]=="Bag")]
    if pb.empty: return None, "Tom väska"
    
    # Justera effektiv distans baserat på dagsform
    eff_dist = dist / max(form, 0.5)
    
    # Justera target speed baserat på vind
    # Motvind = Discen beter sig som lägre speed/mer understabil -> Vi behöver mer stabilitet
    # Medvind = Discen beter sig som mer överstabil -> Vi behöver understabilitet/glide
    
    target_speed = eff_dist / 10.0
    
    pb = pb.copy()
    for c in ["Speed", "Turn", "Fade"]:
        pb[c] = pd.to_numeric(pb[c], errors='coerce').fillna(0)
    
    # VIND-JUSTERING AV DISCARNA
    if "Motvind" in wind_dir:
        # Motvind gör discen mer "Turny". Vi låtsas att discens Turn är lägre än den är för att hitta stabilare discar.
        pb["Eff_Turn"] = pb["Turn"] - (wind_str * 0.5) 
        advice_suffix = " (Motvind: Välj Stabil)"
    elif "Medvind" in wind_dir:
        # Medvind gör discen mer stabil. Vi behöver discar med högre Turn (mindre stabil) eller mer Glide.
        pb["Eff_Turn"] = pb["Turn"] + (wind_str * 0.3)
        advice_suffix = " (Medvind: Välj Glide/Turn)"
    else:
        pb["Eff_Turn"] = pb["Turn"]
        advice_suffix = ""

    pb["Speed_Diff"] = abs(pb["Speed"] - target_speed)
    candidates = pb.copy()
    
    if eff_dist < 45: candidates = candidates[candidates["Typ"]=="Putter"]
    elif eff_dist < 85: candidates = candidates[candidates["Typ"].isin(["Putter","Midrange"])]
    elif eff_dist < 110: candidates = candidates[candidates["Typ"].isin(["Midrange", "Fairway Driver"])]
    
    if candidates.empty: candidates = pb
    
    # Score baseras nu på Eff_Turn (Vindjusterad)
    if form < 0.9: candidates["Score"] = candidates["Speed_Diff"] + (candidates["Eff_Turn"] * 0.5)
    else: candidates["Score"] = candidates["Speed_Diff"]
    
    if shape == "Höger": best = candidates.sort_values(by=["Score", "Fade"], ascending=[True, False]).iloc[0]; reason="Forehand" + advice_suffix
    elif shape == "Vänster": best = candidates.sort_values(by=["Score", "Fade"], ascending=[True, False]).iloc[0]; reason="Hyzer" + advice_suffix
    else: best = candidates.sort_values(by=["Score", "Eff_Turn"], ascending=[True, True]).iloc[0]; reason="Rakt" + advice_suffix
    return best, reason

def generate_smart_bag(inventory, player, course_name):
    holes = st.session_state.courses[course_name]["holes"]
    avg_len = np.mean([h["l"] for h in holes.values()])
    all_discs = inventory[inventory["Owner"] == player]
    pack_indices = []
    
    putters = all_discs[all_discs["Typ"] == "Putter"].sort_values("Speed")
    if not putters.empty: pack_indices.append(putters.iloc[0].name)
    mids = all_discs[all_discs["Typ"] == "Midrange"]
    if not mids.empty: pack_indices.append(mids.sort_values("Glide", ascending=False).iloc[0].name)
    fairways = all_discs[all_discs["Typ"] == "Fairway Driver"]
    if not fairways.empty: pack_indices.append(fairways.iloc[0].name)
    if avg_len > 80:
        drivers = all_discs[all_discs["Typ"] == "Distance Driver"]
        if not drivers.empty: pack_indices.append(drivers.iloc[0].name)
    else:
        if len(putters) > 1: pack_indices.append(putters.iloc[1].name)
    return list(set(pack_indices))

# --- 4. UI ---
with st.sidebar:
    st.title("🏎️ SCUDERIA CLOUD")
    st.caption("🟢 v35.0 Aero-Dynamic")
    
    # VÄDERSTATION
    with st.expander("🌪️ Väderstation", expanded=True):
        wind_str = st.slider("Vindstyrka (m/s)", 0, 15, 2)
        wind_dir = st.radio("Vindriktning", ["Stilla", "Motvind", "Medvind", "Sidvind"], index=0)
        temp = st.slider("Temperatur (°C)", -5, 35, 20)
    
    st.divider()
    
    all_owners = st.session_state.inventory["Owner"].unique().tolist() if not st.session_state.inventory.empty else []
    
    new_p = st.text_input("Ny spelare:", placeholder="Namn")
    if st.button("Lägg till") and new_p:
        start_kit = [{"Owner": new_p, "Modell": "Start Putter", "Typ": "Putter", "Speed": 3, "Glide": 3, "Turn": 0, "Fade": 0, "Status": "Bag"}]
        st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame(start_kit)], ignore_index=True)
        save_to_sheet(st.session_state.inventory, "Inventory")
        st.success("Spelare skapad!"); st.rerun()

    active = st.multiselect("Välj Spelare", all_owners, default=st.session_state.active_players)
    if active != st.session_state.active_players:
        st.session_state.active_players = active
        for p in active:
            if p not in st.session_state.daily_forms: st.session_state.daily_forms[p] = 1.0
        st.rerun()
        
    if st.button("🔄 Synka Databas"):
        st.cache_resource.clear()
        st.rerun()

t1, t2, t3, t4, t5, t6 = st.tabs(["🔥 WARM-UP", "🏁 RACE", "🤖 AI-CADDY", "🧳 UTRUSTNING", "📊 STATS", "⚙️ ADMIN"])

# TAB 1: WARM-UP (Physics Engine + Styles + Disc Potential)
with t1:
    st.header("🔥 Driving Range")
    
    if st.session_state.active_players:
        curr_p = st.selectbox("Kalibrera Spelare:", st.session_state.active_players)
        
        # Hämta discar
        p_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == curr_p]
        disc_options = ["Välj Disc"] + p_inv["Modell"].unique().tolist()
        
        c_in, c_list = st.columns([1, 1])
        
        # --- DEL 1: INMATNING ---
        with c_in:
            with st.container(border=True):
                st.subheader("1. Kasta & Mät")
                sel_disc_name = st.selectbox("Vilken disc?", disc_options)
                
                # NYTT: Välj stil
                style = st.radio("Kast-stil", ["Backhand (RHBH)", "Forehand (RHFH)"], horizontal=True)
                
                c_d, c_s = st.columns(2)
                kast_len = c_d.number_input("Längd (m)", 0, 200, 50, step=5)
                kast_sida = c_s.number_input("Sida (m)", -50, 50, 0, step=1, help="-Vänster / +Höger")
                
                if st.button("➕ Registrera Kast", type="primary"):
                    if sel_disc_name != "Välj Disc" and kast_len > 0:
                        d_data = p_inv[p_inv["Modell"]==sel_disc_name].iloc[0]
                        st.session_state.warmup_shots.append({
                            "disc": sel_disc_name,
                            "style": style,
                            "len": kast_len,
                            "side": kast_sida,
                            "speed": float(d_data["Speed"]),
                            "turn": float(d_data["Turn"]),
                            "fade": float(d_data["Fade"])
                        })
                        st.success("Kast registrerat!")
                    else:
                        st.error("Välj disc och ange längd.")

        # --- DEL 2: LISTA & ANALYS ---
        with c_list:
            st.subheader("2. Dina kast")
            if st.session_state.warmup_shots:
                shots_df = pd.DataFrame(st.session_state.warmup_shots)
                st.dataframe(shots_df[["disc", "style", "len", "side"]], hide_index=True, height=150)
                
                if st.button("🗑️ Rensa lista"):
                    st.session_state.warmup_shots = []
                    st.rerun()
            else:
                st.info("Inga kast registrerade än.")

        st.markdown("---")
        
        # --- DEL 3: RESULTAT & DISC-POTENTIAL ---
        if st.session_state.warmup_shots:
            st.subheader("3. Analys & Form")
            
            shots = st.session_state.warmup_shots
            
            # Beräkna Formfaktor baserat på DISCENS Potential (Speed * 10)
            total_potential_ratio = 0
            total_tech_side = 0
            
            for s in shots:
                # 1. Discens Optimala Längd (Tumregel: Speed * 10m)
                # En Speed 12 ska gå 120m. En Speed 3 ska gå 30-40m.
                optimal_dist = max(s["speed"] * 10.0, 40.0) 
                ratio = s["len"] / optimal_dist
                total_potential_ratio += ratio
                
                # 2. Teknik-analys (Sida)
                req_dist = s["speed"] * 10.0
                power_ratio = s["len"] / req_dist if req_dist > 0 else 1.0
                expected_fade = s["fade"]
                expected_turn = s["turn"] if power_ratio > 0.9 else 0
                
                if "Backhand" in s["style"]:
                    fade_dir = -1; turn_dir = 1
                else: 
                    fade_dir = 1; turn_dir = -1

                if power_ratio < 0.8: natural_side = (expected_fade * 3 * fade_dir)
                else: natural_side = (expected_turn * 2 * turn_dir) + (expected_fade * 2 * fade_dir)
                
                total_tech_side += (s["side"] - natural_side)

            # Slutlig Form
            avg_form = total_potential_ratio / len(shots)
            avg_tech_side = total_tech_side / len(shots)
            
            # Spara formen
            st.session_state.daily_forms[curr_p] = avg_form

            c_res, c_gr = st.columns(2)
            
            with c_res:
                st.metric("Utnyttjad Potential", f"{int(avg_form*100)}%")
                st.caption("Jämfört med vad dina discar är designade för.")
                
                if avg_form > 1.0: st.success("💪 Du kastar längre än discens rating!")
                elif avg_form < 0.7: st.warning("⚠️ Du får inte upp discarna i fart.")
                
                st.markdown("**Teknik:**")
                if abs(avg_tech_side) < 7: st.success("✅ Rena träffar!")
                else: 
                    dir_txt = "Vänster" if avg_tech_side < 0 else "Höger"
                    st.warning(f"⚠️ Teknikfel: Missar {dir_txt} om linjen ({int(abs(avg_tech_side))}m).")
                    
            with c_gr:
                fig, ax = plt.subplots(figsize=(4,4))
                x_vals = [s["side"] for s in shots]
                y_vals = [s["len"] for s in shots]
                colors = ['#cc0000' if "Backhand" in s["style"] else '#0066cc' for s in shots]
                ax.scatter(x_vals, y_vals, c=colors, s=100, alpha=0.7)
                ax.axvline(0, color='gray', linestyle='--')
                ax.set_xlim(-40, 40); ax.set_ylim(0, max(y_vals)*1.2)
                ax.set_title("Träffbild (Röd=BH, Blå=FH)")
                st.pyplot(fig)

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
        st.metric(f"Hål {hole}", f"{inf['l']}m", f"Par {inf['p']}"); st.caption(inf.get('shape', 'Rak'))
    with col_s:
        if hole not in st.session_state.current_scores: st.session_state.current_scores[hole] = {}
        if hole not in st.session_state.selected_discs: st.session_state.selected_discs[hole] = {}
        for p in st.session_state.active_players:
            if p not in st.session_state.current_scores[hole]: st.session_state.current_scores[hole][p] = inf['p']
            if p not in st.session_state.selected_discs[hole]: st.session_state.selected_discs[hole][p] = None
        for p in st.session_state.active_players:
            with st.expander(f"{p} - {st.session_state.current_scores[hole][p]}", expanded=True):
                curr_form = st.session_state.daily_forms.get(p, 1.0)
                # SKICKA MED VIND-DATA HÄR
                rec, reason = suggest_disc(st.session_state.inventory, p, inf['l'], inf.get('shape', 'Rak'), curr_form, wind_str, wind_dir)
                if rec is not None: st.success(f"Caddy: {rec['Modell']} ({reason})")
                else: st.warning("Tom väska")
                c1, c2, c3 = st.columns([1,2,1])
                if c1.button("➖", key=f"m_{hole}_{p}"): st.session_state.current_scores[hole][p] -= 1; st.rerun()
                c2.markdown(f"<h2 style='text-align:center'>{st.session_state.current_scores[hole][p]}</h2>", unsafe_allow_html=True)
                if c3.button("➕", key=f"p_{hole}_{p}"): st.session_state.current_scores[hole][p] += 1; st.rerun()
                p_bag = st.session_state.inventory[(st.session_state.inventory["Owner"]==p) & (st.session_state.inventory["Status"]=="Bag")]
                opts = ["Välj"] + p_bag["Modell"].tolist()
                st.session_state.selected_discs[hole][p] = st.selectbox("Disc", opts, key=f"d_{hole}_{p}")
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

# TAB 4: UTRUSTNING (LOGISTICS)
with t4:
    st.header("🧳 Logistik-Center")
    default_owner = st.session_state.active_players[0] if st.session_state.active_players else None
    owner = st.selectbox("Hantera", st.session_state.active_players, index=0) if st.session_state.active_players else None
    
    # 1. STRATEGEN
    with st.container(border=True):
        st.markdown("#### 🤖 Strategen")
        c1, c2, c3 = st.columns([2, 1, 1])
        tc = c1.selectbox("Bana:", list(st.session_state.courses.keys()))
        if c2.button("Generera"): st.session_state.suggested_pack = generate_smart_bag(st.session_state.inventory, owner, tc); st.rerun()
        if st.session_state.suggested_pack:
            pack_names = st.session_state.inventory.loc[st.session_state.suggested_pack, "Modell"].tolist()
            c1.info(f"Föreslår: {', '.join(pack_names)}")
            if c3.button("Verkställ", type="primary"):
                st.session_state.inventory.loc[st.session_state.inventory["Owner"]==owner, "Status"] = "Shelf"
                st.session_state.inventory.loc[st.session_state.suggested_pack, "Status"] = "Bag"
                save_to_sheet(st.session_state.inventory, "Inventory")
                st.session_state.suggested_pack = []
                st.success("Bagen packad!"); st.rerun()

    # 2. LISTOR (Sortering & Delete)
    if owner:
        st.markdown("---")
        sort_mode = st.radio("Sortera på:", ["Speed", "Modell", "Typ"], horizontal=True)
        my_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == owner]
        
        c_shelf = st.container(border=True)
        c_bag = st.container(border=True)
        
        with c_shelf:
            st.subheader("🏠 Hyllan")
            shelf = my_inv[my_inv["Status"] == "Shelf"].sort_values(sort_mode)
            if shelf.empty: st.caption("Tomt.")
            else:
                for idx, row in shelf.iterrows():
                    c_txt, c_btn, c_del = st.columns([3, 1, 0.5])
                    c_txt.text(f"{row['Modell']} ({int(row['Speed'])})")
                    if c_btn.button("➡️", key=f"s2b_{idx}"):
                        st.session_state.inventory.at[idx, "Status"] = "Bag"
                        save_to_sheet(st.session_state.inventory, "Inventory"); st.rerun()
                    if c_del.button("🗑️", key=f"del_s_{idx}"):
                        st.session_state.inventory = st.session_state.inventory.drop(idx)
                        save_to_sheet(st.session_state.inventory, "Inventory"); st.rerun()
                        
        with c_bag:
            st.subheader("🎒 Bagen")
            bag = my_inv[my_inv["Status"] == "Bag"].sort_values(sort_mode)
            if bag.empty: st.caption("Tomt.")
            else:
                for idx, row in bag.iterrows():
                    c_btn, c_txt, c_del = st.columns([1, 3, 0.5])
                    if c_btn.button("⬅️", key=f"b2s_{idx}"):
                        st.session_state.inventory.at[idx, "Status"] = "Shelf"
                        save_to_sheet(st.session_state.inventory, "Inventory"); st.rerun()
                    c_txt.text(f"{row['Modell']} ({int(row['Speed'])})")
                    if c_del.button("🗑️", key=f"del_b_{idx}"):
                        st.session_state.inventory = st.session_state.inventory.drop(idx)
                        save_to_sheet(st.session_state.inventory, "Inventory"); st.rerun()

    # 3. LÄGG TILL
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
                            st.session_state.ai_disc_data = json.loads(json_str)
                            st.success("Hittad!")
                        except: st.error("Försök igen.")

        with st.form("add_cloud"):
            ai_d = st.session_state.ai_disc_data if st.session_state.ai_disc_data else {}
            c1, c2 = st.columns(2)
            mn = c1.text_input("Modell", value=ai_d.get("Modell", ""))
            v_types = ["Putter", "Midrange", "Fairway Driver", "Distance Driver"]
            r_type = ai_d.get("Typ", "Putter")
            f_idx = 0
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
                st.success(f"{mn} sparad!")
                st.session_state.ai_disc_data = None; st.rerun()

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
    st.subheader("⚙️ Admin & Import")
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
