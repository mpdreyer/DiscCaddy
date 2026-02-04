import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
# Förberedelse för bildhantering
from PIL import Image

# --- 1. KONFIGURATION ---
st.set_page_config(page_title="Scuderia Wonka Caddy", page_icon="🏎️", layout="wide")
FILES = {"inv": "inventory.csv", "hist": "round_history.csv", "courses": "courses.json"}

# --- 2. CSS ---
st.markdown("""
    <style>
    div.stButton > button { background-color: #cc0000; color: white; border-radius: 6px; font-weight: bold; width: 100%; }
    .stat-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #cc0000; box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: black; margin-bottom: 10px;}
    h3 { font-size: 1.1em; font-weight: bold; color: #333; margin-bottom: 0px;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & INIT ---
DEFAULT_COURSES = {
    "Kungsbackaskogen": {"lat": 57.492, "lon": 12.075, "holes": {str(h): {"l": l, "p": 3, "shape": "Rak", "desc": ""} for h, l in zip(range(1,10), [63, 81, 48, 65, 75, 55, 62, 78, 52])}},
    "Lygnevi Sätila": {"lat": 57.545, "lon": 12.433, "holes": {str(h): {"l": 100, "p": 3, "shape": "Rak", "desc": ""} for h in range(1,19)}},
    "Åbyvallen": {"lat": 57.480, "lon": 12.070, "holes": {str(h): {"l": 70, "p": 3, "shape": "Vänster", "desc": ""} for h in range(1,9)}},
}
DEFAULT_COURSES["Kungsbackaskogen"]["holes"]["5"]["desc"] = "Korg till vänster efter 55m."

def load_data():
    if os.path.exists(FILES["inv"]):
        try:
            inv = pd.read_csv(FILES["inv"])
            for c in ["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"]:
                if c not in inv.columns: inv[c] = "Bag" if c == "Status" else 0
        except: inv = pd.DataFrame(columns=["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"])
    else: inv = pd.DataFrame(columns=["Owner", "Modell", "Typ", "Speed", "Glide", "Turn", "Fade", "Status"])

    if os.path.exists(FILES["hist"]):
        try:
            hist = pd.read_csv(FILES["hist"])
            if "Hål" not in hist.columns: hist = pd.DataFrame(columns=["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
        except: hist = pd.DataFrame(columns=["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
    else: hist = pd.DataFrame(columns=["Datum", "Bana", "Spelare", "Hål", "Resultat", "Par", "Disc_Used"])
    
    courses = json.load(open(FILES["courses"])) if os.path.exists(FILES["courses"]) else DEFAULT_COURSES
    return inv, hist, courses

def save_data(inv, hist, courses):
    inv.to_csv(FILES["inv"], index=False)
    hist.to_csv(FILES["hist"], index=False)
    json.dump(courses, open(FILES["courses"], "w"))

# --- 4. STATE ---
if 'data_loaded' not in st.session_state:
    i, h, c = load_data()
    st.session_state.inventory = i; st.session_state.history = h; st.session_state.courses = c; st.session_state.data_loaded = True
if 'active_players' not in st.session_state: st.session_state.active_players = []
if 'current_scores' not in st.session_state: st.session_state.current_scores = {}
if 'selected_discs' not in st.session_state: st.session_state.selected_discs = {}
if 'daily_forms' not in st.session_state: st.session_state.daily_forms = {}
if 'suggested_pack' not in st.session_state: st.session_state.suggested_pack = []

# --- 5. LOGIK ---
def suggest_disc(bag, player, dist, shape, form=1.0):
    pb = bag[(bag["Owner"]==player) & (bag["Status"]=="Bag")]
    if pb.empty: return None, "Tom väska"
    
    eff_dist = dist / max(form, 0.5)
    target_speed = eff_dist / 10.0
    candidates = pb.copy()
    candidates["Speed_Diff"] = abs(candidates["Speed"] - target_speed)
    
    if eff_dist < 40: candidates = candidates[candidates["Typ"] == "Putter"]
    elif eff_dist < 75: candidates = candidates[candidates["Typ"].isin(["Putter", "Midrange"])]
    elif eff_dist < 100: candidates = candidates[candidates["Typ"].isin(["Midrange", "Fairway Driver"])]
    
    if candidates.empty: candidates = pb 
    
    if form < 0.9: candidates["Score"] = candidates["Speed_Diff"] + (candidates["Turn"] * 0.5)
    else: candidates["Score"] = candidates["Speed_Diff"]

    if shape == "Höger":
        best = candidates.sort_values(by=["Score", "Fade"], ascending=[True, False]).iloc[0]; advice = "Forehand"
    elif shape == "Vänster":
        best = candidates.sort_values(by=["Score", "Fade"], ascending=[True, False]).iloc[0]; advice = "Backhand Hyzer"
    else: 
        candidates["Neu"] = abs(candidates["Turn"] + candidates["Fade"])
        best = candidates.sort_values(by=["Score", "Neu"], ascending=[True, True]).iloc[0]; advice = "Backhand Rakt"
        
    return best, advice

def generate_smart_bag(inventory, player, course_name):
    holes = st.session_state.courses[course_name]["holes"]
    avg_len = np.mean([h["l"] for h in holes.values()])
    all_discs = inventory[inventory["Owner"] == player]
    pack_list = []
    
    putters = all_discs[all_discs["Typ"] == "Putter"].sort_values("Speed")
    if not putters.empty: pack_list.append(putters.iloc[0].name)
    mids = all_discs[all_discs["Typ"] == "Midrange"]
    if not mids.empty: pack_list.append(mids.sort_values("Glide", ascending=False).iloc[0].name)
    fairways = all_discs[all_discs["Typ"] == "Fairway Driver"]
    if not fairways.empty: pack_list.append(fairways.iloc[0].name)
    
    if avg_len > 80:
        drivers = all_discs[all_discs["Typ"] == "Distance Driver"]
        if not drivers.empty: pack_list.append(drivers.iloc[0].name)
    else:
        if len(putters) > 1: pack_list.append(putters.iloc[1].name)
    return list(set(pack_list))

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

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("🏎️ SCUDERIA")
    st.subheader("👥 Lobby")
    all_owners = st.session_state.inventory["Owner"].unique().tolist() if not st.session_state.inventory.empty else []
    
    new_p = st.text_input("Ny spelare:", placeholder="Namn")
    if st.button("Lägg till") and new_p:
        start_kit = [{"Owner": new_p, "Modell": "Start Putter", "Typ": "Putter", "Speed": 3, "Glide": 3, "Turn": 0, "Fade": 0, "Status": "Bag"}]
        st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame(start_kit)], ignore_index=True)
        save_data(st.session_state.inventory, st.session_state.history, st.session_state.courses); st.rerun()
    
    active = st.multiselect("Vilka spelar?", all_owners, default=st.session_state.active_players)
    if active != st.session_state.active_players:
        st.session_state.active_players = active
        for p in active:
            if p not in st.session_state.daily_forms: st.session_state.daily_forms[p] = 1.0
        st.rerun()
    if not st.session_state.active_players: st.warning("Välj spelare!"); st.stop()

# --- 7. TABS ---
t1, t2, t3, t4, t5, t6 = st.tabs(["🔥 WARM-UP", "🏁 RACE", "🧳 UTRUSTNING", "🥏 AERO-LAB", "📊 BI-STATS", "⚙️ ADMIN"])

# TAB 1: KALIBRERING
with t1:
    st.header("🔥 Pro Kalibrering")
    curr_p = st.selectbox("Vem?", st.session_state.active_players)
    p_bag = st.session_state.inventory[(st.session_state.inventory["Owner"]==curr_p) & (st.session_state.inventory["Status"]=="Bag")]
    if not p_bag.empty:
        d1 = st.selectbox("Disc", p_bag["Modell"].unique())
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown("##### Kast 1"); k1l = st.number_input("Längd", 0, key="k1"); k1s = st.slider("Sida", -20, 20, 0, key="s1"); k1h = st.radio("H1", ["Låg", "Normal", "Hög"], horizontal=True, label_visibility="collapsed")
        with c2: st.markdown("##### Kast 2"); k2l = st.number_input("Längd", 0, key="k2"); k2s = st.slider("Sida", -20, 20, 0, key="s2"); k2h = st.radio("H2", ["Låg", "Normal", "Hög"], horizontal=True, label_visibility="collapsed")
        with c3: st.markdown("##### Kast 3"); k3l = st.number_input("Längd", 0, key="k3"); k3s = st.slider("Sida", -20, 20, 0, key="s3"); k3h = st.radio("H3", ["Låg", "Normal", "Hög"], horizontal=True, label_visibility="collapsed")
        
        if st.button(f"Analysera {curr_p}"):
            valid = [k for k in [k1l, k2l, k3l] if k > 0]
            if valid:
                avg = sum(valid)/len(valid); form = avg/80.0; st.session_state.daily_forms[curr_p] = form
                tips = []
                if (k1s+k2s+k3s)/3 < -5: tips.append("Du drar Vänster: Släpper du för tidigt?")
                if [k1h,k2h,k3h].count("Hög") >= 2: tips.append("Nose Up: Tryck ner tummen!")
                c_res, c_gr = st.columns(2)
                with c_res: 
                    st.metric("Form", f"{int(form*100)}%")
                    if tips: [st.info(t) for t in tips]
                    else: st.success("Bra teknik!")
                with c_gr: fig, ax = plt.subplots(figsize=(3,3)); ax.scatter([k1s,k2s,k3s], [k1l,k2l,k3l]); ax.axvline(0, c='gray', ls='--'); ax.set_xlim(-25,25); ax.set_ylim(0, max(avg*1.2, 50)); st.pyplot(fig)
    else: st.warning("Tom väska.")

# TAB 2: RACE
with t2:
    courses = list(st.session_state.courses.keys()); bana = st.selectbox("Välj Bana", courses); c_data = st.session_state.courses[bana]
    with st.expander("🗺️ Bankarta", expanded=False): m = folium.Map([c_data["lat"], c_data["lon"]], zoom_start=16); st_folium(m, height=200)
    col_n, col_s = st.columns([1, 2])
    with col_n:
        holes = sorted(list(c_data["holes"].keys()), key=lambda x: int(x) if x.isdigit() else x); hole = st.selectbox("Hål", holes); inf = c_data["holes"][hole]
        st.metric(f"Hål {hole}", f"{inf['l']}m", f"Par {inf['p']}"); st.caption(f"**{inf.get('shape', 'Rak')}**"); st.info(inf.get('desc', 'Ingen info'))
        with st.popover("✏️ Redigera"):
            nd = st.text_input("Info", inf.get('desc','')); ns = st.selectbox("Form", ["Vänster","Rak","Höger"])
            if st.button("Spara"): st.session_state.courses[bana]["holes"][hole].update({"desc": nd, "shape": ns}); save_data(st.session_state.inventory, st.session_state.history, st.session_state.courses); st.rerun()
    with col_s:
        if hole not in st.session_state.current_scores: st.session_state.current_scores[hole] = {p: inf['p'] for p in st.session_state.active_players}
        if hole not in st.session_state.selected_discs: st.session_state.selected_discs[hole] = {p: None for p in st.session_state.active_players}
        
        p_tabs = st.tabs(st.session_state.active_players)
        for i, p in enumerate(st.session_state.active_players):
            with p_tabs[i]:
                col_mode, col_set = st.columns([1, 2])
                with col_mode: mode = st.radio("Läge", ["Tee", "Fairway"], key=f"mode_{p}", label_visibility="collapsed", horizontal=True)
                calc_dist = inf['l']; calc_shape = inf.get('shape', 'Rak')
                if mode == "Fairway":
                    with col_set:
                        calc_dist = st.slider("Avstånd kvar (m)", 10, 200, 50, key=f"d_sli_{p}")
                        calc_shape = st.selectbox("Vinkel", ["Vänster", "Rak", "Höger"], key=f"s_sel_{p}")

                curr_form = st.session_state.daily_forms.get(p, 1.0)
                rec, reason = suggest_disc(st.session_state.inventory, p, calc_dist, calc_shape, curr_form)
                if rec is not None: st.success(f"🤖 **Caddy:** {rec['Modell']} ({reason})"); st.caption(f"Stats: {rec['Speed']} | {rec['Glide']} | {rec['Turn']} | {rec['Fade']}")
                else: st.warning(reason)
                
                p_bag = st.session_state.inventory[(st.session_state.inventory["Owner"]==p) & (st.session_state.inventory["Status"]=="Bag")]
                opts = ["Välj Disc"] + p_bag["Modell"].tolist(); sel = st.selectbox(f"Disc ({p})", opts, key=f"d_{hole}_{p}")
                st.session_state.selected_discs[hole][p] = sel if sel != "Välj Disc" else None
                c_m, c_v, c_p = st.columns([1,2,1])
                curr = st.session_state.current_scores[hole][p]
                if c_m.button("➖", key=f"m_{hole}_{p}"): st.session_state.current_scores[hole][p] = max(1, curr-1); st.rerun()
                c_v.markdown(f"<h1 style='text-align:center; margin:0;'>{curr}</h1>", unsafe_allow_html=True)
                if c_p.button("➕", key=f"p_{hole}_{p}"): st.session_state.current_scores[hole][p] = curr+1; st.rerun()
    st.divider()
    if st.button("🏁 SPARA RUNDA", type="primary"):
        rows = []
        d = datetime.now().strftime("%Y-%m-%d")
        for h, scores in st.session_state.current_scores.items():
            for p, s in scores.items():
                disc = st.session_state.selected_discs[h].get(p, "Unknown")
                rows.append({"Datum": d, "Bana": bana, "Spelare": p, "Hål": h, "Resultat": s, "Par": c_data["holes"][h]["p"], "Disc_Used": disc})
        st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame(rows)], ignore_index=True)
        save_data(st.session_state.inventory, st.session_state.history, st.session_state.courses); st.success("Sparat!"); st.session_state.current_scores = {}; st.session_state.selected_discs = {}

# TAB 3: UTRUSTNING (MED VISION SETUP)
with t3:
    st.header("🧳 Utrustning")
    owner = st.selectbox("Väska:", st.session_state.active_players, key="bag_owner")
    
    # AI PACKNING
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        tc = c1.selectbox("Bana", list(st.session_state.courses.keys()))
        if c2.button("Generera"): st.session_state.suggested_pack = generate_smart_bag(st.session_state.inventory, owner, tc); st.rerun()
        if st.session_state.suggested_pack:
            if c3.button("Verkställ", type="primary"):
                st.session_state.inventory.loc[st.session_state.inventory["Owner"]==owner, "Status"] = "Shelf"
                st.session_state.inventory.loc[st.session_state.suggested_pack, "Status"] = "Bag"
                save_data(st.session_state.inventory, st.session_state.history, st.session_state.courses); st.session_state.suggested_pack = []; st.rerun()
            st.info(f"Förslag: {len(st.session_state.suggested_pack)} discar")

    # DISC LISTA
    my_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == owner]
    c_s, c_b = st.columns(2)
    with c_s:
        st.subheader("🏠 Hyllan")
        sh = my_inv[my_inv["Status"]=="Shelf"]
        for cat in ["Putter", "Midrange", "Fairway Driver", "Distance Driver"]:
            ss = sh[sh["Typ"]==cat].sort_values("Speed")
            if not ss.empty:
                with st.expander(f"{cat} ({len(ss)})"):
                    for i,r in ss.iterrows():
                        if st.button(f"➡️ {r['Modell']}", key=f"s_{i}"): st.session_state.inventory.at[i,"Status"]="Bag"; save_data(st.session_state.inventory, st.session_state.history, st.session_state.courses); st.rerun()
    with c_b:
        st.subheader("🎒 I Bagen")
        ba = my_inv[my_inv["Status"]=="Bag"]
        for cat in ["Putter", "Midrange", "Fairway Driver", "Distance Driver"]:
            ss = ba[ba["Typ"]==cat].sort_values("Speed")
            if not ss.empty:
                st.markdown(f"**{cat}**")
                for i,r in ss.iterrows():
                    c1,c2 = st.columns([4,1])
                    c1.markdown(f"{r['Modell']} ({int(r['Speed'])})")
                    if c2.button("⬅️", key=f"b_{i}"): st.session_state.inventory.at[i,"Status"]="Shelf"; save_data(st.session_state.inventory, st.session_state.history, st.session_state.courses); st.rerun()
    
    # LÄGG TILL (NU MED FOTO)
    with st.expander("➕ Lägg till ny disc (AI Vision)"):
        tab_man, tab_ai = st.tabs(["Manuell", "Fota Disc 📸"])
        
        with tab_man:
            with st.form("nd"):
                mn = st.text_input("Modell"); ty = st.selectbox("Typ", ["Putter", "Midrange", "Fairway Driver", "Distance Driver"])
                c1,c2,c3,c4 = st.columns(4); sp=c1.number_input("Speed",1.,14.,7.); gl=c2.number_input("Glide",1.,7.,5.); tu=c3.number_input("Turn",-5.,1.,0.); fa=c4.number_input("Fade",0.,5.,2.)
                if st.form_submit_button("Spara"): 
                    nw = {"Owner": owner, "Modell": mn, "Typ": ty, "Speed": sp, "Glide": gl, "Turn": tu, "Fade": fa, "Status": "Shelf"}
                    st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame([nw])], ignore_index=True)
                    save_data(st.session_state.inventory, st.session_state.history, st.session_state.courses); st.rerun()
        
        with tab_ai:
            img = st.camera_input("Ta en bild på discen")
            if img:
                st.image(img)
                st.info("🤖 AI-modulen är redo för molnet! (När vi kopplar på GPT-4 Vision i nästa steg kommer den läsa av: Modell, Speed, Glide etc. automatiskt här).")

# TAB 4: AERO-LAB
with t4:
    st.header("🥏 Aero-Lab"); pwr = st.slider("Kraft", 30, 150, 80); sty = st.radio("Stil", ["Backhand", "Forehand"])
    bd = st.session_state.inventory[st.session_state.inventory["Status"]=="Bag"]
    if not bd.empty:
        sel = st.multiselect("Jämför:", (bd["Modell"]+" ("+bd["Owner"]+")").unique())
        if sel:
            fig, ax = plt.subplots(figsize=(8, 10)); is_dark = st.get_option("theme.base") == "dark"; bg='#1e1e1e' if is_dark else 'white'; fg='white' if is_dark else 'black'; fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
            ax.set_xlim(-50, 50); ax.set_ylim(0, pwr * 1.3); ax.set_aspect('equal'); ax.plot(0, 0, 'ws', markersize=10); ax.axvline(0, color='gray', linestyle='--')
            for s in sel:
                mod, own = s.rsplit(" (", 1); own = own[:-1]; d_row = bd[(bd["Modell"]==mod)&(bd["Owner"]==own)].iloc[0]
                x, y = simulate_flight_path_hd(d_row['Speed'], d_row['Glide'], d_row['Turn'], d_row['Fade'], pwr, sty)
                ax.plot(x, y, linewidth=2.5, label=s)
            ax.legend(); ax.grid(True, alpha=0.1); ax.spines['bottom'].set_color(fg); ax.spines['left'].set_color(fg); ax.tick_params(colors=fg); st.pyplot(fig)
    else: st.warning("Tomma bagar.")

# TAB 5: BI-STATS
with t5:
    st.header("📊 Statistik Center")
    df = st.session_state.history
    if not df.empty:
        c1, c2 = st.columns(2)
        sel_p = c1.multiselect("Spelare", df["Spelare"].unique(), default=df["Spelare"].unique())
        sel_c = c2.multiselect("Bana", df["Bana"].unique(), default=df["Bana"].unique())
        dff = df[(df["Spelare"].isin(sel_p)) & (df["Bana"].isin(sel_c))]
        
        if not dff.empty:
            cols = st.columns(len(sel_p))
            for idx, p in enumerate(sel_p):
                with cols[idx]:
                    p_data = dff[dff["Spelare"]==p]
                    if not p_data.empty:
                        avg = p_data["Resultat"].mean()
                        birdie = len(p_data[p_data["Resultat"] < p_data["Par"]])
                        tot = len(p_data)
                        st.markdown(f"<div class='stat-card'><h3>{p}</h3>"
                                    f"<b>Snitt:</b> {avg:.1f}<br>"
                                    f"<b>Birdies:</b> {int(birdie/tot*100)}%</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            g1, g2 = st.columns(2)
            with g1:
                st.subheader("📈 Trend")
                trend = dff.groupby(["Datum", "Spelare"])["Resultat"].mean().reset_index()
                st.line_chart(trend, x="Datum", y="Resultat", color="Spelare")
            
            with g2:
                st.subheader("🥏 Bästa Discarna")
                clean = dff[~dff["Disc_Used"].isin(["Ingen vald", "Unknown"])]
                if not clean.empty:
                    st.bar_chart(clean.groupby("Disc_Used")["Resultat"].mean().sort_values().head(7))
                else: st.info("Ingen disc-data.")

            st.subheader("⛳ Hål-Analys (Jämförelse)")
            try:
                dff['Hål_Int'] = dff['Hål'].astype(int)
                c = alt.Chart(dff).mark_bar().encode(
                    x=alt.X('Hål_Int:O', title='Hål'),
                    y=alt.Y('mean(Resultat):Q', title='Snittscore'),
                    color='Spelare:N',
                    xOffset='Spelare:N',
                    tooltip=['Spelare', 'Hål_Int', alt.Tooltip('mean(Resultat)', format='.2f')]
                ).interactive()
                st.altair_chart(c, use_container_width=True)
            except Exception as e:
                st.error(f"Graf-fel: {e}")
                st.bar_chart(dff.groupby("Hål")["Resultat"].mean())

        else: st.warning("Inget data med detta filter.")
    else: st.info("Ingen historik.")

# TAB 6: ADMIN
with t6:
    st.subheader("⚙️ System")
    st.markdown("### 📥 Importera UDisc")
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
                st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame(nd)], ignore_index=True)
                save_data(st.session_state.inventory, st.session_state.history, st.session_state.courses); st.success(f"Importerade {len(nd)} hål!")
        except Exception as e: st.error(f"Importfel: {e}")

    st.divider()
    if st.button("♻️ Hämta gammal bag"):
        if os.path.exists("min_bag.csv"): 
            st.session_state.inventory = pd.concat([st.session_state.inventory, pd.read_csv("min_bag.csv").assign(Owner="Mattias", Status="Bag")], ignore_index=True)
            save_data(st.session_state.inventory, st.session_state.history, st.session_state.courses); st.success("Klart!")
    
    if st.button("🗑️ Reset All"): [os.remove(f) for f in FILES.values() if os.path.exists(f)]; st.warning("Klart.")
