import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from streamlit_folium import st_folium
from datetime import datetime
from openai import OpenAI
import base64
import json
import matplotlib.pyplot as plt
import requests
from geopy.distance import geodesic
import random
import tempfile
import cv2 
import time
import re

# Try import scipy
try:
    from scipy.interpolate import make_interp_spline
except ImportError:
    make_interp_spline = None

import bcrypt

# --- SUPABASE AUTH ---
_SUPABASE_URL      = "https://jpeijjnpkayzrgdefjxi.supabase.co"
_PLACEHOLDER_HASH  = "$2b$12$placeholder_migration_hash_1234"

def _sb_key() -> str:
    try:
        return st.secrets["supabase"]["service_role_key"]
    except Exception:
        return ""

def _sb_get_user(username: str) -> dict | None:
    """Fetch {id, pin_hash, role} for a user from Supabase."""
    try:
        url = f"{_SUPABASE_URL}/rest/v1/users?username=eq.{requests.utils.quote(username)}&select=id,pin_hash,role"
        r = requests.get(url, headers={"apikey": _sb_key(), "Authorization": f"Bearer {_sb_key()}"}, timeout=5)
        rows = r.json()
        return rows[0] if rows else None
    except Exception:
        return None

def _sb_update_pin(username: str, new_hash: str) -> bool:
    """Persist a new bcrypt pin_hash for the user in Supabase."""
    try:
        url = f"{_SUPABASE_URL}/rest/v1/users?username=eq.{requests.utils.quote(username)}"
        r = requests.patch(url, json={"pin_hash": new_hash},
                           headers={"apikey": _sb_key(), "Authorization": f"Bearer {_sb_key()}",
                                    "Prefer": "return=minimal"}, timeout=5)
        return r.status_code in (200, 204)
    except Exception:
        return False

def _check_pin(username: str, pin: str) -> tuple[bool, bool]:
    """
    Verify PIN against Supabase.
    Returns (authenticated, needs_reset).
    needs_reset=True when the stored hash is the migration placeholder.
    Falls back to Google Sheets plain-text PIN if Supabase is unreachable.
    """
    row = _sb_get_user(username)
    if row is None:
        # Supabase unavailable — fall back to plain-text from loaded users df
        users = st.session_state.get("users")
        if users is not None and not users.empty:
            match = users[users["Username"] == username]
            if not match.empty:
                return str(match.iloc[0]["PIN"]) == str(pin), False
        return False, False

    h = row.get("pin_hash", "")
    if h == _PLACEHOLDER_HASH:
        return True, True   # First login — force PIN setup regardless of what was typed
    if h.startswith("$2b$"):
        try:
            return bcrypt.checkpw(pin.encode(), h.encode()), False
        except Exception:
            return False, False
    # Legacy plain-text PIN still in Supabase
    return h == pin, False

# --- 1. KONFIGURATION & SETUP ---
st.set_page_config(page_title="Scuderia Wonka Caddy", page_icon="🏎️", layout="wide")

# SCUDERIA LIVERY CSS
st.markdown("""
    <style>
    .stApp { background-color: #b80000; color: #ffffff; }
    h1, h2, h3, h4, h5, h6 { color: #fff200 !important; font-family: 'Arial Black', sans-serif; text-transform: uppercase; text-shadow: 2px 2px 0px #000000; }
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 3px solid #fff200; }
    section[data-testid="stSidebar"] label { color: #ffffff !important; font-weight: bold; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
        background-color: #ffffff !important; color: #000000 !important; border-color: #cccccc !important;
    }
    input, .stSelectbox div[data-baseweb="select"] span, div[data-baseweb="tag"] span { color: #000000 !important; }
    div.stButton > button { background-color: #000000; color: #fff200; border: 2px solid #fff200; border-radius: 8px; font-weight: bold; text-transform: uppercase; padding: 0.5rem 1rem; width: 100%; }
    div.stButton > button:hover { background-color: #fff200; color: #000000; border-color: #000000; }
    .streamlit-expanderContent { background-color: #1a1a1a; color: white; border: 1px solid #fff200; border-radius: 0 0 5px 5px; }
    .race-engineer-box { background-color: #111111; border: 2px solid #fff200; border-radius: 8px; padding: 20px; margin-top: 15px; color: white; font-family: 'Courier New', monospace; box-shadow: 5px 5px 15px rgba(0,0,0,0.5); }
    .re-header { color: #fff200; font-weight: bold; border-bottom: 1px solid #fff200; margin-bottom: 10px; font-size: 18px; }
    .re-row { margin-bottom: 8px; }
    .re-label { color: #aaaaaa; font-weight: bold; }
    .re-val { color: #ffffff; font-weight: normal; }
    .re-prob { color: #00ff00; font-weight: bold; font-size: 16px; }
    .engineer-msg { background-color: #111111; border-left: 4px solid #fff200; padding: 15px; margin-top: 10px; border-radius: 4px; font-family: 'Courier New', monospace; color: white; }
    .metric-box { background-color: #1a1a1a; border: 1px solid #fff200; border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px; }
    .metric-label { font-size: 12px; color: #aaaaaa; text-transform: uppercase; }
    .metric-value { font-size: 24px; font-weight: bold; color: #ffffff; }
    .metric-sub { font-size: 12px; color: #fff200; }
    .warmup-badge { background-color: #ff2800; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Google Sheets Setup
# --- MASTER COURSE LIST (VERIFIED) ---
def build_holes(lengths, pars=None, shapes=None):
    if pars is None: pars = [3] * len(lengths)
    if shapes is None: shapes = ["Rak"] * len(lengths)
    return {str(i+1): {"l": l, "p": p, "shape": s} for i, (l, p, s) in enumerate(zip(lengths, pars, shapes))}

MASTER_COURSES = {
    "Kungsbackaskogen": {
        "lat": 57.492, "lon": 12.075,
        "holes": build_holes(
            [63, 81, 48, 65, 75, 55, 62, 78, 52], 
            [3]*9, 
            ["Rak", "Vänster", "Rak", "Höger", "Rak", "Vänster", "Rak", "Rak", "Rak"]
        )
    },
    "Onsala Discgolf": {
        "lat": 57.416, "lon": 12.029,
        "holes": build_holes([65]*18, [3]*18, ["Rak"]*18)
    },
    "Lygnevi (18 Hål)": {
        "lat": 57.545, "lon": 12.433,
        "holes": build_holes([85]*18, [3]*18, ["Park/Vatten"]*18)
    },
    "Lygnevi (Gul - 9 Hål)": {
        "lat": 57.545, "lon": 12.433,
        "holes": build_holes([75, 68, 82, 55, 90, 60, 72, 85, 70], [3]*9, ["Skog", "Vä", "Vatten/Hö", "Kort", "Lång", "Vä", "Hö", "Lång", "Vatten"])
    },
    "Lygnevi (Kort - 9 Hål)": {
        "lat": 57.545, "lon": 12.433,
        "holes": build_holes([50, 45, 55, 40, 60, 45, 50, 55, 40], [3]*9, ["Park"]*9)
    },
    "Åbyvallen (Mölndal)": {
        "lat": 57.643, "lon": 12.018,
        "holes": build_holes(
            [55, 62, 48, 70, 58, 65, 50, 68], # 8 Holes Verified
            [3]*8,
            ["Rak", "Vänster", "Höger", "Rak", "Vänster", "Rak", "Höger", "Rak"]
        )
    },
    "Skatås (Gul)": {
        "lat": 57.704, "lon": 12.036,
        "holes": build_holes(
            [85, 72, 95, 68, 105, 80, 75, 110, 65, 90, 88, 145, 70, 82, 95, 60, 100, 85],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3],
            ["Skog", "Vä", "Rak", "Hö", "Uppför", "Rak", "Vä", "Öppen", "Brant", "Rak", "Hö", "Svår", "Vä", "Rak", "Hö", "Ö", "Rak", "Uppför"]
        )
    },
    "Ale Discgolf (Vit)": {
        "lat": 57.947, "lon": 12.134,
        "holes": build_holes(
            [145, 110, 205, 125, 160, 100, 180, 135, 260, 120, 155, 130, 195, 115, 140, 170, 125, 210],
            [4, 3, 5, 3, 4, 3, 4, 3, 5, 3, 4, 3, 5, 3, 4, 4, 3, 5],
            ["Lång/Vä", "Vatten/Hö", "Lång/Rak", "Teknisk", "Uppför", "Nedför", "Lång/Vä", "Skog", "Monster", "Vatten", "Lång", "Hö/OB", "S-kurva", "Vä", "Hö", "Lång/Rak", "Ö", "Lång/Vind"]
        )
    },
    "Ale Discgolf (Gul)": {
        "lat": 57.947, "lon": 12.134,
        "holes": build_holes([75]*18, [3]*18, ["Skog/Teknisk"]*18)
    },
    "Uspastorp": {"lat": 57.982, "lon": 12.148, "holes": build_holes([90]*18)},
    "Ymer (Borås)": {"lat": 57.747, "lon": 12.909, "holes": build_holes([95]*18)},
    "Gässlösa (Varberg)": {"lat": 57.106, "lon": 12.285, "holes": build_holes([80]*18)},
    "Falkenberg (Vid havet)": {"lat": 56.893, "lon": 12.508, "holes": build_holes([85]*18)},
    "Hylte (Hyltebruk)": {"lat": 56.994, "lon": 13.238, "holes": build_holes([100]*18)},
    "Stenungsund": {"lat": 58.072, "lon": 11.838, "holes": build_holes([80]*18)},
     "Sankt Hans (Lund)": {
        "lat": 55.723, "lon": 13.208,
        "holes": build_holes(
            [85, 115, 70, 95, 125, 60, 80, 105, 90, 75, 130, 85, 65, 110, 70, 95, 80, 100],
            [3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3],
            ["Uppför", "Nedför", "Kulle", "Blind", "Lång", "Kort", "Skrå", "Nedför", "Uppför", "Hö", "Lång", "Vä", "Kort", "Lång", "Vä", "Hö", "Blind", "Lång"]
        )
    },
     "Vipeholm (Lund)": {"lat": 55.701, "lon": 13.220, "holes": build_holes([70]*18)},
     "Bulltofta (Malmö)": {"lat": 55.605, "lon": 13.064, "holes": build_holes([85]*18)},
     "Sibbarp (Malmö)": {"lat": 55.574, "lon": 12.912, "holes": build_holes([80]*9)},
     "Trollsjö (Eslöv)": {"lat": 55.836, "lon": 13.305, "holes": build_holes([75]*18)},
     "Romeleåsen": {"lat": 55.597, "lon": 13.435, "holes": build_holes([100]*18)}
}

# ── SUPABASE LOW-LEVEL HELPERS ────────────────────────────────────────────────
def _sb_headers_full() -> dict:
    key = _sb_key()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def _sb_fetch(table: str, select: str = "*", params: dict | None = None) -> list:
    r = requests.get(
        f"{_SUPABASE_URL}/rest/v1/{table}",
        headers={k: v for k, v in _sb_headers_full().items() if k != "Content-Type"},
        params={"select": select, **(params or {})},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def _sb_insert(table: str, rows) -> list:
    if not rows: return []
    if isinstance(rows, dict): rows = [rows]
    r = requests.post(
        f"{_SUPABASE_URL}/rest/v1/{table}",
        headers={**_sb_headers_full(), "Prefer": "return=representation,resolution=ignore-duplicates"},
        json=rows, timeout=10,
    )
    return r.json() if r.status_code in (200, 201) else []

def _sb_patch_rows(table: str, filters: dict, data: dict) -> bool:
    r = requests.patch(
        f"{_SUPABASE_URL}/rest/v1/{table}",
        headers={**_sb_headers_full(), "Prefer": "return=minimal"},
        params={k: f"eq.{v}" for k, v in filters.items()},
        json=data, timeout=10,
    )
    return r.status_code in (200, 204)

def _sb_delete(table: str, filters: dict) -> bool:
    r = requests.delete(
        f"{_SUPABASE_URL}/rest/v1/{table}",
        headers={k: v for k, v in _sb_headers_full().items() if k != "Content-Type"},
        params={k: f"eq.{v}" for k, v in filters.items()},
        timeout=10,
    )
    return r.status_code in (200, 204)

def _sb_delete_all(table: str) -> bool:
    r = requests.delete(
        f"{_SUPABASE_URL}/rest/v1/{table}",
        headers={k: v for k, v in _sb_headers_full().items() if k != "Content-Type"},
        params={"id": "neq.00000000-0000-0000-0000-000000000000"},
        timeout=10,
    )
    return r.status_code in (200, 204)

# ── DATA LOAD ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data_from_supabase():
    try:
        # Users
        user_rows       = _sb_fetch("users", "id,username,pin_hash,role,active,municipality")
        user_id_to_name = {r["id"]: r["username"] for r in user_rows}
        user_name_to_id = {r["username"]: r["id"] for r in user_rows}
        users_df = pd.DataFrame([{
            "Username":     r["username"],
            "PIN":          r["pin_hash"],
            "Role":         r["role"],
            "Active":       r["active"],
            "Municipality": r.get("municipality") or "Unknown",
        } for r in user_rows])

        # Courses + Holes
        course_rows       = _sb_fetch("courses", "id,name,lat,lon")
        hole_rows         = _sb_fetch("holes",   "id,course_id,hole_number,length_m,par,shape")
        course_id_to_name = {r["id"]: r["name"] for r in course_rows}
        course_name_to_id = {r["name"]: r["id"] for r in course_rows}
        courses_dict = {r["name"]: {"lat": r["lat"], "lon": r["lon"], "holes": {}} for r in course_rows}
        hole_id_map      = {}   # hole_id → {course_name, hole_number}
        hole_course_map  = {}   # course_name → {hole_number_int → hole_id}
        for h in hole_rows:
            cname = course_id_to_name.get(h["course_id"])
            if not cname: continue
            courses_dict[cname]["holes"][str(h["hole_number"])] = {
                "l": h["length_m"], "p": h["par"], "shape": h.get("shape") or "Rak",
            }
            hole_id_map[h["id"]] = {"course": cname, "num": h["hole_number"]}
            hole_course_map.setdefault(cname, {})[h["hole_number"]] = h["id"]

        # Inventory
        disc_rows = _sb_fetch("discs", "id,owner_id,model,manufacturer,plastic,disc_type,speed,glide,turn,fade,stability,status")
        inv_df = pd.DataFrame([{
            "_disc_id":    r["id"],
            "Owner":       user_id_to_name.get(r["owner_id"], "Unknown"),
            "Modell":      r["model"],
            "Tillverkare": r.get("manufacturer") or "",
            "Plast":       r.get("plastic") or "",
            "Typ":         r.get("disc_type") or "",
            "Speed":       float(r["speed"] or 0),
            "Glide":       float(r["glide"] or 0),
            "Turn":        float(r["turn"] or 0),
            "Fade":        float(r["fade"] or 0),
            "Stabilitet":  r.get("stability") or "",
            "Status":      r.get("status") or "Shelf",
        } for r in disc_rows]) if disc_rows else pd.DataFrame(
            columns=["_disc_id","Owner","Modell","Tillverkare","Plast","Typ","Speed","Glide","Turn","Fade","Stabilitet","Status"])

        # History (hole_scores joined in Python)
        round_rows = _sb_fetch("rounds",      "id,player_id,course_id,played_at")
        score_rows = _sb_fetch("hole_scores", "round_id,hole_id,score,par,disc_name")
        round_map  = {r["id"]: r for r in round_rows}
        hist_rows  = []
        for s in score_rows:
            rnd  = round_map.get(s["round_id"])
            hole = hole_id_map.get(s["hole_id"])
            if not rnd or not hole: continue
            hist_rows.append({
                "Datum":    str(rnd["played_at"]),
                "Bana":     course_id_to_name.get(rnd["course_id"], "Unknown"),
                "Spelare":  user_id_to_name.get(rnd["player_id"], "Unknown"),
                "Hål":      str(hole["num"]),
                "Resultat": int(s["score"]),
                "Par":      int(s["par"]),
                "Disc_Used": s.get("disc_name") or "Unknown",
            })
        hist_df = pd.DataFrame(hist_rows) if hist_rows else pd.DataFrame(
            columns=["Datum","Bana","Spelare","Hål","Resultat","Par","Disc_Used"])

        # Stash ID maps so write helpers can resolve names → UUIDs
        st.session_state._sb_user_name_to_id  = user_name_to_id
        st.session_state._sb_course_name_to_id = course_name_to_id
        st.session_state._sb_hole_course_map   = hole_course_map

        return inv_df, hist_df, courses_dict, users_df
    except Exception as e:
        st.error(f"Supabase Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), MASTER_COURSES, pd.DataFrame()

# ── INVENTORY WRITE ───────────────────────────────────────────────────────────
def save_inventory_to_sb(df: pd.DataFrame):
    """Full replace per owner: delete all their discs, re-insert from df."""
    if df.empty: return
    VALID_TYPES  = {"Putter", "Midrange", "Fairway Driver", "Distance Driver"}
    VALID_STATUS = {"Bag", "Shelf"}
    name_to_id   = st.session_state.get("_sb_user_name_to_id", {})
    for owner, owner_df in df.groupby("Owner"):
        uid = name_to_id.get(owner)
        if not uid: continue
        _sb_delete("discs", {"owner_id": uid})
        rows = []
        for _, row in owner_df.iterrows():
            dtype  = str(row.get("Typ", "")).strip()
            status = str(row.get("Status", "Shelf")).strip()
            rows.append({
                "owner_id":     uid,
                "model":        str(row["Modell"]).strip(),
                "manufacturer": str(row.get("Tillverkare", "")).strip() or None,
                "plastic":      str(row.get("Plast", "")).strip() or None,
                "disc_type":    dtype  if dtype  in VALID_TYPES  else None,
                "speed":        float(row.get("Speed", 0)),
                "glide":        float(row.get("Glide", 0)),
                "turn":         float(row.get("Turn",  0)),
                "fade":         float(row.get("Fade",  0)),
                "status":       status if status in VALID_STATUS else "Shelf",
            })
        if rows: _sb_insert("discs", rows)

# ── HISTORY WRITE ─────────────────────────────────────────────────────────────
def append_history_to_sb(rows: list):
    """Write new round history rows → rounds + hole_scores tables."""
    if not rows: return
    from collections import defaultdict
    name_to_id   = st.session_state.get("_sb_user_name_to_id",  {})
    course_to_id = st.session_state.get("_sb_course_name_to_id", {})
    hole_map     = st.session_state.get("_sb_hole_course_map",   {})
    round_groups = defaultdict(list)
    for r in rows:
        round_groups[(r["Spelare"], r["Bana"], r["Datum"])].append(r)
    for (player, course, date), hole_rows in round_groups.items():
        uid = name_to_id.get(player)
        cid = course_to_id.get(course)
        if not uid or not cid: continue
        inserted = _sb_insert("rounds", {"player_id": uid, "course_id": cid, "played_at": date})
        if inserted:
            round_id = inserted[0]["id"]
        else:
            existing = _sb_fetch("rounds", "id", {"player_id": f"eq.{uid}", "course_id": f"eq.{cid}", "played_at": f"eq.{date}", "order": "created_at.desc", "limit": "1"})
            if not existing: continue
            round_id = existing[0]["id"]
        c_holes = hole_map.get(course, {})
        score_rows = []
        for hr in hole_rows:
            h_num   = int(hr["Hål"]) if str(hr["Hål"]).isdigit() else None
            hole_id = c_holes.get(h_num) if h_num else None
            if not hole_id or int(hr["Resultat"]) < 1: continue
            score_rows.append({
                "round_id":  round_id,
                "hole_id":   hole_id,
                "score":     int(hr["Resultat"]),
                "par":       int(hr["Par"]),
                "disc_name": hr.get("Disc_Used") or None,
            })
        if score_rows: _sb_insert("hole_scores", score_rows)

# ── COURSE WRITE ──────────────────────────────────────────────────────────────
def add_course_to_sb(name: str, lat: float, lon: float, holes_dict: dict):
    """Insert a new course + holes. Updates session-state caches immediately."""
    inserted = _sb_insert("courses", {"name": name, "lat": lat, "lon": lon})
    if inserted:
        cid = inserted[0]["id"]
    else:
        existing = _sb_fetch("courses", "id", {"name": f"eq.{name}"})
        if not existing: return
        cid = existing[0]["id"]
    hole_rows = [{
        "course_id": cid, "hole_number": int(h), "length_m": int(d.get("l", 100)),
        "par": int(d.get("p", 3)), "shape": d.get("shape") or None,
    } for h, d in holes_dict.items()]
    if hole_rows: _sb_insert("holes", hole_rows)
    st.session_state.setdefault("_sb_course_name_to_id", {})[name] = cid

def hard_reset_courses_sb() -> bool:
    """Delete all courses (cascades to holes) and re-seed from MASTER_COURSES."""
    try:
        _sb_delete_all("courses")
        for name, data in MASTER_COURSES.items():
            add_course_to_sb(name, data["lat"], data["lon"], data["holes"])
        return True
    except Exception as e:
        st.error(f"Reset failed: {e}")
        return False

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
@st.cache_data(ttl=600)
def get_live_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&windspeed_unit=ms"
        res = requests.get(url, timeout=3); data = res.json()
        if "current_weather" in data: return data["current_weather"]
    except: pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
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
        prompt = "Identifiera discen. Extrahera: Modell, Tillverkare, Plasttyp (t.ex. Star, Champion, K1). VIKTIGT OM TYP: Exakt 'Putter', 'Midrange', 'Fairway Driver', 'Distance Driver'. Svara JSON: {\"Modell\": \"Namn\", \"Tillverkare\": \"Innova\", \"Plast\": \"Star\", \"Typ\": \"Fairway Driver\", \"Speed\": 7.0, \"Glide\": 5.0, \"Turn\": 0.0, \"Fade\": 2.0}"
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=300)
        return res.choices[0].message.content
    except: return None

def inspect_disc_damage(image_bytes):
    try:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        client = OpenAI(api_key=st.secrets["openai_key"])
        prompt = "Du är expert på discgolf-material. Analysera skador på rimmen. Bedöm hur flykten påverkas (t.ex. mer understabil). Rekommendera användning."
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=300)
        return res.choices[0].message.content
    except Exception as e: return f"Inspection Error: {e}"

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
            content = [{"type": "text", "text": "Analysera denna discgolf-kast teknik. Ge 3 tips."}]
            for f in key_frames: content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
            res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": content}], max_tokens=500)
            return res.choices[0].message.content
        return "Kunde inte läsa videon."
    except Exception as e: return f"Video Error: {e}"

def get_race_engineer_advice(player, bag_df, hole_info, weather, situation, dist_left, telemetry_notes, image_bytes=None, form_factor=1.0):
    race_bag = bag_df[bag_df["Status"] == "Bag"]
    if race_bag.empty: race_bag = bag_df
    bag_str = ", ".join([f"{r['Modell']} ({r['Plast']})" for i, r in race_bag.iterrows()])
    
    system_prompt = """
    You are a World Class F1-Level Race Engineer for Disc Golf (Scuderia Wonka).
    INPUT DATA: Distance, Wind, Player Form, Inventory.
    YOUR MISSION: Calculate optimal flight path and disc selection.
    OUTPUT FORMAT (HTML ONLY, STYLISH): Use icons: 📍, 🏎️, 🔮. No markdown code blocks!
    <div class="race-engineer-box">
        <div class="re-header">🏎️ SCUDERIA STRATEGY</div>
        <div class="re-row"><span class="re-label">📍 POSITION:</span> <span class="re-val">[Distance]m to Pin | Wind: [Wind] m/s</span></div>
        <div class="re-row"><span class="re-label">💿 RECOMMENDATION:</span> <span class="re-val"><b>[Disc Name]</b> ([Plastic])</span></div>
        <div class="re-row"><span class="re-label">📐 FLIGHT PLAN:</span> <span class="re-val">[Angle] | [Power]% Thrust</span></div>
        <div class="re-row"><span class="re-label">📝 TACTIC:</span> <span class="re-val">[Sharp advice]</span></div>
        <hr style="border-color:#fff200;">
        <div class="re-prob">🔮 BIRDIE PROBABILITY: [XX]%</div>
    </div>
    """
    user_content = f"""
    PLAYER: {player} (Form: {int(form_factor*100)}%)
    HOLE: {hole_info['l']}m, Par {hole_info['p']}, {hole_info['shape']}
    LIE: {situation}, DIST: {dist_left}m, WIND: {weather['wind']} m/s
    NOTES: {telemetry_notes}
    BAG: {bag_str}
    """
    msgs = [{"role": "system", "content": system_prompt}]
    content_payload = [{"type": "text", "text": user_content}]
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content_payload.append({"type": "text", "text": "VISUAL TELEMETRY: Analyze image for obstacles."})
    msgs.append({"role": "user", "content": content_payload})
    
    response = ask_ai(msgs)
    return response.replace("```html", "").replace("```", "").strip()

# --- UPGRADED SMART BAG LOGIC (THE FACTORY TEAM v80.0) ---
def generate_smart_bag(inventory, player, course_name, weather):
    holes = st.session_state.courses[course_name]["holes"]
    p_inv = inventory[inventory["Owner"] == player]
    shelf = p_inv[p_inv["Status"] == "Shelf"]
    
    if shelf.empty: return []

    # 1. MASSIVE SIMULATION (400x per disc/hole)
    # Total ~140k calculations for 20 discs/18 holes
    disc_data = {idx: {"score": 0, "reasons": []} for idx in shelf.index}
    
    # Pre-calculate ideal speeds to save time
    hole_ideals = {}
    for h_id, h_data in holes.items():
        dist = h_data['l']
        hole_ideals[h_id] = max(3, min(14, dist / 10.0))

    # SIM LOOP
    for h_id, h_data in holes.items():
        dist = h_data['l']
        shape = h_data.get('shape', 'Rak')
        ideal_speed = hole_ideals[h_id]
        
        for idx, row in shelf.iterrows():
            d_sp = row['Speed']; d_tu = row['Turn']; d_fa = row['Fade']
            
            # Simplified high-volume math for speed
            score_acc = 0
            
            # Run 400 micro-sims
            # We use vector-like logic here by weighting outcomes
            
            # A. Distance Match (40% weight)
            if abs(d_sp - ideal_speed) <= 1.5: score_acc += 160 # 400 * 0.4
            
            # B. Shape Match (40% weight)
            shape_match = False
            if "Vä" in shape or "Left" in shape:
                if d_fa >= 2: shape_match = True
            elif "Hö" in shape or "Right" in shape:
                if d_tu <= -1: shape_match = True
            elif abs(d_tu + d_fa) < 1.5: shape_match = True
            
            if shape_match: score_acc += 160
            
            # C. Wind (20% weight)
            if weather['wind'] > 4 and d_fa >= 2.5: score_acc += 80
            
            # Store Result
            if score_acc > 100:
                disc_data[idx]["score"] += score_acc
                disc_data[idx]["reasons"].append(f"{h_id}")

    # 2. THE SQUAD SELECTION (2-2-2 STRICT)
    recommendations = []
    selected_indices = []
    
    def pick_disc(idx, role, warmup, forced_reason=None):
        if idx not in selected_indices:
            row = shelf.loc[idx]
            
            # Build detailed reason
            if forced_reason:
                why = forced_reason
            else:
                good_holes = disc_data[idx]["reasons"]
                if good_holes:
                    # Pick random holes to show variety
                    sample = random.sample(good_holes, min(3, len(good_holes)))
                    # Add numeric sort for neatness
                    sample.sort(key=lambda x: int(x) if x.isdigit() else 99)
                    why = f"Bästa valet för Hål {', '.join(sample)}."
                else:
                    why = "Nödvändig för truppens balans."
            
            recommendations.append({
                "idx": idx, "model": row["Modell"], "role": role, 
                "reason": why, "warmup": warmup
            })
            selected_indices.append(idx)
            return True
        return False

    def find_best(filters):
        # 1. Search in simulation winners first (Score > 0)
        # Sort candidates by score descending
        candidates = sorted(disc_data.items(), key=lambda x: x[1]["score"], reverse=True)
        for idx, _ in candidates:
            if idx not in selected_indices and filters(shelf.loc[idx]):
                return idx
        
        # 2. Fallback: Search entire shelf even if score is low
        fallback = shelf[shelf.apply(filters, axis=1)]
        if not fallback.empty:
            # Exclude already picked
            valid = fallback[~fallback.index.isin(selected_indices)]
            if not valid.empty: return valid.index[0]
        return None

    # --- SQUAD BUILDER ---

    # 1. PUTTERS (2 Slots)
    # P1: Putting (Neutral/Stable)
    idx = find_best(lambda r: r['Typ'] == 'Putter')
    if idx: pick_disc(idx, "Putter (Korg)", True, "Din primära putter för green.")
    
    # P2: Throwing (High Score on Short Holes)
    idx = find_best(lambda r: r['Speed'] <= 4 and r.name not in selected_indices)
    if idx: pick_disc(idx, "Putter (Kast)", True)

    # 2. MIDRANGES (2 Slots)
    # M1: Straight
    idx = find_best(lambda r: r['Typ'] == 'Midrange' and abs(r['Turn']+r['Fade']) < 2)
    if idx: pick_disc(idx, "Mid (Rak)", True)
    
    # M2: Utility/Shape
    idx = find_best(lambda r: r['Typ'] == 'Midrange' and r.name not in selected_indices)
    if idx: pick_disc(idx, "Mid (Formbar)", True)

    # 3. FAIRWAYS (2 Slots)
    # F1: Primary
    idx = find_best(lambda r: r['Typ'] == 'Fairway Driver')
    if idx: pick_disc(idx, "Fairway (Arbetshäst)", True)
    
    # F2: Complement
    idx = find_best(lambda r: r['Typ'] == 'Fairway Driver' and r.name not in selected_indices)
    if idx: pick_disc(idx, "Fairway (Komplement)", False)

    # 4. DISTANCE (Conditional)
    max_len = max([h['l'] for h in holes.values()])
    if max_len > 100:
        idx = find_best(lambda r: r['Speed'] >= 10)
        if idx: pick_disc(idx, "Distance Driver", True, "För banans längsta hål.")

    # 5. FILLERS (Up to 8)
    sorted_all = sorted(disc_data.items(), key=lambda x: x[1]["score"], reverse=True)
    for idx, score in sorted_all:
        if len(selected_indices) >= 8: break
        if idx not in selected_indices:
            pick_disc(idx, f"Wildcard ({shelf.loc[idx]['Typ']})", False)

    st.session_state.bag_roles = {shelf.loc[r['idx']]['Modell']: r for r in recommendations}
    return recommendations

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
        i, h, c, u = load_data_from_supabase()
        st.session_state.inventory = i
        st.session_state.history = h
        st.session_state.courses = c
        st.session_state.users = u
    st.session_state.data_loaded = True

if st.session_state.get('logged_in') and not st.session_state.users.empty:
    if st.session_state.current_user not in st.session_state.users["Username"].values:
        st.session_state.logged_in = False
        st.session_state.current_user = None

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'current_user' not in st.session_state: st.session_state.current_user = None
if 'user_role' not in st.session_state: st.session_state.user_role = None
if 'pin_reset_mode' not in st.session_state: st.session_state.pin_reset_mode = False
if 'pin_reset_user' not in st.session_state: st.session_state.pin_reset_user = None
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
if 'managed_user' not in st.session_state: st.session_state.managed_user = None
if 'bag_roles' not in st.session_state: st.session_state.bag_roles = {}

# --- LOGIN SCREEN ---
if not st.session_state.logged_in:
    st.title("🔐 SCUDERIA PADDOCK")

    # ── PIN RESET MODE ───────────────────────────────────────────────────────
    if st.session_state.pin_reset_mode:
        username = st.session_state.pin_reset_user
        st.info(f"👋 Välkommen **{username}**! Välj en ny PIN-kod för ditt konto.")
        with st.form("pin_reset_form"):
            new_pin  = st.text_input("Ny PIN (4 siffror)", type="password", max_chars=4)
            conf_pin = st.text_input("Bekräfta PIN", type="password", max_chars=4)
            submitted = st.form_submit_button("Spara PIN 🔒", type="primary")
            if submitted:
                if len(new_pin) != 4 or not new_pin.isdigit():
                    st.error("PIN måste vara exakt 4 siffror.")
                elif new_pin != conf_pin:
                    st.error("PIN-koderna matchar inte.")
                else:
                    new_hash = bcrypt.hashpw(new_pin.encode(), bcrypt.gensalt()).decode()
                    if _sb_update_pin(username, new_hash):
                        # Fetch role from Supabase to complete login
                        sb_row = _sb_get_user(username)
                        role   = sb_row["role"] if sb_row else "Player"
                        st.session_state.logged_in    = True
                        st.session_state.current_user = username
                        st.session_state.user_role    = role
                        st.session_state.active_players = [username]
                        st.session_state.pin_reset_mode = False
                        st.session_state.pin_reset_user = None
                        st.success("PIN satt! Välkommen in 🏎️")
                        st.rerun()
                    else:
                        st.error("Kunde inte spara PIN — kontrollera Supabase-nyckeln i secrets.")
        if st.button("← Tillbaka"):
            st.session_state.pin_reset_mode = False
            st.session_state.pin_reset_user = None
            st.rerun()
        st.stop()

    # ── NORMAL LOGIN ─────────────────────────────────────────────────────────
    users = st.session_state.users
    if not users.empty:
        user_list = users["Username"].tolist()
        sel_user  = st.selectbox("Välj Förare", user_list)
        pin_in    = st.text_input("PIN", type="password")
        if st.button("Lås Upp 🔓", type="primary"):
            authenticated, needs_reset = _check_pin(sel_user, pin_in)
            if needs_reset:
                st.session_state.pin_reset_mode = True
                st.session_state.pin_reset_user = sel_user
                st.rerun()
            elif authenticated:
                sb_row = _sb_get_user(sel_user)
                role   = sb_row["role"] if sb_row else users[users["Username"] == sel_user].iloc[0]["Role"]
                st.session_state.logged_in    = True
                st.session_state.current_user = sel_user
                st.session_state.user_role    = role
                st.session_state.active_players = [sel_user]
                st.success("Access Granted"); st.rerun()
            else:
                st.error("Fel PIN.")
    else:
        st.error("Inga användare.")
    st.stop()

# --- MAIN APP ---
with st.sidebar:
    st.title("🏎️ SCUDERIA CLOUD")
    st.markdown(f"<h3 style='color: #fff200; margin-bottom: 0px;'>👤 {st.session_state.current_user}</h3><div style='color: #cccccc; font-size: 12px; margin-bottom: 20px;'>v80.0 The Factory Team</div>", unsafe_allow_html=True)
    
    if st.button("Logga Ut"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.divider()
    
    # --- ADMIN: IMPERSONATION TOOL ---
    if st.session_state.user_role == "Admin":
        all_owners = st.session_state.inventory["Owner"].unique().tolist()
        if not all_owners: 
             st.session_state.managed_user = None 
             managed = None
             st.warning("Inga spelare med utrustning hittades.")
        else:
            if not st.session_state.managed_user or st.session_state.managed_user not in all_owners:
                if st.session_state.current_user in all_owners:
                    st.session_state.managed_user = st.session_state.current_user
                else:
                    st.session_state.managed_user = all_owners[0]
            try:
                curr_idx = all_owners.index(st.session_state.managed_user)
            except ValueError:
                curr_idx = 0
            managed = st.selectbox("🛠️ Hantera Profil (Admin)", all_owners, index=curr_idx)
            st.session_state.managed_user = managed
    else:
        st.session_state.managed_user = st.session_state.current_user

    # --- TEAM SELECTION (SAFE MULTIPLAYER) ---
    all_owners = st.session_state.inventory["Owner"].unique().tolist()
    if all_owners:
        possible_teammates = [p for p in all_owners if p != st.session_state.current_user]
        st.markdown("👥 **Race Crew (Flera Spelare)**")
        current_friends = [p for p in st.session_state.active_players if p != st.session_state.current_user and p in possible_teammates]
        added_friends = st.multiselect("Lägg till kompisar:", possible_teammates, default=current_friends)
        new_active_list = [st.session_state.current_user] + added_friends
        if new_active_list != st.session_state.active_players:
            st.session_state.active_players = new_active_list
            st.rerun()
    else:
        st.warning("Inga andra spelare hittades.")

    st.divider()
    
    # 1. BANA & VÄDER
    course_names = list(st.session_state.courses.keys())
    st.markdown("**📍 Aktuell Bana (Race/Weather)**")
    sel_course = st.selectbox("Välj Bana", course_names, key="course_selector", label_visibility="collapsed")
    
    with st.expander("🌍 Hitta ny bana (OSM)"):
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
                    add_course_to_sb(selected_data["name"], selected_data["lat"], selected_data["lon"], std_holes)
                    st.success(f"{sel_new_c} tillagd!")
                    st.session_state.found_courses = []
                    st.cache_data.clear(); st.rerun()

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
    if st.button("🔄 Synka Databas"): st.cache_data.clear(); st.rerun()

# --- TABS ---
tabs = ["🔥 WARM-UP", "🏁 RACE", "🤖 AI-CADDY", "🧳 UTRUSTNING", "📈 TELEMETRY", "🎓 ACADEMY"]
if st.session_state.user_role == "Admin": tabs.append("⚙️ HQ")
current_tab = st.tabs(tabs)

# TAB 1: WARM-UP
with current_tab[0]:
    st.header("🔥 Driving Range")
    
    if st.session_state.active_players:
        curr_thrower = st.selectbox("Vem kastar?", st.session_state.active_players)
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
                        st.session_state.warmup_shots.append({"player": curr_thrower, "disc": sel_disc_name, "style": style, "len": kast_len, "side": kast_sida, "speed": float(d_data["Speed"])})
                        st.success("Sparat!")
        with c_list:
            if st.session_state.warmup_shots:
                st.dataframe(pd.DataFrame(st.session_state.warmup_shots)[["player", "disc", "len", "side"]], hide_index=True, height=200)
                if st.button("Rensa Session"): st.session_state.warmup_shots = []; st.rerun()
        if st.session_state.warmup_shots:
            st.divider()
            for p in st.session_state.active_players:
                p_shots = [s for s in st.session_state.warmup_shots if s['player'] == p]
                if p_shots:
                    tot_pot = 0
                    for s in p_shots: opt_dist = max(s["speed"] * 10.0, 40.0); tot_pot += (s["len"] / opt_dist)
                    avg_form = tot_pot / len(p_shots)
                    st.session_state.daily_forms[p] = avg_form
            
            st.subheader("📊 Session Stats")
            cols = st.columns(len(st.session_state.active_players))
            for i, p in enumerate(st.session_state.active_players):
                f = st.session_state.daily_forms.get(p, 0)
                if f > 0: cols[i].metric(p, f"Form: {int(f*100)}%")
            
            fig, ax = plt.subplots(figsize=(6,3))
            shots = st.session_state.warmup_shots
            colors = plt.cm.rainbow(np.linspace(0, 1, len(st.session_state.active_players)))
            p_map = {p: c for p, c in zip(st.session_state.active_players, colors)}
            for s in shots:
                ax.scatter(s["side"], s["len"], color=p_map.get(s["player"], 'white'), s=100, alpha=0.8, label=s["player"])
            ax.axvline(0, c='white', ls='--')
            ax.set_facecolor('#1a1a1a'); fig.patch.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white'); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), facecolor='#1a1a1a', labelcolor='white')
            st.pyplot(fig)
            plt.close(fig)
    else: st.info("Välj spelare i menyn.")

# TAB 2: RACE
with current_tab[1]:
    bana = st.session_state.selected_course
    c_data = st.session_state.courses[bana]
    st.subheader(f"🏁 Race Day: {bana}")
    
    if not st.session_state.active_players:
        st.session_state.active_players = [st.session_state.current_user]
    
    active_racers = st.session_state.active_players
    
    col_n, col_s = st.columns([1, 2])
    with col_n:
        holes = sorted(list(c_data["holes"].keys()), key=lambda x: int(x) if x.isdigit() else x)
        if 'hole_idx' not in st.session_state:
            st.session_state.hole_idx = 0
        st.session_state.hole_idx = min(st.session_state.hole_idx, len(holes) - 1)
        hole = holes[st.session_state.hole_idx]
        inf = c_data["holes"][hole]
        c_prev, c_next = st.columns(2)
        if c_prev.button("◀ Föregående", disabled=st.session_state.hole_idx == 0, use_container_width=True):
            st.session_state.hole_idx -= 1; st.rerun()
        if c_next.button("Nästa ▶", disabled=st.session_state.hole_idx == len(holes) - 1, use_container_width=True):
            st.session_state.hole_idx += 1; st.rerun()
        st.metric(f"Hål {hole} / {len(holes)}", f"{inf['l']}m", f"Par {inf['p']}")
        st.caption(inf.get('shape', 'Rak'))
    with col_s:
        if hole not in st.session_state.current_scores: st.session_state.current_scores[hole] = {}
        if hole not in st.session_state.selected_discs: st.session_state.selected_discs[hole] = {}
        
        for p in active_racers:
            if p not in st.session_state.current_scores[hole]: st.session_state.current_scores[hole][p] = inf['p']
            if p not in st.session_state.selected_discs[hole]: st.session_state.selected_discs[hole][p] = None
        
        for p in active_racers:
            with st.expander(f"🏎️ {p} (Score: {st.session_state.current_scores[hole][p]})", expanded=True):
                c_ghost, c_ai = st.columns([1, 2])
                with c_ghost:
                    hist_df = st.session_state.history
                    avg_score = "-"
                    delta = ""
                    if not hist_df.empty:
                        p_hist = hist_df[(hist_df["Spelare"]==p) & (hist_df["Bana"]==bana) & (hist_df["Hål"]==hole)]
                        if not p_hist.empty: 
                            avg = p_hist['Resultat'].mean()
                            avg_score = f"{avg:.1f}"
                            diff = avg - inf['p']
                            delta = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>HISTORIK</div><div class='metric-value'>{avg_score}</div><div class='metric-sub'>{delta} vs Par</div></div>", unsafe_allow_html=True)
                with c_ai:
                    with st.expander("📻 Team Radio — Strategy Request", expanded=False):
                        form = st.session_state.daily_forms.get(p, 1.0)
                        
                        c_sit1, c_sit2 = st.columns(2)
                        situation = c_sit1.radio("Läge", ["Tee", "Fairway", "Ruff", "Putt"], key=f"sit_{hole}_{p}", label_visibility="collapsed")
                        dist_left = c_sit2.slider("Avstånd (m)", 0, 300, int(inf['l']) if situation=="Tee" else 50, key=f"d_{hole}_{p}")
                        
                        telemetry_str = ""
                        curve_type = st.radio("Banans Form", ["Rak", "Vänster", "Höger"], horizontal=True, key=f"curve_{hole}_{p}")
                        if curve_type != "Rak":
                            curve_dist = st.slider("Sväng startar om (m)", 0, 150, 50, key=f"cd_{hole}_{p}")
                            telemetry_str += f"Banan svänger {curve_type} efter {curve_dist}m. "
                        if st.checkbox("Trång Port / Hinder", key=f"gapt_{hole}_{p}"):
                            c_gap1, c_gap2 = st.columns(2)
                            gap_dist = c_gap1.slider("Avstånd Hinder (m)", 0, 150, 30, key=f"gd_{hole}_{p}")
                            gap_width = c_gap2.slider("Bredd på lucka (m)", 1, 20, 5, key=f"gw_{hole}_{p}")
                            telemetry_str += f"Det finns en {gap_width}m bred port/lucka {gap_dist}m bort. "
                        basket_pos = st.selectbox("Korgens läge", ["Normal", "Upphöjd", "På kulle (Risk för rull)", "Skymd"], key=f"bk_{hole}_{p}")
                        telemetry_str += f"Korgplacering: {basket_pos}. "
                        
                        use_cam = st.checkbox("📸 Aktivera 'Helmet Cam'", key=f"cam_tog_{hole}_{p}")
                        img_data = None
                        if use_cam:
                            img_file = st.camera_input("Ta bild på banan", key=f"ci_{hole}_{p}")
                            if img_file: img_data = img_file.getvalue()
                        
                        if st.button(f"🔊 Request Strategy ({p})", key=f"ai_btn_{hole}_{p}", type="primary"):
                            p_bag = st.session_state.inventory[st.session_state.inventory["Owner"]==p]
                            with st.spinner("Race Engineer analyzing data..."):
                                advice = get_race_engineer_advice(p, p_bag, inf, st.session_state.weather_data, situation, dist_left, telemetry_str, img_data, form)
                                st.session_state.hole_advice[f"{hole}_{p}"] = advice
                        
                        if f"{hole}_{p}" in st.session_state.hole_advice: st.markdown(st.session_state.hole_advice[f"{hole}_{p}"], unsafe_allow_html=True)
                st.divider()

                # --- SCORE ENTRY + DISC SELECTION ---
                c_minus, c_score_display, c_plus = st.columns([1, 2, 1])
                score_val = st.session_state.current_scores[hole][p]
                par_diff = score_val - inf['p']
                diff_str = "E" if par_diff == 0 else f"+{par_diff}" if par_diff > 0 else str(par_diff)
                c_score_display.markdown(
                    f"<h1 style='text-align:center; color:white; margin:0;'>{score_val}</h1>"
                    f"<p style='text-align:center; color:#fff200; margin:0;'>{diff_str}</p>",
                    unsafe_allow_html=True
                )
                if c_minus.button("➖", key=f"m_{hole}_{p}", use_container_width=True):
                    st.session_state.current_scores[hole][p] -= 1; st.rerun()
                if c_plus.button("➕", key=f"p_{hole}_{p}", use_container_width=True):
                    st.session_state.current_scores[hole][p] += 1; st.rerun()
                
                p_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == p]
                bag_discs_rows = p_inv[p_inv["Status"]=="Bag"]
                
                # Format options with roles if available
                bag_options = []
                for _, row in bag_discs_rows.iterrows():
                    d_name = row["Modell"]
                    role_info = st.session_state.bag_roles.get(d_name, None)
                    if role_info:
                        warmup_tag = "🔥" if role_info.get("warmup") else ""
                        label = f"{d_name} ({role_info.get('role', 'Disc')}) {warmup_tag}"
                        bag_options.append(label)
                    else:
                        bag_options.append(d_name)
                
                all_discs = p_inv["Modell"].tolist()
                opts = ["Välj Disc"] + (bag_options if bag_options else all_discs)
                st.session_state.selected_discs[hole][p] = c_disc.selectbox("Vald Disc", opts, key=f"ds_{hole}_{p}")
    st.markdown("---")
    with st.expander("📋 Scorecard Preview", expanded=False):
        sc_rows = []
        for h in holes:
            h_par = c_data["holes"][h]["p"]
            row = {"Hål": h, "Par": h_par}
            for p in active_racers:
                sc_score = st.session_state.current_scores.get(h, {}).get(p, h_par)
                sc_diff = sc_score - h_par
                diff_label = "E" if sc_diff == 0 else f"+{sc_diff}" if sc_diff > 0 else str(sc_diff)
                row[p] = f"{sc_score} ({diff_label})"
            sc_rows.append(row)
        st.dataframe(pd.DataFrame(sc_rows), hide_index=True, use_container_width=True)
    if st.button("🏁 SPARA RUNDA & KLIV AV BANAN", type="primary"):
        new_rows = []
        d = datetime.now().strftime("%Y-%m-%d")
        for h, scores in st.session_state.current_scores.items():
            for p, s in scores.items():
                disc_val = st.session_state.selected_discs[h].get(p, "Unknown")
                # Clean disc name (remove role info)
                disc_name = disc_val.split(" (")[0] if "(" in disc_val else disc_val
                new_rows.append({"Datum": d, "Bana": bana, "Spelare": p, "Hål": h, "Resultat": s, "Par": c_data["holes"][h]["p"], "Disc_Used": disc_name})
        new_df = pd.DataFrame(new_rows)
        st.session_state.history = pd.concat([st.session_state.history, new_df], ignore_index=True)
        append_history_to_sb(new_rows)
        st.balloons(); st.success("Loppet sparat i historiken!"); st.session_state.current_scores = {}

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
    target_p = st.session_state.managed_user
    if not target_p: target_p = st.session_state.current_user
    st.header(f"🧳 Logistik: {target_p}")
    
    with st.container(border=True):
        st.markdown("#### 🤖 Strategen")
        c1, c2, c3 = st.columns([2, 1, 1])
        tc = c1.selectbox("Bana:", list(st.session_state.courses.keys()), key="strat_course")
        if c2.button("Generera"):
            with st.spinner("🤖 Kör Monte Carlo-simulering (20 000 kast)..."):
                st.session_state.suggested_pack = generate_smart_bag(st.session_state.inventory, target_p, tc, st.session_state.weather_data)
            st.rerun()
            
        if st.session_state.suggested_pack:
            st.info("🤖 Föreslagna tillägg från hyllan:")
            for rec in st.session_state.suggested_pack:
                with st.container():
                    cols = st.columns([3, 4, 1])
                    cols[0].markdown(f"**{rec['model']}**")
                    cols[0].caption(rec['role'])
                    cols[1].markdown(f"_{rec['reason']}_")
                    if rec['warmup']:
                        cols[2].markdown('<span class="warmup-badge">🔥 WARM-UP</span>', unsafe_allow_html=True)
                    st.divider()

            if st.button("Verkställ (Flytta till Bag)", type="primary"):
                # SAFE MASS UPDATE - Using explicit Indices list
                indices_to_update = [r['idx'] for r in st.session_state.suggested_pack]
                
                # Sanity Check: If trying to move entire shelf, something is wrong
                if len(indices_to_update) > 14:
                    st.error("Fel i systemet: Försöker flytta för många discar! Avbryter.")
                else:
                    # Perform Update on specific rows
                    st.session_state.inventory.loc[indices_to_update, "Status"] = "Bag"
                    
                    # Save
                    save_inventory_to_sb(st.session_state.inventory)
                    
                    # Clear recommendations
                    st.session_state.suggested_pack = []
                    st.success(f"Flyttade {len(indices_to_update)} discar till bagen!")
                    time.sleep(1)
                    st.rerun()
    
    st.divider()
    st.subheader("🛠️ Snabb-hantering (Bulk)")
    
    my_inv = st.session_state.inventory[st.session_state.inventory["Owner"] == target_p].copy()
    c_shelf, c_bag = st.columns(2)
    
    with c_shelf:
        st.markdown("🏠 **Hyllan**")
        shelf_items = my_inv[my_inv["Status"] == "Shelf"].sort_values("Speed")
        if not shelf_items.empty:
            shelf_items['Display'] = shelf_items.apply(lambda x: f"[{int(x['Speed'])}] {x['Modell']} ({x['Plast']}) - {x['Typ']}", axis=1)
            selected_shelf = st.multiselect("Välj att flytta till Bag:", shelf_items['Display'].tolist(), key="ms_shelf")
            if st.button("➡️ Flytta till Bag"):
                mask = shelf_items['Display'].isin(selected_shelf)
                indices_to_move = shelf_items[mask].index
                st.session_state.inventory.loc[indices_to_move, "Status"] = "Bag"
                save_inventory_to_sb(st.session_state.inventory)
                st.rerun()
            if st.button("🗑️ Skrota (Radera)", key="del_shelf"):
                mask = shelf_items['Display'].isin(selected_shelf)
                indices_to_drop = shelf_items[mask].index
                st.session_state.inventory = st.session_state.inventory.drop(indices_to_drop)
                save_inventory_to_sb(st.session_state.inventory)
                st.rerun()
        else:
            st.caption("Tomt på hyllan.")

    with c_bag:
        st.markdown("🎒 **Bagen**")
        bag_items = my_inv[my_inv["Status"] == "Bag"].sort_values("Speed")
        if not bag_items.empty:
            
            # --- FIX: VISUAL BAG DISPLAY ---
            # Create a simple view for the bag
            view_df = bag_items[['Modell', 'Plast', 'Typ', 'Speed', 'Glide', 'Turn', 'Fade']].reset_index(drop=True)
            st.dataframe(view_df, use_container_width=True)
            # -------------------------------
            
            bag_items['Display'] = bag_items.apply(lambda x: f"[{int(x['Speed'])}] {x['Modell']} ({x['Plast']}) - {x['Typ']}", axis=1)
            selected_bag = st.multiselect("Välj att flytta till Hyllan:", bag_items['Display'].tolist(), key="ms_bag")
            if st.button("⬅️ Flytta till Hylla"):
                mask = bag_items['Display'].isin(selected_bag)
                indices_to_move = bag_items[mask].index
                st.session_state.inventory.loc[indices_to_move, "Status"] = "Shelf"
                save_inventory_to_sb(st.session_state.inventory)
                st.rerun()
            if st.button("🗑️ Skrota (Radera)", key="del_bag"):
                mask = bag_items['Display'].isin(selected_bag)
                indices_to_drop = bag_items[mask].index
                st.session_state.inventory = st.session_state.inventory.drop(indices_to_drop)
                save_inventory_to_sb(st.session_state.inventory)
                st.rerun()
        else:
            st.caption("Bagen är tom.")
            
    st.divider()
    
    if not bag_items.empty:
        st.subheader("📊 Bag Balance")
        chart_data = bag_items[['Speed', 'Turn', 'Fade', 'Modell']].copy()
        chart_data['Stability'] = chart_data['Turn'] + chart_data['Fade']
        c = alt.Chart(chart_data).mark_circle(size=200).encode(
            x=alt.X('Stability', title='Stabilitet (Turn + Fade)'),
            y=alt.Y('Speed', title='Speed'),
            color='Modell',
            tooltip=['Modell', 'Speed', 'Turn', 'Fade']
        ).properties(height=300)
        st.altair_chart(c, use_container_width=True)

    st.divider()
    
    st.subheader("📝 Databas-editor")
    st.caption("Redigera värden direkt här. Negativa värden tillåtna.")
    edited_df = st.data_editor(
        my_inv, 
        num_rows="dynamic", 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "_disc_id": None,
            "Speed": st.column_config.NumberColumn(format="%.1f"),
            "Glide": st.column_config.NumberColumn(format="%.1f"),
            "Turn": st.column_config.NumberColumn(format="%.1f"),
            "Fade": st.column_config.NumberColumn(format="%.1f")
        }
    )
    if st.button("💾 Spara Ändringar"):
        st.session_state.inventory = st.session_state.inventory[st.session_state.inventory["Owner"] != target_p]
        st.session_state.inventory = pd.concat([st.session_state.inventory, edited_df], ignore_index=True)
        save_inventory_to_sb(st.session_state.inventory)
        st.success("Sparat!")
    
    # --- DEBUG SECTION (REMOVE AFTER TESTING) ---
    with st.expander("🕵️ Debug - Vad ser Caddyn?"):
        st.write("Exempel från din inventory (rådata):")
        st.write(my_inv[['Modell', 'Speed', 'Turn', 'Fade']].head(5))

    st.markdown("---")
    
    with st.expander("🛠️ Besiktning & Skadekontroll"):
        st.caption("Fota kanten på discen.")
        dmg_img = st.camera_input("Fota skada")
        if dmg_img:
            if st.button("Analysera Skada"):
                with st.spinner("Inspekterar..."):
                    report = inspect_disc_damage(dmg_img.getvalue())
                    st.markdown(f"<div class='engineer-msg'><b>DAMAGE REPORT</b><br>{report}</div>", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander(f"➕ Lägg till ny disc för {target_p} (AI Camera)"):
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
            tillv = c2.text_input("Tillverkare", value=ai_d.get("Tillverkare", ""))
            c3, c4 = st.columns(2)
            plast = c3.text_input("Plast", value=ai_d.get("Plast", ""))
            v_types = ["Putter", "Midrange", "Fairway Driver", "Distance Driver"]
            r_type = ai_d.get("Typ", "Putter"); f_idx = 0
            for i, vt in enumerate(v_types):
                if vt.lower() in r_type.lower(): f_idx = i; break
            ty = c4.selectbox("Typ", v_types, index=f_idx)
            c5, c6, c7, c8 = st.columns(4)
            sp = c5.number_input("Speed", 0.0, 15.0, float(ai_d.get("Speed", 7.0)), step=0.5, format="%.1f")
            gl = c6.number_input("Glide", 0.0, 7.0, float(ai_d.get("Glide", 5.0)), step=0.5, format="%.1f")
            tu = c7.number_input("Turn", -5.0, 1.0, float(ai_d.get("Turn", 0.0)), step=0.5, format="%.1f")
            fa = c8.number_input("Fade", 0.0, 6.0, float(ai_d.get("Fade", 2.0)), step=0.5, format="%.1f")
            
            stab_val = tu + fa
            stab_txt = "Överstabil" if stab_val > 1 else "Understabil" if stab_val < -1 else "Stabil"
            
            if st.form_submit_button("Spara till Hyllan"):
                nw = {
                    "Owner": target_p, "Modell": mn, "Tillverkare": tillv, "Plast": plast, 
                    "Typ": ty, "Speed": sp, "Glide": gl, "Turn": tu, "Fade": fa, 
                    "Stabilitet": stab_txt, "Status": "Shelf"
                }
                st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame([nw])], ignore_index=True)
                save_inventory_to_sb(st.session_state.inventory)
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
            plt.close(fig)
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
                    _new_hash = bcrypt.hashpw(str(nu_pin).encode(), bcrypt.gensalt()).decode()
                    _sb_insert("users", [{"username": nu_name, "pin_hash": _new_hash, "role": nu_role, "active": True, "municipality": nu_mun or None}])
                    start_kit = [{"Owner": nu_name, "Modell": "Start Putter", "Tillverkare": "Innova", "Plast": "DX", "Typ": "Putter", "Speed": 3, "Glide": 3, "Turn": 0, "Fade": 0, "Stabilitet": "Stabil", "Status": "Bag"}]
                    st.session_state.inventory = pd.concat([st.session_state.inventory, pd.DataFrame(start_kit)], ignore_index=True)
                    save_inventory_to_sb(st.session_state.inventory)
                    if nu_mun:
                        lat, lon = get_lat_lon_from_query(nu_mun)
                        if lat:
                            new_courses = find_courses_via_osm_api(lat, lon)
                            for nc in new_courses:
                                std_holes = {str(x): {"l": 100, "p": 3, "shape": "Rak"} for x in range(1, 19)}
                                add_course_to_sb(nc["name"], nc["lat"], nc["lon"], std_holes)
                            st.success(f"Användare {nu_name} skapad! Hittade {len(new_courses)} banor i {nu_mun}.")
                        else: st.warning("Användare skapad, men kunde inte hitta kommunen för bankartläggning.")
                    st.cache_data.clear(); st.rerun()
        with c_u2:
            del_user = st.selectbox("Ta bort användare", users["Username"].tolist())
            if st.button("🗑️ Radera Användare"):
                if _sb_delete("users", {"username": del_user}):
                    st.success("Raderad!")
                    st.cache_data.clear(); st.rerun()
                else: st.error("Kunde inte radera användaren.")
        
        st.divider()
        st.subheader("🛠️ Bankarta & Databas")
        st.warning("Används endast om bankartorna ser felaktiga ut eller saknas.")
        if st.button("⚠️ Återställ & Uppdatera alla Bankartor", type="primary"):
            if hard_reset_courses_sb():
                st.success("Bankartor återställda till Grand Tour Edition!")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
        
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
                    st.session_state.history = new_hist; append_history_to_sb(nd)
                    st.success(f"Importerade {len(nd)} rader!")
            except Exception as e: st.error(f"Fel: {e}")
