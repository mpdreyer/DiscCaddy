"""
DiscCaddy → Supabase migration script
Run once: python3 migrate_to_supabase.py
"""
import csv
import json
import sys
import uuid
import urllib.request
import urllib.error
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpwZWlqam5wa2F5enJnZGVmanhpIiwicm9sZSI6"
    "InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTkyMDk0NywiZXhwIjoyMDg3NDk2OTQ3fQ"
    ".sF8Jf-wPi7AZD9s8MurN8f6C6yoAlLzyGE68hVZE_Xw"
)
PROJECT_URL = "https://jpeijjnpkayzrgdefjxi.supabase.co"

INVENTORY_CSV    = "/Users/mattiasdreyer/DiscCaddy/inventory.csv"
HISTORY_CSV      = "/Users/mattiasdreyer/DiscCaddy/round_history.csv"
COURSES_JSON     = "/Users/mattiasdreyer/DiscCaddy/courses.json"

# Default PIN hash — bcrypt("1234"), pre-computed so we don't need bcrypt installed
# In production replace with proper bcrypt; this is a placeholder migration value
DEFAULT_PIN_HASH = "$2b$12$placeholder_migration_hash_1234"

# ── Course name normalisation ─────────────────────────────────────────────────
# Maps the long names stored in round_history.csv → canonical names used in DB
HISTORY_NAME_MAP = {
    "Kungsbackaskogens Discgolfbana": "Kungsbackaskogen",
    "Lygnevi Discgolf":               "Lygnevi (18 Hål)",
    "Skatås Discgolf":                "Skatås (Gul)",
    "St Hans DGB":                    "Sankt Hans (Lund)",
    "Åbyvallens Discgolfbana":        "Åbyvallen (Mölndal)",
}

# ── MASTER_COURSES (extracted from disc_app.py) ───────────────────────────────
def _holes(lengths, pars=None, shapes=None):
    if pars   is None: pars   = [3]     * len(lengths)
    if shapes is None: shapes = ["Rak"] * len(lengths)
    return {str(i+1): {"l": l, "p": p, "shape": s}
            for i, (l, p, s) in enumerate(zip(lengths, pars, shapes))}

MASTER_COURSES = {
    "Kungsbackaskogen": {
        "lat": 57.492, "lon": 12.075,
        "holes": _holes(
            [63,81,48,65,75,55,62,78,52], [3]*9,
            ["Rak","Vänster","Rak","Höger","Rak","Vänster","Rak","Rak","Rak"])},
    "Onsala Discgolf":      {"lat":57.416,"lon":12.029,"holes":_holes([65]*18,[3]*18,["Rak"]*18)},
    "Lygnevi (18 Hål)":     {"lat":57.545,"lon":12.433,"holes":_holes([85]*18,[3]*18,["Park/Vatten"]*18)},
    "Lygnevi (Gul - 9 Hål)":{"lat":57.545,"lon":12.433,
        "holes":_holes([75,68,82,55,90,60,72,85,70],[3]*9,
                       ["Skog","Vä","Vatten/Hö","Kort","Lång","Vä","Hö","Lång","Vatten"])},
    "Lygnevi (Kort - 9 Hål)":{"lat":57.545,"lon":12.433,"holes":_holes([50,45,55,40,60,45,50,55,40],[3]*9,["Park"]*9)},
    "Åbyvallen (Mölndal)":  {"lat":57.643,"lon":12.018,
        "holes":_holes([55,62,48,70,58,65,50,68],[3]*8,
                       ["Rak","Vänster","Höger","Rak","Vänster","Rak","Höger","Rak"])},
    "Skatås (Gul)":         {"lat":57.704,"lon":12.036,
        "holes":_holes(
            [85,72,95,68,105,80,75,110,65,90,88,145,70,82,95,60,100,85],
            [3,3,3,3,3,3,3,3,3,3,3,4,3,3,3,3,3,3],
            ["Skog","Vä","Rak","Hö","Uppför","Rak","Vä","Öppen","Brant","Rak","Hö","Svår","Vä","Rak","Hö","Ö","Rak","Uppför"])},
    "Ale Discgolf (Vit)":   {"lat":57.947,"lon":12.134,
        "holes":_holes(
            [145,110,205,125,160,100,180,135,260,120,155,130,195,115,140,170,125,210],
            [4,3,5,3,4,3,4,3,5,3,4,3,5,3,4,4,3,5],
            ["Lång/Vä","Vatten/Hö","Lång/Rak","Teknisk","Uppför","Nedför","Lång/Vä","Skog","Monster","Vatten","Lång","Hö/OB","S-kurva","Vä","Hö","Lång/Rak","Ö","Lång/Vind"])},
    "Ale Discgolf (Gul)":   {"lat":57.947,"lon":12.134,"holes":_holes([75]*18,[3]*18,["Skog/Teknisk"]*18)},
    "Uspastorp":            {"lat":57.982,"lon":12.148,"holes":_holes([90]*18)},
    "Ymer (Borås)":         {"lat":57.747,"lon":12.909,"holes":_holes([95]*18)},
    "Gässlösa (Varberg)":   {"lat":57.106,"lon":12.285,"holes":_holes([80]*18)},
    "Falkenberg (Vid havet)":{"lat":56.893,"lon":12.508,"holes":_holes([85]*18)},
    "Hylte (Hyltebruk)":    {"lat":56.994,"lon":13.238,"holes":_holes([100]*18)},
    "Stenungsund":          {"lat":58.072,"lon":11.838,"holes":_holes([80]*18)},
    "Sankt Hans (Lund)":    {"lat":55.723,"lon":13.208,
        "holes":_holes(
            [85,115,70,95,125,60,80,105,90,75,130,85,65,110,70,95,80,100],
            [3,3,3,3,4,3,3,3,3,3,4,3,3,3,3,3,3,3],
            ["Uppför","Nedför","Kulle","Blind","Lång","Kort","Skrå","Nedför","Uppför","Hö","Lång","Vä","Kort","Lång","Vä","Hö","Blind","Lång"])},
    "Vipeholm (Lund)":      {"lat":55.701,"lon":13.220,"holes":_holes([70]*18)},
    "Bulltofta (Malmö)":    {"lat":55.605,"lon":13.064,"holes":_holes([85]*18)},
    "Sibbarp (Malmö)":      {"lat":55.574,"lon":12.912,"holes":_holes([80]*9)},
    "Trollsjö (Eslöv)":     {"lat":55.836,"lon":13.305,"holes":_holes([75]*18)},
    "Romeleåsen":           {"lat":55.597,"lon":13.435,"holes":_holes([100]*18)},
}

# ── Supabase REST helpers ─────────────────────────────────────────────────────
HEADERS = {
    "apikey":        SERVICE_KEY,
    "Authorization": f"Bearer {SERVICE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=representation",
}

def sb_post(path: str, rows: list) -> list:
    """POST (upsert) rows to a Supabase table. Returns inserted rows."""
    if not rows:
        return []
    url  = f"{PROJECT_URL}/rest/v1/{path}"
    data = json.dumps(rows).encode()
    req  = urllib.request.Request(url, data=data, headers={
        **HEADERS, "Prefer": "return=representation,resolution=ignore-duplicates"
    }, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  ERROR on {path}: {e.code} {e.read().decode()[:300]}")
        sys.exit(1)

def sb_get(path: str, select: str = "*") -> list:
    url = f"{PROJECT_URL}/rest/v1/{path}?select={select}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

# ── 1. USERS ─────────────────────────────────────────────────────────────────
def migrate_users() -> dict:
    """Returns {normalised_username: uuid}"""
    print("\n── Users ────────────────────────────────────")

    # Collect unique owners from inventory AND history (normalise to Title Case)
    owners = set()
    with open(INVENTORY_CSV) as f:
        for row in csv.DictReader(f):
            owners.add(row["Owner"].strip().title())
    with open(HISTORY_CSV) as f:
        for row in csv.DictReader(f):
            owners.add(row["Spelare"].strip().title())

    rows = []
    for owner in sorted(owners):
        rows.append({
            "username":     owner,
            "pin_hash":     DEFAULT_PIN_HASH,
            "role":         "Player",
            "active":       True,
            "municipality": "Kungsbacka" if owner.lower() == "mattias" else None,
        })
    # Ensure an Admin account exists
    if "Admin" not in {r["username"] for r in rows}:
        rows.append({
            "username":     "Admin",
            "pin_hash":     DEFAULT_PIN_HASH,
            "role":         "Admin",
            "active":       True,
            "municipality": None,
        })
    else:
        # Promote existing Admin user
        for r in rows:
            if r["username"] == "Admin":
                r["role"] = "Admin"

    inserted = sb_post("users", rows)
    id_map   = {r["username"]: r["id"] for r in inserted}

    # If rows already existed (ignore-duplicates) re-fetch
    if not id_map:
        existing = sb_get("users", "id,username")
        id_map   = {r["username"]: r["id"] for r in existing}

    for name, uid in id_map.items():
        print(f"  {name:<20} {uid}")
    return id_map

# ── 2. COURSES + HOLES ────────────────────────────────────────────────────────
def migrate_courses() -> dict:
    """Returns {course_name: {id, holes: {hole_number: hole_id}}}"""
    print("\n── Courses + Holes ──────────────────────────")

    # Merge: MASTER_COURSES + courses.json (json wins on overlap)
    all_courses = dict(MASTER_COURSES)
    try:
        with open(COURSES_JSON) as f:
            local = json.load(f)
        all_courses.update(local)
    except FileNotFoundError:
        pass

    # Insert courses
    course_rows = [
        {"name": name, "lat": data["lat"], "lon": data["lon"]}
        for name, data in all_courses.items()
    ]
    inserted_courses = sb_post("courses", course_rows)
    if not inserted_courses:
        inserted_courses = sb_get("courses", "id,name")

    course_id_map = {r["name"]: r["id"] for r in inserted_courses}
    print(f"  {len(course_id_map)} courses upserted")

    # Insert holes for each course
    hole_rows = []
    for name, data in all_courses.items():
        cid = course_id_map.get(name)
        if not cid:
            continue
        for h_num, h_data in data["holes"].items():
            hole_rows.append({
                "course_id":   cid,
                "hole_number": int(h_num),
                "length_m":    int(h_data["l"]),
                "par":         int(h_data.get("p", 3)),
                "shape":       h_data.get("shape") or None,
            })

    inserted_holes = sb_post("holes", hole_rows)
    if not inserted_holes:
        inserted_holes = sb_get("holes", "id,course_id,hole_number")

    # Build nested map: course_name → hole_number → hole_id
    cid_to_name = {v: k for k, v in course_id_map.items()}
    course_map  = {name: {"id": cid, "holes": {}} for name, cid in course_id_map.items()}
    for h in inserted_holes:
        cname = cid_to_name.get(h["course_id"])
        if cname:
            course_map[cname]["holes"][int(h["hole_number"])] = h["id"]

    print(f"  {len(inserted_holes)} holes upserted")
    return course_map

# ── 3. DISCS (INVENTORY) ─────────────────────────────────────────────────────
def migrate_discs(user_id_map: dict) -> dict:
    """Returns {(owner_title, model): disc_id}"""
    print("\n── Discs (Inventory) ────────────────────────")

    VALID_TYPES   = {"Putter", "Midrange", "Fairway Driver", "Distance Driver"}
    VALID_STATUS  = {"Bag", "Shelf"}

    rows = []
    with open(INVENTORY_CSV) as f:
        for row in csv.DictReader(f):
            owner = row["Owner"].strip().title()
            uid   = user_id_map.get(owner)
            if not uid:
                print(f"  SKIP disc — unknown owner: {owner!r}")
                continue

            disc_type = row.get("Typ", "").strip()
            status    = row.get("Status", "Shelf").strip().title()

            def safe_float(v, default=0.0):
                try: return float(v)
                except: return default

            rows.append({
                "owner_id":     uid,
                "model":        row["Modell"].strip(),
                "manufacturer": row.get("Tillverkare", "").strip() or None,
                "plastic":      row.get("Plast", "").strip() or None,
                "disc_type":    disc_type if disc_type in VALID_TYPES else None,
                "speed":        safe_float(row.get("Speed")),
                "glide":        safe_float(row.get("Glide")),
                "turn":         safe_float(row.get("Turn")),
                "fade":         safe_float(row.get("Fade")),
                "status":       status if status in VALID_STATUS else "Shelf",
            })

    inserted = sb_post("discs", rows)
    if not inserted:
        inserted = sb_get("discs", "id,owner_id,model")

    # Build lookup: (owner_id, model) → disc_id
    disc_map = {(r["owner_id"], r["model"]): r["id"] for r in inserted}
    print(f"  {len(disc_map)} discs upserted")
    return disc_map

# ── 4. ROUNDS + HOLE SCORES ──────────────────────────────────────────────────
def migrate_history(user_id_map: dict, course_map: dict, disc_map: dict):
    print("\n── Rounds + Hole Scores ─────────────────────")

    # Read and group history rows by (player, course_canonical, date)
    raw_rows = []
    with open(HISTORY_CSV) as f:
        for row in csv.DictReader(f):
            bana      = row["Bana"].strip()
            canonical = HISTORY_NAME_MAP.get(bana, bana)
            raw_rows.append({
                "datum":    row["Datum"].strip(),
                "bana":     canonical,
                "spelare":  row["Spelare"].strip().title(),
                "hål":      row["Hål"].strip(),
                "resultat": int(row["Resultat"]),
                "par":      int(row["Par"]),
                "disc":     row.get("Disc_Used", "").strip(),
            })

    # Group into rounds: key = (player, course, date)
    from collections import defaultdict
    round_groups = defaultdict(list)
    for r in raw_rows:
        round_groups[(r["spelare"], r["bana"], r["datum"])].append(r)

    # Insert rounds batch
    round_rows = []
    round_keys = []
    for (player, course, date), _ in round_groups.items():
        uid = user_id_map.get(player)
        cid = course_map.get(course, {}).get("id")
        if not uid:
            print(f"  SKIP round — unknown player: {player!r}")
            continue
        if not cid:
            print(f"  SKIP round — unknown course: {course!r}")
            continue
        round_rows.append({"player_id": uid, "course_id": cid, "played_at": date})
        round_keys.append((player, course, date))

    inserted_rounds = sb_post("rounds", round_rows)

    # Re-fetch if upsert returned nothing (already existed)
    if not inserted_rounds:
        inserted_rounds = sb_get("rounds", "id,player_id,course_id,played_at")

    # Build round lookup: (player_id, course_id, date) → round_id
    round_id_map = {}
    for r in inserted_rounds:
        round_id_map[(r["player_id"], r["course_id"], r["played_at"])] = r["id"]

    print(f"  {len(inserted_rounds)} rounds upserted")

    # Insert hole scores
    score_rows = []
    skipped    = 0
    for (player, course, date), scores in round_groups.items():
        uid  = user_id_map.get(player)
        cid  = course_map.get(course, {}).get("id")
        if not uid or not cid:
            continue

        round_id   = round_id_map.get((uid, cid, date))
        course_holes = course_map.get(course, {}).get("holes", {})

        if not round_id:
            skipped += 1
            continue

        for s in scores:
            h_num   = int(s["hål"]) if s["hål"].isdigit() else None
            hole_id = course_holes.get(h_num) if h_num else None

            if not hole_id:
                skipped += 1
                continue

            # Try to find the disc_id by model name
            disc_id = disc_map.get((uid, s["disc"])) if s["disc"] not in ("", "Unknown") else None

            if s["resultat"] < 1:
                skipped += 1
                continue  # skip invalid zero/negative scores

            score_rows.append({
                "round_id":  round_id,
                "hole_id":   hole_id,
                "score":     s["resultat"],
                "par":       s["par"],
                "disc_id":   disc_id,
                "disc_name": s["disc"] or None,
            })

    # Deduplicate: if same (round_id, hole_id) appears twice (player played course
    # twice same day but history doesn't distinguish rounds), keep last occurrence
    deduped = {}
    for row in score_rows:
        deduped[(row["round_id"], row["hole_id"])] = row
    score_rows = list(deduped.values())

    inserted_scores = sb_post("hole_scores", score_rows)
    print(f"  {len(score_rows)} hole scores submitted ({skipped} skipped — missing course/hole mapping)")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("  DiscCaddy → Supabase Migration")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 52)

    user_id_map = migrate_users()
    course_map  = migrate_courses()
    disc_map    = migrate_discs(user_id_map)
    migrate_history(user_id_map, course_map, disc_map)

    print("\n✓ Migration complete")
