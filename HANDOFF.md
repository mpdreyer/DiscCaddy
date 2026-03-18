# DiscCaddy — Handoff Notes
_Last updated: 2026-03-18_

---

## Supabase Project

| Field | Value |
|-------|-------|
| Project ID | `jpeijjnpkayzrgdefjxi` |
| Region | eu-west-1 |
| REST base URL | `https://jpeijjnpkayzrgdefjxi.supabase.co/rest/v1` |
| Auth | Service-role key in `.streamlit/secrets.toml` → `[supabase].service_role_key` |

### Table Structure

| Table | Rows | Key columns |
|-------|------|-------------|
| `users` | 9 | id, username, pin_hash, role, active, municipality |
| `courses` | 23 | id, name, lat, lon |
| `holes` | 358 | id, course_id, hole_number, length_m, par, shape |
| `discs` | 25 | id, owner_id, model, manufacturer, plastic, disc_type, speed/glide/turn/fade, stability (generated), status |
| `rounds` | 52 | id, player_id, course_id, played_at — no unique constraint |
| `hole_scores` | 491 | id, round_id, hole_id, score, par, disc_id (SET NULL), disc_name |

---

## Completed Today

### Bug Fix
- **#4** `WARM-UP` line 742: `c2.pyplot` → `st.pyplot` — scatter chart now renders in main area instead of column overflow

### Performance Fixes
- **#5** `get_live_weather`: added `@st.cache_data(ttl=600)` — eliminates repeated API calls per course selection
- **#6** `ask_ai`: added `@st.cache_data(ttl=3600, show_spinner=False)` — same hole/wind/disc combo won't hit GPT-4o twice
- **#7** `plt.close(fig)` after warm-up scatter (line 744) and Aero Lab chart (line 1108) — prevents matplotlib memory leaks
- **#8** `save_to_sheet` replaced with append-only `append_history_to_sb()` — no more full sheet rewrites on each round save
- **#9** Removed `time.sleep(1.0)` at line 892 — unnecessary blocking call eliminated
- **#10** `st.cache_data.clear()` on lines 661+677 — cache properly invalidated without destroying GSheet auth client

### UX Improvements
- **#1** Prev/Next hole navigation buttons replacing selectbox — displays "Hål 5 / 18"
- **#2** 3-column score layout `[1,2,1]` with scorecard preview in collapsed expander
- **#3** AI strategy panel "📻 Team Radio — Strategy Request" collapsed by default

### Infrastructure
- `.streamlit/config.toml`: `gatherUsageStats=false`, `headless=true`, `runOnSave=false`
- `secrets.toml` added to `.gitignore`
- Playwright MCP configured and working locally

### Supabase Migration
- 6 tables created and seeded (see table above)
- **bcrypt PIN system**: `_check_pin()`, `_sb_get_user()`, `_sb_update_pin()` — PIN reset forced on first login via placeholder hash detection
- **Full gspread → Supabase replacement**:
  - `load_data_from_sheet()` → `load_data_from_supabase()`
  - `save_to_sheet()` ×7 → `save_inventory_to_sb()`
  - `append_rows_to_sheet()` ×2 → `append_history_to_sb()`
  - `add_course_to_sheet()` ×2 → `add_course_to_sb()`
  - `hard_reset_courses()` → `hard_reset_courses_sb()`
  - HQ create/delete user: gspread `ws.append_row/find/delete_rows` → `_sb_insert` / `_sb_delete`
- Course name mapping fixed in `MASTER_COURSES` — 42 previously skipped `hole_scores` resolved
- Guest profiles (Jenny, Isabella Dreyer, etc.) municipality updated directly in Supabase dashboard

---

## Known Issues

- **All 9 users have `DEFAULT_PIN_HASH`** — PIN reset flow triggers automatically on each user's first login. This is expected behaviour; not a bug.
- **Playwright cannot test Streamlit Cloud** — org auth enforced at infrastructure level. E2E tests limited to local dev only.
- **Hole data is placeholder** — All 23 courses seeded with `(100 m, par 3, Rak)` per hole. Real values need entry via a hole editor (not yet built).
- **`_disc_id` not populated on new inventory rows** — New rows added via `st.data_editor` get inserted correctly by `save_inventory_to_sb()` but won't carry a UUID in the in-memory df until next `load_data_from_supabase()` (next page load or cache clear).

---

## Next Session Priorities

1. Deploy updated app to Streamlit Cloud — add `service_role_key` in the Streamlit Cloud Secrets dashboard
2. UAT with Jenny — full round simulation end-to-end
3. Test PIN reset flow with all 9 users
4. Explore ruflo hive-mind for Dreyer Council integration
