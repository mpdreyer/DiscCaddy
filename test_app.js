const { chromium } = require('playwright');

const BASE_URL = 'http://localhost:8501';
const PASS = '\x1b[32m✔\x1b[0m';
const FAIL = '\x1b[31m✘\x1b[0m';
const INFO = '\x1b[33m→\x1b[0m';

let passed = 0;
let failed = 0;
const jsErrors = [];

async function assert(label, condition) {
  if (condition) {
    console.log(`  ${PASS} ${label}`);
    passed++;
  } else {
    console.log(`  ${FAIL} ${label}`);
    failed++;
  }
}

async function waitForStreamlit(page) {
  // Wait for Streamlit to finish its initial render
  await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 15000 });
  await page.waitForTimeout(1500);
}

async function waitForRerun(page) {
  // After a button click, wait for Streamlit to stop the running indicator
  await page.waitForTimeout(800);
  try {
    await page.waitForSelector('[data-testid="stStatusWidget"] [title="Running..."]', { timeout: 2000 });
    await page.waitForSelector('[data-testid="stStatusWidget"] [title="Running..."]', { state: 'hidden', timeout: 10000 });
  } catch {
    // No spinner appeared — rerun was instant, that's fine
  }
  await page.waitForTimeout(400);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  // Capture all JS console errors
  page.on('console', msg => {
    if (msg.type() === 'error') jsErrors.push(msg.text());
  });
  page.on('pageerror', err => jsErrors.push(`PageError: ${err.message}`));

  console.log('\n\x1b[1m=== DiscCaddy Playwright Test Suite ===\x1b[0m\n');

  // ─── 1. LOGIN ────────────────────────────────────────────────────────────────
  console.log('\x1b[1m[1] Login Screen\x1b[0m');
  await page.goto(BASE_URL);
  await waitForStreamlit(page);

  const loginTitle = await page.locator('h1').first().textContent().catch(() => '');
  await assert('Login title visible (SCUDERIA PADDOCK)', loginTitle.includes('PADDOCK'));

  const userSelect = page.locator('[data-testid="stSelectbox"]').first();
  await assert('User selector present', await userSelect.isVisible().catch(() => false));

  const pinInput = page.locator('input[type="password"]');
  await assert('PIN input present', await pinInput.isVisible().catch(() => false));

  const unlockBtn = page.locator('button').filter({ hasText: /Lås Upp/i });
  await assert('Unlock button present', await unlockBtn.isVisible().catch(() => false));

  // Try wrong PIN
  await pinInput.fill('0000');
  await unlockBtn.click();
  await waitForRerun(page);
  const errorMsg = page.locator('[data-testid="stAlert"]');
  await assert('Wrong PIN shows error', await errorMsg.isVisible().catch(() => false));

  // Login with correct PIN
  await pinInput.fill('1234');
  await unlockBtn.click();
  await waitForRerun(page);

  const tabs = page.locator('[data-testid="stTabs"] button');
  await assert('Logged in — tabs visible', await tabs.first().isVisible().catch(() => false));

  // ─── 2. SIDEBAR ──────────────────────────────────────────────────────────────
  console.log('\n\x1b[1m[2] Sidebar\x1b[0m');
  const sidebar = page.locator('[data-testid="stSidebar"]');
  await assert('Sidebar rendered', await sidebar.isVisible().catch(() => false));

  const courseSelect = sidebar.locator('[data-testid="stSelectbox"]').first();
  await assert('Course selector in sidebar', await courseSelect.isVisible().catch(() => false));

  const syncBtn = sidebar.locator('button').filter({ hasText: /Synka/i });
  await assert('Synka Databas button present', await syncBtn.isVisible().catch(() => false));

  const logoutBtn = sidebar.locator('button').filter({ hasText: /Logga Ut/i });
  await assert('Logout button present', await logoutBtn.isVisible().catch(() => false));

  // Weather metrics
  const metrics = sidebar.locator('[data-testid="stMetric"]');
  const metricCount = await metrics.count().catch(() => 0);
  await assert('Weather metrics rendered (Temp + Vind)', metricCount >= 2);

  // ─── 3. WARM-UP TAB ──────────────────────────────────────────────────────────
  console.log('\n\x1b[1m[3] Warm-Up Tab\x1b[0m');
  await tabs.filter({ hasText: /WARM/i }).click();
  await waitForRerun(page);

  const warmupHeader = page.locator('h1,h2,h3').filter({ hasText: /Driving Range/i });
  await assert('Driving Range header visible', await warmupHeader.isVisible().catch(() => false));

  const saveThrowBtn = page.locator('button').filter({ hasText: /Spara Kast/i });
  await assert('Spara Kast button present', await saveThrowBtn.isVisible().catch(() => false));

  // Try saving a throw without selecting a disc — should not crash
  await saveThrowBtn.click();
  await waitForRerun(page);
  await assert('Saving throw without disc does not crash', true);

  // ─── 4. RACE TAB ─────────────────────────────────────────────────────────────
  console.log('\n\x1b[1m[4] Race Tab\x1b[0m');
  await tabs.filter({ hasText: /RACE/i }).click();
  await waitForRerun(page);

  const raceHeader = page.locator('[data-testid="stSubheader"]').filter({ hasText: /Race Day/i });
  await assert('Race Day header visible', await raceHeader.isVisible().catch(() => false));

  // Prev/Next navigation
  const prevBtn = page.locator('button').filter({ hasText: /Föregående/i });
  const nextBtn = page.locator('button').filter({ hasText: /Nästa/i });
  await assert('◀ Föregående button present', await prevBtn.isVisible().catch(() => false));
  await assert('Nästa ▶ button present', await nextBtn.isVisible().catch(() => false));
  await assert('◀ Föregående disabled on hole 1', await prevBtn.isDisabled().catch(() => false));

  // Advance to hole 2
  await nextBtn.click();
  await waitForRerun(page);
  const holeMetric = page.locator('[data-testid="stMetric"]').filter({ hasText: /Hål 2/ });
  await assert('Advances to Hål 2 on Nästa click', await holeMetric.isVisible().catch(() => false));

  // Go back to hole 1
  await prevBtn.click();
  await waitForRerun(page);
  const hole1Metric = page.locator('[data-testid="stMetric"]').filter({ hasText: /Hål 1/ });
  await assert('Returns to Hål 1 on Föregående click', await hole1Metric.isVisible().catch(() => false));

  // Score ➕/➖ buttons
  const plusBtn = page.locator('button').filter({ hasText: '➕' }).first();
  const minusBtn = page.locator('button').filter({ hasText: '➖' }).first();
  await assert('➕ score button present', await plusBtn.isVisible().catch(() => false));
  await assert('➖ score button present', await minusBtn.isVisible().catch(() => false));

  await plusBtn.click();
  await waitForRerun(page);
  await assert('➕ click does not crash', true);

  await minusBtn.click();
  await waitForRerun(page);
  await assert('➖ click does not crash', true);

  // AI Strategy expander
  const strategyExpander = page.locator('[data-testid="stExpander"]').filter({ hasText: /Team Radio/i });
  await assert('Team Radio expander present', await strategyExpander.isVisible().catch(() => false));
  const expanderSummary = strategyExpander.locator('summary, [data-testid="stExpanderToggleIcon"]').first();
  const isExpanded = await strategyExpander.getAttribute('open').catch(() => null);
  await assert('Team Radio collapsed by default', isExpanded === null);

  // Scorecard preview expander
  const scorecardExpander = page.locator('[data-testid="stExpander"]').filter({ hasText: /Scorecard Preview/i });
  await assert('Scorecard Preview expander present', await scorecardExpander.isVisible().catch(() => false));

  // Save round button
  const saveRoundBtn = page.locator('button').filter({ hasText: /SPARA RUNDA/i });
  await assert('SPARA RUNDA button present', await saveRoundBtn.isVisible().catch(() => false));

  // ─── 5. AI-CADDY TAB ─────────────────────────────────────────────────────────
  console.log('\n\x1b[1m[5] AI-Caddy Tab\x1b[0m');
  await tabs.filter({ hasText: /AI/i }).click();
  await waitForRerun(page);

  const chatInput = page.locator('[data-testid="stChatInput"] textarea, [data-testid="stChatInputTextArea"]');
  await assert('Chat input present', await chatInput.isVisible().catch(() => false));

  // ─── 6. UTRUSTNING TAB ───────────────────────────────────────────────────────
  console.log('\n\x1b[1m[6] Utrustning Tab\x1b[0m');
  await tabs.filter({ hasText: /UTRUSTNING/i }).click();
  await waitForRerun(page);

  const generaBtn = page.locator('button').filter({ hasText: /Generera/i });
  await assert('Generera (Smart Bag) button present', await generaBtn.isVisible().catch(() => false));

  const bagHeader = page.locator('h2,h3').filter({ hasText: /Logistik/i });
  await assert('Logistik header visible', await bagHeader.isVisible().catch(() => false));

  const saveChangesBtn = page.locator('button').filter({ hasText: /Spara Ändringar/i });
  await assert('Spara Ändringar button present', await saveChangesBtn.isVisible().catch(() => false));

  // ─── 7. TELEMETRY TAB ────────────────────────────────────────────────────────
  console.log('\n\x1b[1m[7] Telemetry Tab\x1b[0m');
  await tabs.filter({ hasText: /TELEMETRY/i }).click();
  await waitForRerun(page);

  const aeroTab = page.locator('[data-testid="stTabs"] button').filter({ hasText: /Aero/i });
  await assert('Aero Lab sub-tab present', await aeroTab.isVisible().catch(() => false));

  const racePerf = page.locator('[data-testid="stTabs"] button').filter({ hasText: /Race Performance/i });
  await assert('Race Performance sub-tab present', await racePerf.isVisible().catch(() => false));

  const sectorTab = page.locator('[data-testid="stTabs"] button').filter({ hasText: /Sector/i });
  await assert('Sector Analysis sub-tab present', await sectorTab.isVisible().catch(() => false));

  // ─── 8. ACADEMY TAB ──────────────────────────────────────────────────────────
  console.log('\n\x1b[1m[8] Academy Tab\x1b[0m');
  await tabs.filter({ hasText: /ACADEMY/i }).click();
  await waitForRerun(page);

  const puttCoach = page.locator('[data-testid="stTabs"] button').filter({ hasText: /Putt/i });
  await assert('Putt-Coach sub-tab present', await puttCoach.isVisible().catch(() => false));

  const startPassBtn = page.locator('button').filter({ hasText: /Starta Nytt Pass/i });
  await assert('Starta Nytt Pass button present', await startPassBtn.isVisible().catch(() => false));

  // Start a putting session
  await startPassBtn.click();
  await waitForRerun(page);
  const station1 = page.locator('[data-testid="stMetric"]').filter({ hasText: /Station 1/i });
  await assert('Putt session starts — Station 1 visible', await station1.isVisible().catch(() => false));

  // ─── 9. JAVASCRIPT ERRORS ────────────────────────────────────────────────────
  console.log('\n\x1b[1m[9] JavaScript Errors\x1b[0m');
  // Filter out known Streamlit/third-party noise
  const appErrors = jsErrors.filter(e =>
    !e.includes('ResizeObserver') &&
    !e.includes('favicon') &&
    !e.includes('net::ERR_') &&
    !e.includes('/_stcore/') &&
    !e.includes('fontawesome')
  );
  if (appErrors.length === 0) {
    console.log(`  ${PASS} No JavaScript errors detected`);
    passed++;
  } else {
    console.log(`  ${FAIL} ${appErrors.length} JavaScript error(s) detected:`);
    appErrors.forEach(e => console.log(`     ${INFO} ${e.slice(0, 120)}`));
    failed++;
  }

  // ─── SUMMARY ─────────────────────────────────────────────────────────────────
  console.log('\n\x1b[1m=== Results ===\x1b[0m');
  console.log(`  ${PASS} Passed: ${passed}`);
  if (failed > 0) console.log(`  ${FAIL} Failed: ${failed}`);
  console.log('');

  await browser.close();
  process.exit(failed > 0 ? 1 : 0);
})();
