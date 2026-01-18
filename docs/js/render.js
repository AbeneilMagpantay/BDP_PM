/**
 * BDP Premier Dashboard - Render Module
 * Handles rendering of bets table, history, and filters
 */

import { getData, getHistory } from './api.js';
import { formatEV, formatKelly, formatOdds, formatDateTime, formatHistoryDate, getSportName } from './utils.js';

// Current history filter state
let historyFilter = 'all';

/**
 * Render sport filter tabs
 */
export function renderFilters() {
    const data = getData();
    const betHistory = getHistory();
    const container = document.getElementById('sport-tabs-container');

    // Count edges per sport
    const counts = { all: 0, nfl: 0, nba: 0, soccer: 0 };

    for (const [key, sport] of Object.entries(data.sports || {})) {
        const count = (sport && sport.edges) ? sport.edges.length : 0;
        counts[key] = count;
        counts.all += count;
    }

    const buttons = [
        { key: 'all', label: 'All', count: counts.all },
        { key: 'nfl', label: 'NFL', count: counts.nfl },
        { key: 'nba', label: 'NBA', count: counts.nba },
        { key: 'soccer', label: 'Soccer', count: counts.soccer },
        { key: 'history', label: 'History', count: betHistory.filter(b => b.result === 'WON' || b.result === 'LOST').length }
    ];

    container.innerHTML = buttons.map((b, i) => {
        const countText = b.count !== null && b.count > 0 ? ` (${b.count})` : '';
        return `<button class="filter-btn ${i === 0 ? 'active' : ''}" data-sport="${b.key}">${b.label}${countText}</button>`;
    }).join('');

    // Add click handlers
    container.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            container.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const sport = btn.dataset.sport;
            updateIntel(sport);
            if (sport === 'history') {
                renderHistory();
            } else {
                renderBets(sport);
            }
        });
    });
}

/**
 * Render value bets table
 * @param {string} filter - Sport filter (all, nfl, nba, soccer)
 */
export function renderBets(filter) {
    const data = getData();
    const list = document.getElementById('bets-list');
    const headerRow = document.getElementById('table-header-row');
    const avgEdgeEl = document.getElementById('avg-edge');
    const bestEdgeEl = document.getElementById('best-edge');

    // Restore stat card visibility
    const avgEdgeContainer = avgEdgeEl.parentElement.parentElement;
    const bestEdgeContainer = bestEdgeEl.parentElement.parentElement;
    avgEdgeContainer.classList.remove('hidden-stat');
    bestEdgeContainer.classList.remove('hidden-stat');

    // Reset Stat Titles
    avgEdgeEl.parentElement.querySelector('p').textContent = "Avg. Edge";
    avgEdgeEl.style.color = 'var(--accent-green)';
    bestEdgeEl.parentElement.querySelector('p').textContent = "Top Opportunity";

    // Reset UI for Value Bets
    document.getElementById('table-title-text').textContent = 'Latest Value Bets';

    headerRow.style.gridTemplateColumns = "1fr 2fr 1.5fr 1fr 0.7fr 0.7fr 0.7fr 0.7fr";
    headerRow.innerHTML = `
        <div>Date</div>
        <div>Match</div>
        <div>Pick</div>
        <div>Book</div>
        <div>Odds</div>
        <div>EV</div>
        <div>Kelly</div>
        <div>Prob</div>
    `;

    let allEdges = [];

    for (const [key, sport] of Object.entries(data.sports || {})) {
        if (filter !== 'all' && key !== filter) continue;
        if (sport && sport.edges) {
            sport.edges.forEach(e => allEdges.push({ ...e, sport_key: key }));
        }
    }

    allEdges.sort((a, b) => {
        const timeA = new Date(a.commence_time).getTime();
        const timeB = new Date(b.commence_time).getTime();
        if (timeA !== timeB) return timeA - timeB;
        return b.ev - a.ev;
    });

    // Calculate Stats
    if (allEdges.length > 0) {
        const totalEv = allEdges.reduce((sum, e) => sum + e.ev, 0);
        const avgEv = totalEv / allEdges.length;
        avgEdgeEl.textContent = '+' + avgEv.toFixed(1) + '%';

        const maxEv = Math.max(...allEdges.map(e => e.ev));
        bestEdgeEl.textContent = '+' + maxEv.toFixed(1) + '%';
        bestEdgeEl.classList.add('text-green');
    } else {
        avgEdgeEl.textContent = '--%';
        bestEdgeEl.textContent = '--%';
    }

    if (allEdges.length === 0) {
        list.innerHTML = `
            <div class="empty-state">
                <i class="ph ph-empty"></i>
                <p>No value bets found for this filter.</p>
            </div>`;
        return;
    }

    list.innerHTML = allEdges.map(e => {
        const prob = e.implied_prob > 1 ? e.implied_prob : (e.implied_prob * 100).toFixed(1);
        const pick = e.bet_team || e.bet_on || 'Unknown';
        const matchup = e.matchup || `${e.away_team} @ ${e.home_team}`;
        const book = e.bookmaker || 'N/A';
        const kelly = formatKelly(e.kelly_bet);
        const { date: dateStr, time: timeStr } = formatDateTime(e.commence_time);

        return `
            <div class="bet-row" style="grid-template-columns: 1fr 2fr 1.5fr 1fr 0.7fr 0.7fr 0.7fr 0.7fr;">
                <div class="bet-date">
                    <div>${dateStr}</div>
                    <div style="font-size: 11px; opacity: 0.7; margin-top: 2px;">${timeStr}</div>
                </div>
                <div class="bet-match">${matchup}</div>
                <div><span class="bet-pick">${pick}</span></div>
                <div class="bet-book">${book}</div>
                <div class="bet-odds">${formatOdds(e.odds)}</div>
                <div class="bet-ev">${formatEV(e.ev)}</div>
                <div style="color: var(--accent-blue);">${kelly}</div>
                <div class="bet-prob">${prob}%</div>
            </div>
        `;
    }).join('');
}

/**
 * Render betting history
 * @param {string} sportFilter - Sport filter (all, nfl, nba, soccer)
 */
export function renderHistory(sportFilter) {
    if (sportFilter !== undefined) historyFilter = sportFilter;

    const betHistory = getHistory();
    const list = document.getElementById('bets-list');
    const headerRow = document.getElementById('table-header-row');

    // Only show graded bets
    let gradedBets = betHistory
        .filter(b => b.result === 'WON' || b.result === 'LOST')
        .sort((a, b) => new Date(b.date) - new Date(a.date));

    // Apply sport filter
    if (historyFilter !== 'all') {
        gradedBets = gradedBets.filter(b => b.sport === historyFilter);
    }

    // Update Stats for History
    const avgEdgeEl = document.getElementById('avg-edge');
    avgEdgeEl.parentElement.querySelector('p').textContent = "Net Profit";
    const totalProfit = gradedBets.reduce((sum, b) => sum + (b.profit || 0), 0);
    avgEdgeEl.textContent = (totalProfit >= 0 ? '+' : '') + '$' + totalProfit.toFixed(2);
    avgEdgeEl.style.color = totalProfit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';

    const bestEdgeEl = document.getElementById('best-edge');
    bestEdgeEl.parentElement.querySelector('p').textContent = "Total Bets";
    bestEdgeEl.textContent = gradedBets.length;
    bestEdgeEl.classList.remove('text-green');

    document.getElementById('table-title-text').textContent = "Betting History";

    const bestEdgeContainer = bestEdgeEl.parentElement.parentElement;
    bestEdgeContainer.classList.add('hidden-stat');

    // Update Header with dropdown filter
    headerRow.style.gridTemplateColumns = "0.8fr 1fr 2fr 1.5fr 0.8fr 0.8fr 0.7fr 0.8fr";
    headerRow.innerHTML = `
        <div>
            <select id="sport-filter-dropdown" class="sport-filter-dropdown">
                <option value="all" ${historyFilter === 'all' ? 'selected' : ''}>SPORT</option>
                <option value="nfl" ${historyFilter === 'nfl' ? 'selected' : ''}>NFL</option>
                <option value="nba" ${historyFilter === 'nba' ? 'selected' : ''}>NBA</option>
                <option value="soccer" ${historyFilter === 'soccer' ? 'selected' : ''}>SOCCER</option>
            </select>
        </div>
        <div>Date</div>
        <div>Match</div>
        <div>Pick</div>
        <div>EV</div>
        <div>Kelly</div>
        <div>Odds</div>
        <div>Result</div>
    `;

    document.getElementById('sport-filter-dropdown').addEventListener('change', (e) => {
        renderHistory(e.target.value);
    });

    list.innerHTML = gradedBets.map(b => {
        const evDisplay = formatEV(b.ev);
        const kellyDisplay = formatKelly(b.kelly);
        const resultClass = b.result === 'WON' ? 'result-won' : 'result-lost';
        const sportName = getSportName(b.sport);

        return `
            <div class="bet-row" style="grid-template-columns: 0.7fr 1fr 2fr 1.5fr 0.8fr 0.8fr 0.7fr 0.8fr;">
                <div style="font-size: 12px; font-weight: 600; color: var(--text-muted);">${sportName}</div>
                <div class="bet-date" style="font-size: 12px;">${formatHistoryDate(b.date)}</div>
                <div class="bet-match" style="font-size: 12px;">${b.match}</div>
                <div><span class="bet-pick bet-pick-history">${b.pick}</span></div>
                <div class="bet-ev" style="font-size:12px">${evDisplay}</div>
                <div style="font-size:12px; color: var(--accent-blue);">${kellyDisplay}</div>
                <div class="bet-odds" style="font-weight:600; font-size: 12px;">${b.odds}</div>
                <div class="${resultClass}">${b.result}</div>
            </div>
        `;
    }).join('');
}

/**
 * Update intel sidebar content
 * @param {string} sport - Sport key
 */
function updateIntel(sport) {
    const intelContent = {
        all: {
            t1: "Multi-Sport AI Ensemble",
            d1: "Aggregates predictions from specialized XGBoost models for NFL, NBA, and Soccer. Retrained daily (2015-2025 datasets) on new data to adapt to shifting team form and injuries.",
            t2: "EV Engine & Discovery",
            d2: "Scans 40+ global bookmakers in real-time to find market inefficiencies where the AI's win probability significantly exceeds the implied bookmaker odds."
        },
        nfl: {
            t1: "NFL Market Feed",
            d1: "Ingests real-time spread and moneyline movements. Focuses on key number disagreements (3, 7) and significant lineup-change impacts.",
            t2: "XGBoost NFL Model",
            d2: "Trained on 10 years (2015-2025) of play-by-play statistics. Key features: Yards Per Play Differential, Turnover Margin, and DVOA efficiency metrics. (80% Historical Accuracy)"
        },
        nba: {
            t1: "NBA Odds Stream",
            d1: "Monitors rapid line movement and player prop availability. Detects value opportunities when impact players are questioned (GTD) or ruled out.",
            t2: "NBA Pace & Efficiency",
            d2: "Trained on 2020-2025 Season Data. Evaluates Pace (possessions/48m), Offensive Rating, and Rest Days. Identifies fatigue spots and mismatches. (90% Win Rate on Values >5%)"
        },
        soccer: {
            t1: "EPL & Euro Leagues",
            d1: "Scrapes odds for English Premier League. Focuses on 1x2, Asian Handicap, and Over/Under markets where liquidity is highest.",
            t2: "Smart Form Model",
            d2: "Trained on 2018-2025 Match Data. Incorporates 5-game rolling Form Momentum and H2H history. Filters for >4% edge to ensure high confidence. (62% Accuracy on PL Matches)"
        },
        history: {
            t1: "Historical Tracking",
            d1: "A complete record of all AI-recommended bets. Transparency is key to validating the model's long-term edge and refining future predictions.",
            t2: "Performance Audit",
            d2: "Win rate and ROI are recalculated daily. The system learns from past losses to refine edge thresholds and improve future selection."
        }
    };

    const content = intelContent[sport] || intelContent.all;
    document.getElementById('intel-title-1').textContent = content.t1;
    document.getElementById('intel-desc-1').textContent = content.d1;
    document.getElementById('intel-title-2').textContent = content.t2;
    document.getElementById('intel-desc-2').textContent = content.d2;
}
