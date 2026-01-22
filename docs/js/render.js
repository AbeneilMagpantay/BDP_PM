/**
 * BDP Premier Dashboard - Render Module
 * Handles rendering of bets table, history, and filters
 */

import { getData, getHistory } from './api.js';
import { formatEV, formatOdds, formatDateTime, formatHistoryDate, getSportName } from './utils.js';

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

    headerRow.style.gridTemplateColumns = "1fr 2fr 1.5fr 1fr 0.8fr 0.8fr 0.8fr 0.8fr";
    headerRow.innerHTML = `
        <div>Date</div>
        <div>Match</div>
        <div>Pick</div>
        <div>Book</div>
        <div>Odds</div>
        <div>EV</div>
        <div>Model</div>
        <div>Implied</div>
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
        const impliedProb = e.implied_prob > 1 ? e.implied_prob : (e.implied_prob * 100).toFixed(1);
        const modelProb = e.model_prob ? (e.model_prob * 100).toFixed(1) : '??';
        const pick = e.bet_team || e.bet_on || 'Unknown';
        const matchup = e.matchup || `${e.away_team} @ ${e.home_team}`;
        const book = e.bookmaker || 'N/A';
        const { date: dateStr, time: timeStr } = formatDateTime(e.commence_time);

        return `
            <div class="bet-row" style="grid-template-columns: 1fr 2fr 1.5fr 1fr 0.8fr 0.8fr 0.8fr 0.8fr;">
                <div class="bet-date">
                    <div>${dateStr}</div>
                    <div style="font-size: 11px; opacity: 0.7; margin-top: 2px;">${timeStr}</div>
                </div>
                <div class="bet-match">${matchup}</div>
                <div><span class="bet-pick">${pick}</span></div>
                <div class="bet-book">${book}</div>
                <div class="bet-odds">${formatOdds(e.odds)}</div>
                <div class="bet-ev">${formatEV(e.ev)}</div>
                <div class="bet-model-prob" style="color: var(--accent-green); font-weight: 600;">${modelProb}%</div>
                <div class="bet-prob" style="opacity: 0.7;">${impliedProb}%</div>
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
    headerRow.style.gridTemplateColumns = "0.8fr 1.2fr 2fr 1.5fr 0.8fr 0.8fr 0.8fr 0.8fr";
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
        <div>Model</div>
        <div>Odds</div>
        <div>Result</div>
    `;

    document.getElementById('sport-filter-dropdown').addEventListener('change', (e) => {
        renderHistory(e.target.value);
    });

    list.innerHTML = gradedBets.map(b => {
        const evDisplay = formatEV(b.ev);
        const resultClass = b.result === 'WON' ? 'result-won' : 'result-lost';
        const sportName = getSportName(b.sport);

        // Calculate model_prob from EV and odds if missing
        // Formula: EV = (Model_Prob × Decimal_Odds) - 1
        // So: Model_Prob = (EV/100 + 1) / Decimal_Odds
        let modelProb = b.model_prob;
        if (!modelProb && b.ev && b.odds) {
            const odds = b.odds;
            const decimalOdds = odds > 0 ? (odds / 100) + 1 : (100 / Math.abs(odds)) + 1;
            const ev = b.ev / 100; // Convert from percentage
            modelProb = (ev + 1) / decimalOdds;
        }
        const modelProbDisplay = modelProb ? (modelProb * 100).toFixed(0) + '%' : '--';

        return `
            <div class="bet-row" style="grid-template-columns: 0.8fr 1.2fr 2fr 1.5fr 0.8fr 0.8fr 0.8fr 0.8fr;">
                <div style="font-size: 12px; font-weight: 600; color: var(--text-muted);">${sportName}</div>
                <div class="bet-date" style="font-size: 12px;">${formatHistoryDate(b.date)}</div>
                <div class="bet-match" style="font-size: 12px;">${b.match}</div>
                <div><span class="bet-pick bet-pick-history">${b.pick}</span></div>
                <div class="bet-ev" style="font-size:12px">${evDisplay}</div>
                <div class="bet-model-prob" style="font-size: 12px; color: var(--accent-green);">${modelProbDisplay}</div>
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
            t1: "AI-Powered Predictions",
            d1: "Our system uses machine learning models trained on years of historical data for NFL, NBA, and Soccer. Models are retrained daily to adapt to changing team performance.",
            t2: "Edge Detection",
            d2: "Scans 40+ bookmakers in real-time to find bets where our predicted win probability is higher than the bookmaker's odds suggest—these are your value opportunities."
        },
        nfl: {
            t1: "NFL Data Analysis",
            d1: "Analyzes play-by-play data from the last 4 NFL seasons. Tracks key metrics like yards per play, turnovers, and scoring efficiency to predict game outcomes.",
            t2: "Model Performance",
            d2: "Our NFL model achieves 84% accuracy on test games with 88% cross-validated accuracy. Key factors: offensive efficiency and turnover differential."
        },
        nba: {
            t1: "NBA Live Tracking",
            d1: "Monitors player availability, rest days, and team form. Detects value when star players are questionable or teams are on back-to-back games.",
            t2: "Model Performance",
            d2: "Our NBA model achieves 90% accuracy on test games. Analyzes shooting percentages, rebounds, assists, and turnovers to predict winners."
        },
        soccer: {
            t1: "Premier League Focus",
            d1: "Tracks English Premier League matches with odds from major bookmakers. Focuses on 1X2 (win/draw/lose) and over/under markets.",
            t2: "Model Performance",
            d2: "Our Soccer model uses team strength ratings and recent form. Filters for high-confidence picks with edges above 4% to improve win rate."
        },
        history: {
            t1: "Full Transparency",
            d1: "Every bet recommended by the AI is recorded here. Track our wins, losses, and profit over time to verify the system's long-term edge.",
            t2: "Daily Updates",
            d2: "Results are graded automatically after games complete. Win rate and profit stats update daily so you can see real performance—not just cherry-picked wins."
        }
    };

    const content = intelContent[sport] || intelContent.all;
    document.getElementById('intel-title-1').textContent = content.t1;
    document.getElementById('intel-desc-1').textContent = content.d1;
    document.getElementById('intel-title-2').textContent = content.t2;
    document.getElementById('intel-desc-2').textContent = content.d2;
}
