/**
 * BDP Premier Dashboard - Utilities Module
 * Helper functions for formatting and display
 */

/**
 * Format time ago from date string
 * @param {string} dateString
 * @returns {string}
 */
export function timeAgo(dateString) {
    if (!dateString.endsWith('Z') && !dateString.includes('+')) {
        dateString += 'Z';
    }
    const date = new Date(dateString);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    let interval = seconds / 3600;
    if (interval > 1) return Math.floor(interval) + " hours ago";
    interval = seconds / 60;
    if (interval > 1) return Math.floor(interval) + " mins ago";
    return "just now";
}

/**
 * Format odds with + sign for positive
 * @param {number} odds
 * @returns {string}
 */
export function formatOdds(odds) {
    return odds > 0 ? '+' + odds : String(odds);
}

/**
 * Format EV with % sign
 * @param {number} ev
 * @returns {string}
 */
export function formatEV(ev) {
    if (ev == null) return '-';
    return '+' + Number(ev).toFixed(2) + '%';
}

/**
 * Format Kelly with % sign
 * @param {number} kelly
 * @returns {string}
 */
export function formatKelly(kelly) {
    if (kelly == null || kelly <= 0) return '-';
    return Number(kelly).toFixed(2) + '%';
}

/**
 * Format date for display
 * @param {string} dateString
 * @returns {Object} { date, time }
 */
export function formatDateTime(dateString) {
    const gameDate = new Date(dateString);
    return {
        date: gameDate.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' }),
        time: gameDate.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' })
    };
}

/**
 * Format date for history
 * @param {string} dateString
 * @returns {string}
 */
export function formatHistoryDate(dateString) {
    return new Date(dateString).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

/**
 * Calculate Kelly percentage from EV and odds
 * Kelly % = (EV/100) / (odds/100 - 1) for positive odds
 * @param {number} ev - Expected value percentage
 * @param {number} odds - American odds
 * @returns {number} Kelly percentage
 */
export function calculateKelly(ev, odds) {
    if (ev == null || odds == null) return null;

    // Convert American odds to decimal
    let decimalOdds;
    if (odds > 0) {
        decimalOdds = (odds / 100) + 1;
    } else {
        decimalOdds = (100 / Math.abs(odds)) + 1;
    }

    // Estimate probability from EV (simplified: assume fair odds + EV edge)
    const impliedProb = 1 / decimalOdds;
    const ourProb = impliedProb + (ev / 100);

    // Kelly formula: (bp - q) / b where b = decimal odds - 1, p = our prob, q = 1 - p
    const b = decimalOdds - 1;
    const p = Math.min(ourProb, 0.99); // Cap probability
    const q = 1 - p;

    const kelly = ((b * p) - q) / b;
    return Math.max(0, kelly * 100); // Return as percentage
}

/**
 * Get sport display name
 * @param {string} sportKey
 * @returns {string}
 */
export function getSportName(sportKey) {
    const names = { 'nfl': 'NFL', 'nba': 'NBA', 'soccer': 'Soccer' };
    return names[sportKey] || '-';
}
