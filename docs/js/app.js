/**
 * BDP Premier Dashboard - Main Application
 * Entry point that initializes the dashboard
 */

import { fetchPredictions, fetchHistory, getData, getHistory } from './api.js';
import { renderFilters, renderBets } from './render.js';
import { timeAgo } from './utils.js';

/**
 * Initialize the dashboard
 */
async function init() {
    try {
        // Fetch data
        const data = await fetchPredictions();
        const betHistory = await fetchHistory();

        // Update timestamp
        document.getElementById('updated-at').textContent = 'Updated ' + timeAgo(data.generated_at);

        // Set record from history
        const wins = betHistory.filter(h => h.result === 'WON').length;
        const losses = betHistory.filter(h => h.result === 'LOST').length;
        document.getElementById('record').textContent = `${wins}-${losses}`;

        // Render UI
        renderFilters();
        renderBets('all');

    } catch (e) {
        console.error('Failed to load dashboard:', e);
    }
}

// Start the app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
