/**
 * BDP Premier Dashboard - API Module
 * Handles data fetching from JSON files
 */

// Global data stores
let predictionsData = null;
let betHistory = [];

/**
 * Fetch predictions data
 * @returns {Promise<Object>}
 */
export async function fetchPredictions() {
    const res = await fetch('data/predictions.json');
    predictionsData = await res.json();
    return predictionsData;
}

/**
 * Fetch bet history
 * @returns {Promise<Array>}
 */
export async function fetchHistory() {
    try {
        const res = await fetch('data/history.json');
        betHistory = await res.json();
    } catch (e) {
        console.warn("Could not load history.json", e);
        betHistory = [];
    }
    return betHistory;
}

/**
 * Get predictions data
 * @returns {Object}
 */
export function getData() {
    return predictionsData;
}

/**
 * Get bet history
 * @returns {Array}
 */
export function getHistory() {
    return betHistory;
}
