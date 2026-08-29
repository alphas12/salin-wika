// AI Amended: Small fetch wrapper for the FastAPI backend. Reads the base
// URL from SvelteKit public env and falls back to local development.
import { env } from '$env/dynamic/public';

const BASE_URL = env.PUBLIC_API_BASE_URL || 'http://localhost:8000';

/** @returns {Promise<Array<{name: string, bidirectional: boolean, peeky: boolean, bleu: number|null}>>} */
export async function fetchModels() {
  const res = await fetch(`${BASE_URL}/models`);
  if (!res.ok) {
    throw new Error(`Failed to load models (${res.status})`);
  }
  return res.json();
}

/** @returns {Promise<Array<{source_text: string, target_text: string}>>} */
export async function fetchCorpusSamples() {
  const res = await fetch(`${BASE_URL}/corpus/samples?limit=6`);
  if (!res.ok) {
    throw new Error(`Failed to load corpus samples (${res.status})`);
  }
  return res.json();
}

/** @returns {Promise<{status: string}>} */
export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) {
    throw new Error(`Backend health check failed (${res.status})`);
  }
  return res.json();
}

/**
 * @param {string} modelName
 * @param {string} text
 * @returns {Promise<{model_name: string, source_text: string, translated_text: string}>}
 */
export async function translateText(modelName, text) {
  const res = await fetch(`${BASE_URL}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName, text })
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Translation failed (${res.status})`);
  }

  return res.json();
}
