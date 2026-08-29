<script>
  import { createEventDispatcher } from 'svelte';

  /** @type {string} */
  export let value = '';
  /** @type {boolean} */
  export let loading = false;

  const dispatch = createEventDispatcher();

  function submit() {
    if (!value.trim() || loading) return;
    dispatch('translate', value.trim());
  }

  /** @param {KeyboardEvent} event */
  function onKeydown(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') submit();
  }
</script>

<div class="box">
  <div class="box__header">
    <span class="box__eyebrow">Source</span>
    <span class="box__hint">Cebuano input</span>
  </div>
  <textarea
    bind:value
    on:keydown={onKeydown}
    placeholder="I-type ang Cebuano sentence here..."
    rows="8"
  ></textarea>
  <button class="box__submit" on:click={submit} disabled={loading || !value.trim()}>
    {loading ? 'Translating…' : 'Translate to Tagalog'}
  </button>
</div>

<style>
  .box {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 1rem;
    background: rgba(26, 34, 44, 0.92);
    border: 1px solid var(--color-border);
    border-radius: 20px;
    box-shadow: 0 22px 50px rgba(0, 0, 0, 0.22);
  }

  .box__header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.75rem;
  }

  .box__eyebrow {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--color-accent-teal);
  }

  .box__hint {
    color: var(--color-text-muted);
    font-size: 0.85rem;
  }

  textarea {
    background: linear-gradient(180deg, rgba(33, 43, 55, 0.98), rgba(26, 34, 44, 0.98));
    color: var(--color-text);
    border: 1px solid var(--color-border);
    border-radius: 16px;
    padding: 0.9rem;
    font-size: 1rem;
    line-height: 1.5;
    resize: vertical;
    min-height: 14rem;
  }

  .box__submit {
    align-self: flex-start;
    background: var(--color-accent-gold);
    color: #1a140a;
    border: none;
    border-radius: 999px;
    padding: 0.72rem 1.35rem;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
    transition: transform 0.15s ease, opacity 0.15s ease;
  }

  .box__submit:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .box__submit:not(:disabled):hover {
    transform: translateY(-1px);
  }
</style>
