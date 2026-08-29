<script>
  import { onMount } from 'svelte';
  import { fetchCorpusSamples, fetchHealth, fetchModels, translateText } from '$lib/api.js';
  import ModelPicker from '$lib/components/ModelPicker.svelte';
  import TranslateBox from '$lib/components/TranslateBox.svelte';
  import ResultCard from '$lib/components/ResultCard.svelte';

  /** @type {Array<{name: string, bidirectional: boolean, peeky: boolean, bleu: number|null}>} */
  let models = [];
  let selectedModel = '';
  let inputText = '';
  let outputText = '';
  let loading = false;
  let error = '';
  let modelsError = '';
  let backendStatus = 'checking';
  let backendMessage = 'Connecting to the FastAPI backend...';
  let samples = [
    'Maayong buntag.',
    'Asa ka paingon?',
    'Palihog tabangi ko.'
  ];
  let corpusMessage = 'Loading corpus examples...';

  onMount(async () => {
    const [healthResult, modelsResult, samplesResult] = await Promise.allSettled([
      fetchHealth(),
      fetchModels(),
      fetchCorpusSamples()
    ]);

    if (healthResult.status === 'fulfilled') {
      backendStatus = 'online';
      backendMessage = 'FastAPI backend is online.';
    } else {
      backendStatus = 'offline';
      backendMessage = healthResult.reason instanceof Error ? healthResult.reason.message : String(healthResult.reason);
    }

    if (modelsResult.status === 'fulfilled') {
      models = modelsResult.value;
      if (models.length) selectedModel = models[0].name;
    } else {
      modelsError = modelsResult.reason instanceof Error ? modelsResult.reason.message : String(modelsResult.reason);
    }

    if (samplesResult.status === 'fulfilled' && samplesResult.value.length) {
      samples = samplesResult.value.map((sample) => sample.source_text);
      corpusMessage = `Loaded ${samplesResult.value.length} corpus examples from cebtag_bible_31k.csv.`;
    } else if (samplesResult.status === 'rejected') {
      corpusMessage = samplesResult.reason instanceof Error ? samplesResult.reason.message : String(samplesResult.reason);
    } else {
      corpusMessage = 'Corpus examples are unavailable right now.';
    }
  });

  function useSample(text) {
    inputText = text;
  }

  $: canTranslate = models.length > 0 && !!selectedModel;

  /** @param {CustomEvent<string>} event */
  async function handleTranslate(event) {
    const text = event.detail;

    if (!selectedModel) {
      error = 'No trained model is available yet. Add a run under results/ first.';
      return;
    }

    loading = true;
    error = '';
    outputText = '';

    try {
      const result = await translateText(selectedModel, text);
      outputText = result.translated_text;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }
</script>

<svelte:head>
  <title>SalinWika | Cebuano to Tagalog</title>
  <meta
    name="description"
    content="SalinWika translates Cebuano text to Tagalog using a FastAPI backend and Svelte frontend."
  />
</svelte:head>

<main class="page">
  <section class="hero">
    <div class="hero__copy">
      <p class="hero__eyebrow">SalinWika</p>
      <h1>Cebuano to Tagalog translation, tuned for your trained FastAPI model.</h1>
      <p class="hero__tagline">
        Paste Cebuano text, pick the run your partner trained, and translate in a single
        screen.
      </p>
    </div>

    <div class="hero__status" data-state={backendStatus}>
      <span class="hero__status-label">Backend</span>
      <strong>{backendStatus === 'online' ? 'Connected' : backendStatus === 'checking' ? 'Checking' : 'Offline'}</strong>
      <p>{backendMessage}</p>
    </div>
  </section>

  <section class="quick-actions">
    <span class="quick-actions__label">Corpus input</span>
    <p class="quick-actions__message">{corpusMessage}</p>
    <div class="quick-actions__buttons">
      {#each samples as sample}
        <button class="quick-actions__button" on:click={() => useSample(sample)}>{sample}</button>
      {/each}
    </div>
  </section>

  {#if modelsError}
    <p class="page__error">Could not load models: {modelsError}</p>
  {/if}

  {#if models.length}
    <div class="page__controls">
      <ModelPicker {models} bind:selected={selectedModel} />
    </div>
  {:else}
    <p class="page__status">No trained runs were found yet. Put your partner's model under results/ to enable translation.</p>
  {/if}

  <section class="panels">
    <TranslateBox bind:value={inputText} {loading} on:translate={handleTranslate} />
    <div class="panels__divider" aria-hidden="true"></div>
    <ResultCard text={outputText} {loading} {error} />
  </section>
</main>

<style>
  .page {
    max-width: 1180px;
    margin: 0 auto;
    padding: 4rem 1.5rem 4.5rem;
  }

  .hero {
    display: grid;
    grid-template-columns: minmax(0, 1.7fr) minmax(260px, 0.9fr);
    gap: 1.25rem;
    align-items: stretch;
    margin-bottom: 1.25rem;
  }

  .hero__copy,
  .hero__status,
  .quick-actions {
    background: rgba(17, 21, 28, 0.52);
    border: 1px solid rgba(42, 53, 65, 0.95);
    backdrop-filter: blur(14px);
    border-radius: 24px;
    box-shadow: 0 28px 80px rgba(0, 0, 0, 0.22);
  }

  .hero__copy {
    padding: 1.5rem;
  }

  .hero__eyebrow,
  .quick-actions__label,
  .hero__status-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--color-accent-teal);
  }

  .hero__copy h1 {
    margin: 0.45rem 0 0;
    font-family: var(--font-display);
    font-size: clamp(2.35rem, 4.6vw, 4.6rem);
    line-height: 0.98;
    letter-spacing: -0.04em;
    max-width: 12ch;
  }

  .hero__tagline {
    margin: 1rem 0 0;
    max-width: 58ch;
    color: var(--color-text-muted);
    line-height: 1.7;
  }

  .hero__status {
    padding: 1.35rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 0.9rem;
  }

  .hero__status strong {
    font-size: 1.35rem;
  }

  .hero__status p {
    margin: 0;
    color: var(--color-text-muted);
    line-height: 1.55;
  }

  .hero__status[data-state='online'] {
    border-color: rgba(74, 155, 142, 0.45);
  }

  .hero__status[data-state='offline'] {
    border-color: rgba(224, 132, 122, 0.45);
  }

  .quick-actions {
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
  }

  .quick-actions__message {
    margin: 0.4rem 0 0;
    color: var(--color-text-muted);
    line-height: 1.6;
  }

  .quick-actions__buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-top: 0.75rem;
  }

  .quick-actions__button {
    border: 1px solid rgba(74, 155, 142, 0.3);
    background: rgba(26, 34, 44, 0.92);
    color: var(--color-text);
    border-radius: 999px;
    padding: 0.58rem 0.9rem;
    cursor: pointer;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }

  .quick-actions__button:hover {
    transform: translateY(-1px);
    border-color: rgba(212, 162, 76, 0.55);
  }

  .page__controls {
    margin-bottom: 1.5rem;
    max-width: 100%;
  }

  .page__error {
    color: var(--color-error);
  }

  .page__status {
    color: var(--color-text-muted);
  }

  .panels {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 1.25rem;
    align-items: start;
  }

  .panels__divider {
    align-self: stretch;
    width: 1px;
    background-image: repeating-linear-gradient(
      to bottom,
      var(--color-accent-gold) 0 6px,
      transparent 6px 14px
    );
  }

  @media (max-width: 720px) {
    .hero {
      grid-template-columns: 1fr;
    }

    .panels {
      grid-template-columns: 1fr;
    }

    .panels__divider {
      display: none;
    }
  }
</style>
