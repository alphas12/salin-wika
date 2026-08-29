

export const index = 0;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_layout.svelte.js')).default;
export const imports = ["_app/immutable/nodes/0.D6NFlBkO.js","_app/immutable/chunks/CFeOzAGP.js","_app/immutable/chunks/BpZsZxJs.js"];
export const stylesheets = ["_app/immutable/assets/0.C2Cr-e2M.css"];
export const fonts = [];
