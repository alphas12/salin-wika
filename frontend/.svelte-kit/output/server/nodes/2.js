

export const index = 2;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_page.svelte.js')).default;
export const imports = ["_app/immutable/nodes/2.BqbYGmLB.js","_app/immutable/chunks/CFeOzAGP.js","_app/immutable/chunks/BpZsZxJs.js"];
export const stylesheets = ["_app/immutable/assets/2.A-7tdIw9.css"];
export const fonts = [];
