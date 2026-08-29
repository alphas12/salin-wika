export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.zG0BtWIp.js",app:"_app/immutable/entry/app.BqNZn9BO.js",imports:["_app/immutable/entry/start.zG0BtWIp.js","_app/immutable/chunks/g7LcFnlI.js","_app/immutable/chunks/CFeOzAGP.js","_app/immutable/entry/app.BqNZn9BO.js","_app/immutable/chunks/CFeOzAGP.js","_app/immutable/chunks/BpZsZxJs.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:true},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
