
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://kit.svelte.dev/docs/modules#$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://kit.svelte.dev/docs/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://kit.svelte.dev/docs/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://kit.svelte.dev/docs/modules#$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```bash
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const PATH: string;
	export const M2: string;
	export const CONDA_DEFAULT_ENV: string;
	export const CONDA_PYTHON_EXE: string;
	export const CONDA_PREFIX: string;
	export const AZURE_DEVOPS_EXT_PAT: string;
	export const COLORTERM: string;
	export const LOGNAME: string;
	export const HTTPLIB2_CA_CERTS: string;
	export const PWD: string;
	export const CURL_CA_BUNDLE: string;
	export const SHELL: string;
	export const ING_DIR: string;
	export const REQUESTS_CA_BUNDLE: string;
	export const NPM_CONFIG_PREFIX: string;
	export const OLDPWD: string;
	export const TMPDIR: string;
	export const XPC_FLAGS: string;
	export const M2_HOME: string;
	export const __CF_USER_TEXT_ENCODING: string;
	export const CONDA_PROMPT_MODIFIER: string;
	export const NODE_ENV: string;
	export const LC_CTYPE: string;
	export const GSETTINGS_SCHEMA_DIR_CONDA_BACKUP: string;
	export const MODULAR_HOME: string;
	export const FORCE_COLOR: string;
	export const ING_CA_PATH: string;
	export const CONDA_EXE: string;
	export const TERM: string;
	export const DEBUG_COLORS: string;
	export const COMMAND_MODE: string;
	export const NPM_AUTH: string;
	export const MAVEN_OPTS: string;
	export const npm_config_color: string;
	export const MOCHA_COLORS: string;
	export const _CE_M: string;
	export const NODE_EXTRA_CA_CERTS: string;
	export const XPC_SERVICE_NAME: string;
	export const CONDA_SHLVL: string;
	export const __CFBundleIdentifier: string;
	export const ING_ADO_TOKEN: string;
	export const SSL_CERT_FILE: string;
	export const ING_ADO_USER: string;
	export const ING_CA_DIR_PATH: string;
	export const USER: string;
	export const SSH_AUTH_SOCK: string;
	export const _CE_CONDA: string;
	export const GSETTINGS_SCHEMA_DIR: string;
	export const IDEA_INITIAL_DIRECTORY: string;
	export const HOME: string;
}

/**
 * Similar to [`$env/static/private`](https://kit.svelte.dev/docs/modules#$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://kit.svelte.dev/docs/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://kit.svelte.dev/docs/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://kit.svelte.dev/docs/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://kit.svelte.dev/docs/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		PATH: string;
		M2: string;
		CONDA_DEFAULT_ENV: string;
		CONDA_PYTHON_EXE: string;
		CONDA_PREFIX: string;
		AZURE_DEVOPS_EXT_PAT: string;
		COLORTERM: string;
		LOGNAME: string;
		HTTPLIB2_CA_CERTS: string;
		PWD: string;
		CURL_CA_BUNDLE: string;
		SHELL: string;
		ING_DIR: string;
		REQUESTS_CA_BUNDLE: string;
		NPM_CONFIG_PREFIX: string;
		OLDPWD: string;
		TMPDIR: string;
		XPC_FLAGS: string;
		M2_HOME: string;
		__CF_USER_TEXT_ENCODING: string;
		CONDA_PROMPT_MODIFIER: string;
		NODE_ENV: string;
		LC_CTYPE: string;
		GSETTINGS_SCHEMA_DIR_CONDA_BACKUP: string;
		MODULAR_HOME: string;
		FORCE_COLOR: string;
		ING_CA_PATH: string;
		CONDA_EXE: string;
		TERM: string;
		DEBUG_COLORS: string;
		COMMAND_MODE: string;
		NPM_AUTH: string;
		MAVEN_OPTS: string;
		npm_config_color: string;
		MOCHA_COLORS: string;
		_CE_M: string;
		NODE_EXTRA_CA_CERTS: string;
		XPC_SERVICE_NAME: string;
		CONDA_SHLVL: string;
		__CFBundleIdentifier: string;
		ING_ADO_TOKEN: string;
		SSL_CERT_FILE: string;
		ING_ADO_USER: string;
		ING_CA_DIR_PATH: string;
		USER: string;
		SSH_AUTH_SOCK: string;
		_CE_CONDA: string;
		GSETTINGS_SCHEMA_DIR: string;
		IDEA_INITIAL_DIRECTORY: string;
		HOME: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://kit.svelte.dev/docs/modules#$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://kit.svelte.dev/docs/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
