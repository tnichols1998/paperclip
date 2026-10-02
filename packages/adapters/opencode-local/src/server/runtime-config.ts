import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { asBoolean } from "@paperclipai/adapter-utils/server-utils";

type PreparedOpenCodeRuntimeConfig = {
  env: Record<string, string>;
  notes: string[];
  cleanup: () => Promise<void>;
};

function resolveXdgConfigHome(env: Record<string, string>): string {
  return (
    (typeof env.XDG_CONFIG_HOME === "string" && env.XDG_CONFIG_HOME.trim()) ||
    (typeof process.env.XDG_CONFIG_HOME === "string" && process.env.XDG_CONFIG_HOME.trim()) ||
    path.join(os.homedir(), ".config")
  );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

type PluginPinResolution =
  | { kind: "pinned"; plugin: unknown[] }
  | { kind: "unresolvable"; reason: string };

const FILE_URL_PREFIX = "file://";
const NPM_PLUGIN_SPEC_RE = /^(@[a-z0-9][\w.-]*\/)?[a-z0-9][\w.-]*$/i;

async function pathExists(filepath: string): Promise<boolean> {
  try {
    await fs.lstat(filepath);
    return true;
  } catch {
    return false;
  }
}

async function pathIsSymlink(filepath: string): Promise<boolean> {
  try {
    return (await fs.lstat(filepath)).isSymbolicLink();
  } catch {
    return false;
  }
}

// Reject the path if it or any of its parents (up to and including the source
// config dir) is a symlink. Checking only the leaf would miss a symlinked
// intermediate directory that redirects the pin elsewhere.
async function pathHasSymlinkSegment(filepath: string, stopAt: string): Promise<boolean> {
  let current = filepath;
  const stop = path.resolve(stopAt);
  while (true) {
    if (await pathIsSymlink(current)) return true;
    if (current === stop) return false;
    const parent = path.dirname(current);
    if (parent === current) return false;
    current = parent;
  }
}

function pathIsInside(root: string, target: string): boolean {
  const relative = path.relative(root, target);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

// Resolve a single OpenCode plugin spec to a fixed, source-anchored absolute
// reference, or fail. Resolution happens against the SOURCE config dir (owned
// by the config owner) before any runtime-writable copy exists, so the pinned
// result never depends on files the run can edit. Symlink path segments are
// rejected: the anchor must be the real filesystem location of the plugin, so
// retargeting a symlink inside the copied tree cannot substitute code.
async function pinPluginEntry(
  entry: unknown,
  sourceConfigDir: string,
): Promise<{ ok: true; value: unknown } | { ok: false; reason: string }> {
  if (Array.isArray(entry)) {
    const [spec, ...rest] = entry as unknown[];
    const pinned = await pinPluginEntry(spec, sourceConfigDir);
    if (!pinned.ok) return pinned;
    return { ok: true, value: [pinned.value, ...rest] };
  }
  if (typeof entry !== "string" || entry.trim().length === 0) {
    return { ok: false, reason: "non-string plugin entry" };
  }
  const spec = entry.trim();
  if (spec.startsWith(FILE_URL_PREFIX)) {
    const filepath = spec.slice(FILE_URL_PREFIX.length);
    if (!path.isAbsolute(filepath)) {
      return { ok: false, reason: `non-absolute ${FILE_URL_PREFIX} spec ${spec}` };
    }
    if (await pathHasSymlinkSegment(filepath, path.parse(filepath).root)) {
      return { ok: false, reason: `symlinked plugin path ${filepath}` };
    }
    if (!(await pathExists(filepath))) {
      return { ok: false, reason: `missing plugin path ${filepath}` };
    }
    return { ok: true, value: spec };
  }
  if (!path.isAbsolute(spec) && spec.startsWith(".")) {
    const resolved = path.resolve(sourceConfigDir, spec);
    if (!pathIsInside(sourceConfigDir, resolved)) {
      return { ok: false, reason: `plugin spec ${spec} escapes the source config dir` };
    }
    if (await pathHasSymlinkSegment(resolved, path.parse(resolved).root)) {
      return { ok: false, reason: `symlinked plugin path ${resolved}` };
    }
    if (!(await pathExists(resolved))) {
      return { ok: false, reason: `missing plugin path ${resolved}` };
    }
    return { ok: true, value: `${FILE_URL_PREFIX}${resolved}` };
  }
  if (NPM_PLUGIN_SPEC_RE.test(spec)) {
    const packageDir = path.join(sourceConfigDir, "node_modules", spec);
    if (await pathHasSymlinkSegment(packageDir, sourceConfigDir)) {
      return { ok: false, reason: `symlinked plugin package ${spec}` };
    }
    if (!(await pathExists(packageDir))) {
      return { ok: false, reason: `plugin package ${spec} is not installed under the source config dir` };
    }
    return { ok: true, value: spec };
  }
  return { ok: false, reason: `unrecognized plugin spec ${spec}` };
}

// Pin the source config's `plugin` array to fixed, config-owner-anchored
// references that are injected into the runtime copy AFTER the copy lands.
// Without this the runtime-writable per-run config carries a `plugin` array
// the agent can edit to drop enforcement plugins before load.
async function resolvePinnedPluginArray(
  sourceConfig: Record<string, unknown>,
  sourceConfigDir: string,
): Promise<PluginPinResolution> {
  if (!("plugin" in sourceConfig)) {
    return { kind: "pinned", plugin: [] };
  }
  if (!Array.isArray(sourceConfig.plugin)) {
    return { kind: "unresolvable", reason: "source config `plugin` key is not an array" };
  }
  const pinned: unknown[] = [];
  for (const entry of sourceConfig.plugin) {
    const result = await pinPluginEntry(entry, sourceConfigDir);
    if (!result.ok) {
      return { kind: "unresolvable", reason: result.reason };
    }
    pinned.push(result.value);
  }
  return { kind: "pinned", plugin: pinned };
}

// Recursively replace {env:VAR} placeholders with the resolved value. Used to bake
// gateway provider secrets (e.g. the LLM-gateway virtual key) into opencode.json
// SERVER-SIDE, where the value is reliably present. OpenCode's own {env:...}
// resolution happens inside the (possibly sandboxed) run process, whose env
// plumbing is not guaranteed to carry the key to OpenCode's spawned server -- so
// we resolve it here. Unresolvable placeholders are left intact for OpenCode to try.
function expandEnvPlaceholders<T>(value: T, resolve: (name: string) => string | undefined): T {
  if (typeof value === "string") {
    return value.replace(/\{env:([A-Za-z_][A-Za-z0-9_]*)\}/g, (match, name: string) => {
      const resolved = resolve(name);
      return resolved !== undefined && resolved.length > 0 ? resolved : match;
    }) as unknown as T;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => expandEnvPlaceholders(entry, resolve)) as unknown as T;
  }
  if (isPlainObject(value)) {
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value)) {
      out[key] = expandEnvPlaceholders(entry, resolve);
    }
    return out as unknown as T;
  }
  return value;
}

function parseProviderConfig(
  raw: unknown,
  resolveEnv: (name: string) => string | undefined,
  notes: string[],
): Record<string, unknown> | null {
  if (typeof raw !== "string" || raw.trim().length === 0) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // Surface the misconfiguration instead of silently dropping the provider
    // block; an unparseable value would otherwise be undiagnosable.
    notes.push("PAPERCLIP_OPENCODE_PROVIDERS contains invalid JSON; custom providers ignored.");
    return null;
  }
  if (!isPlainObject(parsed)) {
    notes.push(
      "PAPERCLIP_OPENCODE_PROVIDERS is set but is not a JSON object; custom providers ignored.",
    );
    return null;
  }
  // Only keep provider entries that are themselves objects; surface the ones
  // we drop so a malformed entry is just as diagnosable as malformed JSON.
  const providers: Record<string, unknown> = {};
  const skipped: string[] = [];
  for (const [key, value] of Object.entries(parsed)) {
    if (isPlainObject(value)) providers[key] = expandEnvPlaceholders(value, resolveEnv);
    else skipped.push(key);
  }
  if (skipped.length > 0) {
    notes.push(
      `PAPERCLIP_OPENCODE_PROVIDERS: skipped provider(s) with non-object values: ${skipped.join(", ")}.`,
    );
  }
  return Object.keys(providers).length > 0 ? providers : null;
}

function parseConfiguredModelRef(raw: unknown): { provider: string; model: string } | null {
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  const slash = trimmed.indexOf("/");
  if (slash <= 0 || slash === trimmed.length - 1) return null;
  return { provider: trimmed.slice(0, slash), model: trimmed.slice(slash + 1) };
}

async function readJsonObject(filepath: string): Promise<Record<string, unknown>> {
  try {
    const raw = await fs.readFile(filepath, "utf8");
    const parsed = JSON.parse(raw);
    return isPlainObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

export async function prepareOpenCodeRuntimeConfig(input: {
  env: Record<string, string>;
  config: Record<string, unknown>;
  targetIsRemote?: boolean;
}): Promise<PreparedOpenCodeRuntimeConfig> {
  const skipPermissions = asBoolean(input.config.dangerouslySkipPermissions, true);
  if (!skipPermissions) {
    return {
      env: input.env,
      notes: [],
      cleanup: async () => {},
    };
  }

  // For remote execution targets the host XDG_CONFIG_HOME path is meaningless
  // (and actively harmful — it leaks a macOS-only path into the remote Linux
  // env). Callers that need to ship a runtime opencode config to the remote
  // box do that via prepareAdapterExecutionTargetRuntime in execute.ts; this
  // host-fs helper is local-only.
  if (input.targetIsRemote) {
    return {
      env: input.env,
      notes: [],
      cleanup: async () => {},
    };
  }

  const sourceConfigDir = path.join(resolveXdgConfigHome(input.env), "opencode");
  const sourceConfig = await readJsonObject(path.join(sourceConfigDir, "opencode.json"));

  // Resolve the enforcement `plugin` array against the SOURCE config dir before
  // any runtime-writable copy exists. If any entry cannot be pinned to a fixed,
  // source-anchored path, fail closed: launching with a runtime-editable plugin
  // array (or none at all) would silently drop enforcement plugins.
  const pluginPin = await resolvePinnedPluginArray(sourceConfig, sourceConfigDir);
  if (pluginPin.kind === "unresolvable") {
    throw new Error(
      `prepareOpenCodeRuntimeConfig: cannot pin OpenCode plugin array from ${sourceConfigDir}: ${pluginPin.reason}. ` +
        "Refusing to launch with a runtime-editable plugin configuration.",
    );
  }

  const runtimeConfigHome = await fs.mkdtemp(path.join(os.tmpdir(), "paperclip-opencode-config-"));
  const runtimeConfigDir = path.join(runtimeConfigHome, "opencode");
  const runtimeConfigPath = path.join(runtimeConfigDir, "opencode.json");
  const sourceConfigDirResolved = path.resolve(sourceConfigDir);

  await fs.mkdir(runtimeConfigDir, { recursive: true });
  try {
    await fs.cp(sourceConfigDir, runtimeConfigDir, {
      recursive: true,
      force: true,
      errorOnExist: false,
      dereference: false,
      // Plugin code must not flow through the runtime-owned copy: the run could
      // edit the copied files (or retarget copied symlinks) to change or drop
      // enforcement. Plugin specs resolve against the source dir instead.
      filter: (source) => {
        const rel = path.relative(sourceConfigDirResolved, path.resolve(source));
        if (rel === "" || rel.startsWith("..")) return true;
        const firstSegment = rel.split(path.sep)[0];
        return firstSegment !== "plugin" && firstSegment !== "node_modules";
      },
    });
  } catch (err) {
    if ((err as NodeJS.ErrnoException | null)?.code !== "ENOENT") {
      throw err;
    }
  }

  const existingConfig = await readJsonObject(runtimeConfigPath);
  const notes = [
    "Injected runtime OpenCode config with permission=allow for all tools and connections.",
  ];

  // Merge gateway/custom provider definitions supplied via PAPERCLIP_OPENCODE_PROVIDERS
  // (a JSON object in OpenCode's `provider` shape). OpenCode resolves a `--model
  // provider/model` only when that model exists in a provider's `models` map, and
  // OPENCODE_ALLOW_ALL_MODELS does NOT bypass its internal getModel(). So routing a
  // gateway model (e.g. an EU LLM gateway exposing OpenAI-compatible /v1) requires a
  // custom provider with an explicit models map. We accept it as config (not
  // hard-coded) so the gateway URL, key env, and model list stay declarative.
  const resolveEnv = (name: string): string | undefined => input.env[name] ?? process.env[name];
  const gatewayProviders = parseProviderConfig(
    input.env.PAPERCLIP_OPENCODE_PROVIDERS ?? process.env.PAPERCLIP_OPENCODE_PROVIDERS,
    resolveEnv,
    notes,
  );
  const existingProvider = isPlainObject(existingConfig.provider) ? existingConfig.provider : {};
  let nextProvider = gatewayProviders
    ? { ...existingProvider, ...gatewayProviders }
    : existingProvider;
  if (gatewayProviders) {
    notes.push(
      `Injected ${Object.keys(gatewayProviders).length} custom OpenCode provider(s) from PAPERCLIP_OPENCODE_PROVIDERS: ${Object.keys(gatewayProviders).join(", ")}.`,
    );
  }

  // Register the configured model on its provider's models map. OpenCode resolves
  // `--model provider/model` only when the model id exists in that map, so ids the
  // models.dev catalog does not carry — OpenRouter routing variants such as
  // `openai/gpt-oss-120b:nitro`, or models newer than the bundled catalog — are
  // otherwise rejected with "Model not found" even though the provider serves them.
  // An empty entry deep-merges with catalog metadata, so this is a no-op for models
  // the catalog already knows, and we never clobber an explicit definition from the
  // user config or PAPERCLIP_OPENCODE_PROVIDERS.
  const configuredModel = parseConfiguredModelRef(input.config.model);
  if (configuredModel) {
    const providerEntry = isPlainObject(nextProvider[configuredModel.provider])
      ? { ...(nextProvider[configuredModel.provider] as Record<string, unknown>) }
      : {};
    const providerModels = isPlainObject(providerEntry.models)
      ? { ...(providerEntry.models as Record<string, unknown>) }
      : {};
    if (!isPlainObject(providerModels[configuredModel.model])) {
      providerModels[configuredModel.model] = {};
      providerEntry.models = providerModels;
      nextProvider = { ...nextProvider, [configuredModel.provider]: providerEntry };
      notes.push(
        `Registered configured model ${configuredModel.provider}/${configuredModel.model} in the runtime OpenCode config.`,
      );
    }
  }

  const nextConfig: Record<string, unknown> = {
    ...existingConfig,
    permission: "allow",
  };
  if (Object.keys(nextProvider).length > 0) {
    nextConfig.provider = nextProvider;
  }

  // Pin OpenCode's auxiliary "small" model (used for session-title generation and
  // other helper tasks) via PAPERCLIP_OPENCODE_SMALL_MODEL. OpenCode otherwise
  // defaults the small model to a built-in provider default (e.g. a claude-* model
  // for the anthropic provider); when that provider is repointed at a gateway that
  // does not serve that exact model, the title-gen call fails and aborts the run.
  // Setting small_model to a gateway-served model keeps every call on supported models.
  const smallModel = (input.env.PAPERCLIP_OPENCODE_SMALL_MODEL ?? process.env.PAPERCLIP_OPENCODE_SMALL_MODEL)?.trim();
  if (smallModel) {
    nextConfig.small_model = smallModel;
    notes.push(`Pinned OpenCode small_model to ${smallModel}.`);
  }

  // Last write wins: pin the plugin array to the source-resolved references so
  // anything the copy produced (or a run wrote between the copy and now) is
  // replaced. Enforcement plugins load from the config-owner-anchored source
  // paths, not from this runtime-writable directory.
  nextConfig.plugin = pluginPin.plugin;
  notes.push(
    `Pinned ${pluginPin.plugin.length} OpenCode plugin(s) to source config dir paths outside the runtime-owned copy.`,
  );

  await fs.writeFile(runtimeConfigPath, `${JSON.stringify(nextConfig, null, 2)}\n`, "utf8");

  return {
    env: {
      ...input.env,
      XDG_CONFIG_HOME: runtimeConfigHome,
    },
    notes,
    cleanup: async () => {
      await fs.rm(runtimeConfigHome, { recursive: true, force: true });
    },
  };
}

/** Managed credentials must never leave host-only homes in a remote process. */
export function prepareManagedOpenCodeRemoteHomes(input: {
  env: Record<string, string>;
  config: Record<string, unknown>;
  runtimeRootDir: string | null | undefined;
  runId: string;
  configDir?: string;
}): void {
  if (!input.config.managedAiConnection) return;
  if (!input.runtimeRootDir) throw new Error("Managed OpenCode authentication requires an isolated remote runtime directory.");
  const home = path.posix.join(input.runtimeRootDir, "managed-auth", input.runId);
  Object.assign(input.env, {
    HOME: home,
    XDG_CONFIG_HOME: input.configDir ?? path.posix.join(home, "config"),
    XDG_DATA_HOME: path.posix.join(home, "data"),
    XDG_CACHE_HOME: path.posix.join(home, "cache"),
    XDG_STATE_HOME: path.posix.join(home, "state"),
  });
}
