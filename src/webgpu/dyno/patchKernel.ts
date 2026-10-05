// Splices dyno-generated WGSL into a Slang kernel's WGSL at its dyno hooks
// (slang/core/dyno_hooks.slang): each hook function slangc emitted gets a
// body that converts the kernel's Gsplat to the dyno one, calls the graph's
// function, and converts back. The result is a new KernelModule with the
// dyno uniform block and textures as "external" bindings in their own group.

import type { WgslDynoProgram } from "../../dyno/wgsl";
import type { KernelModule } from "../KernelModule";
import { type TextureLayouts, dynoBindingReflections } from "./bindings";

/** Which hooks to fill, with the name of the program function for each. */
export interface DynoHookFunctions {
  dynoSource?: string;
  dynoObjectModifier?: string;
  dynoWorldModifier?: string;
}

// The dyno Gsplat's fields (src/dyno/wgsl/prelude.ts), in order.
const GSPLAT_FIELDS = [
  "center",
  "flags",
  "scales",
  "index",
  "quaternion",
  "rgba",
];

interface FoundFunction {
  start: number;
  end: number;
  name: string;
  params: { name: string; type: string }[];
  returns: string;
}

/** A function slangc emitted for `base` (named base_N), with its extent. */
export function findFunction(wgsl: string, base: string): FoundFunction | null {
  const re = new RegExp(
    `fn (${base}_\\d+)\\s*\\(([^)]*)\\)\\s*->\\s*([\\w<>]+)\\s*\\{`,
  );
  const m = re.exec(wgsl);
  if (!m) return null;
  let depth = 0;
  let end = m.index + m[0].length - 1;
  for (; end < wgsl.length; end++) {
    if (wgsl[end] === "{") depth++;
    else if (wgsl[end] === "}" && --depth === 0) break;
  }
  const params = m[2]
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => {
      const [name, type] = p.split(":").map((s) => s.trim());
      return { name, type };
    });
  return { start: m.index, end: end + 1, name: m[1], params, returns: m[3] };
}

/** The kernel's Gsplat struct name and its fields by unmangled name. */
function slangGsplat(wgsl: string) {
  const m = /struct (Gsplat_\d+)\s*\{([^}]*)\}/.exec(wgsl);
  if (!m) throw new Error("dyno: kernel has no Gsplat struct");
  const fields = new Map<string, string>();
  for (const line of m[2].split(",")) {
    const name = line.split(":")[0].trim();
    if (name) fields.set(name.replace(/_\d+$/, ""), name);
  }
  for (const f of GSPLAT_FIELDS) {
    if (!fields.has(f)) throw new Error(`dyno: kernel Gsplat lacks ${f}`);
  }
  return { name: m[1], fields };
}

// The kernel's Gsplat may carry more fields than dyno's: those pass through
// from `base` (a modifier's input, zero for a generator).
function converters(wgsl: string) {
  const { name, fields } = slangGsplat(wgsl);
  const from = GSPLAT_FIELDS.map((f) => `s.${fields.get(f)}`).join(", ");
  const to = GSPLAT_FIELDS.map((f) => `    k.${fields.get(f)} = s.${f};`).join(
    "\n",
  );
  return {
    type: name,
    code: [
      `fn dyno_from_kernel(s: ${name}) -> Gsplat {\n    return Gsplat(${from});\n}`,
      `fn dyno_to_kernel(s: Gsplat, base: ${name}) -> ${name} {\n    var k = base;\n${to}\n    return k;\n}`,
    ].join("\n\n"),
  };
}

function hookBody(
  hook: keyof DynoHookFunctions,
  fn: string,
  found: FoundFunction,
) {
  const [p] = found.params;
  const call =
    hook === "dynoSource"
      ? `dyno_to_kernel(${fn}(${p.name}), ${found.returns}())`
      : `dyno_to_kernel(${fn}(dyno_from_kernel(${p.name})), ${p.name})`;
  const params = found.params.map((x) => `${x.name}: ${x.type}`).join(", ");
  return `fn ${found.name}(${params}) -> ${found.returns}\n{\n    return ${call};\n}`;
}

export function patchKernel(
  base: KernelModule,
  program: WgslDynoProgram,
  hooks: DynoHookFunctions,
  name: string,
  textureLayouts: TextureLayouts = {},
): KernelModule {
  let wgsl = base.wgsl;
  for (const [hook, fn] of Object.entries(hooks) as [
    keyof DynoHookFunctions,
    string | undefined,
  ][]) {
    if (!fn) continue;
    const found = findFunction(wgsl, hook);
    if (!found) throw new Error(`dyno: ${base.name} has no ${hook} hook`);
    wgsl =
      wgsl.slice(0, found.start) +
      hookBody(hook, fn, found) +
      wgsl.slice(found.end);
  }
  const conv = converters(wgsl);
  wgsl = `${wgsl}\n\n// dyno\n\n${program.code}\n\n${conv.code}\n`;

  const extra = dynoBindingReflections(program, textureLayouts);
  const names = extra.map((b) => b.name);
  return {
    name,
    wgsl,
    reflection: {
      entries: base.reflection.entries.map((e) => ({
        ...e,
        uses: [...e.uses, ...names],
      })),
      bindings: [...base.reflection.bindings, ...extra],
    },
  };
}
