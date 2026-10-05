// A small pull renderer for aofx effects: the host half of the contract.
//
// Asked for a window of a node, it works out the node's region of
// definition from its inputs', asks the effect how much of each input that
// window needs (regionOfInterest), renders those, and then either hands back
// the pass-through input (isIdentity, or a disabled node) or allocates the
// output and calls process. The R/G/B/A switches every aofx node gets are
// applied here, from the pass-through input, as an aofx host does.

import bridgeModule from "../generated/fx/bridge";
import { UniformWriter } from "../uniforms";
import { type Effect, type InputPlane, RenderRequest } from "./Effect";
import type { Gpu } from "./Gpu";
import {
  EMPTY_RECT,
  type FxBuffer,
  type ParamValues,
  type Rect,
  intersectRect,
  isEmptyRect,
  passThroughClip,
  placed,
  rectHeight,
  rectWidth,
} from "./types";

/** A picture that is already rendered: the graph's leaf. */
export interface FxSource {
  readonly picture: FxBuffer;
  /** Its whole region; the picture's own rectangle when left out. */
  readonly rod?: Rect;
  readonly values?: ReadonlyMap<string, readonly number[]>;
}

export interface FxNode {
  readonly effect: Effect;
  params?: ParamValues;
  /** By clip name; a missing or null entry is an unwired input. */
  inputs?: Readonly<Record<string, FxInput | null | undefined>>;
  /** False passes the pass-through input on, unrendered. */
  enabled?: boolean;
  /** Which of R, G, B, A the effect may change; the rest come back from the pass-through input. */
  channels?: readonly [boolean, boolean, boolean, boolean];
  /** Output planes beyond Color to allocate (Grade's "Clip"). */
  planes?: readonly string[];
  /** The node's stable name, for `keep` keys and attached values. */
  instance?: string;
}

export type FxInput = FxNode | FxSource;

export const isSource = (n: FxInput): n is FxSource => "picture" in n;

export interface FxContext {
  time?: number;
  scaleX?: number;
  scaleY?: number;
  projectWidth?: number;
  projectHeight?: number;
}

export interface Rendered {
  /** Null where the node has no picture in the window asked for. */
  readonly picture: FxBuffer | null;
  readonly rod: Rect;
  readonly planes: ReadonlyMap<string, FxBuffer>;
  /** Numbers hung on the picture by this node and those upstream. */
  readonly values: ReadonlyMap<string, readonly number[]>;
  /** Values that arrive after the frame is flushed (ImageStatistics). */
  readonly pending: ReadonlyMap<string, Promise<number[]>>;
}

const NOTHING = (rod: Rect): Rendered => ({
  picture: null,
  rod,
  planes: new Map(),
  values: new Map(),
  pending: new Map(),
});

export class FxGraph {
  private rods = new Map<FxInput, Rect>();
  private ctx: Required<FxContext> = {
    time: 0,
    scaleX: 1,
    scaleY: 1,
    projectWidth: 0,
    projectHeight: 0,
  };

  constructor(readonly gpu: Gpu) {}

  /** Renders `window` of `node`; records into the Gpu's frame, without flushing. */
  render(node: FxInput, window: Rect, ctx: FxContext = {}): Rendered {
    this.rods.clear();
    this.ctx = { ...this.ctx, ...ctx };
    try {
      return this.pull(node, window);
    } finally {
      this.rods.clear();
    }
  }

  /** A node's region of definition, from its inputs'. */
  regionOfDefinition(node: FxInput): Rect {
    const had = this.rods.get(node);
    if (had) return had;
    let rod: Rect;
    if (isSource(node)) {
      rod = node.rod ?? node.picture.rect;
    } else if (node.enabled === false) {
      const pass = this.passThroughInput(node);
      rod = pass ? this.regionOfDefinition(pass) : EMPTY_RECT;
    } else {
      rod = node.effect.regionOfDefinition(this.query(node));
    }
    this.rods.set(node, rod);
    return rod;
  }

  private query(node: FxNode) {
    const clips = node.effect.describe().inputs;
    return {
      time: this.ctx.time,
      scaleX: this.ctx.scaleX,
      scaleY: this.ctx.scaleY,
      params: node.params ?? {},
      inputRods: clips.map((c) => {
        const input = node.inputs?.[c.name];
        return input ? this.regionOfDefinition(input) : EMPTY_RECT;
      }),
    };
  }

  private passThroughInput(node: FxNode): FxInput | null {
    const clip = passThroughClip(node.effect.describe());
    return (clip && node.inputs?.[clip.name]) || null;
  }

  private pull(node: FxInput, window: Rect): Rendered {
    const rod = this.regionOfDefinition(node);
    if (isSource(node)) {
      return {
        ...NOTHING(rod),
        picture: node.picture,
        values: node.values ?? new Map(),
      };
    }
    if (node.enabled === false) {
      const pass = this.passThroughInput(node);
      return pass ? this.pull(pass, window) : NOTHING(rod);
    }
    const area = intersectRect(rod, window);
    if (isEmptyRect(area)) return NOTHING(rod);

    const effect = node.effect;
    const desc = effect.describe();
    const query = this.query(node);
    const rois = effect.regionOfInterest(query, area);
    const request = new RenderRequest(this.gpu, area);
    Object.assign(request, {
      time: this.ctx.time,
      scaleX: this.ctx.scaleX,
      scaleY: this.ctx.scaleY,
      projectWidth: this.ctx.projectWidth,
      projectHeight: this.ctx.projectHeight,
      instance: node.instance ?? desc.identifier,
      outputRod: rod,
      params: node.params ?? {},
    });

    const pass = passThroughClip(desc);
    let passed: Rendered | null = null;
    const pending = new Map<string, Promise<number[]>>();
    desc.inputs.forEach((clip, i) => {
      const input = node.inputs?.[clip.name];
      if (!input) return;
      const roi = rois[i] && !isEmptyRect(rois[i]) ? rois[i] : area;
      // Never more than the input has: the host intersects what is asked.
      const wanted = intersectRect(roi, query.inputRods[i]);
      const got = this.pull(input, isEmptyRect(wanted) ? roi : wanted);
      for (const [k, v] of got.pending) pending.set(k, v);
      if (clip === pass) passed = got;
      if (!got.picture) return;
      const plane: InputPlane = {
        clip: clip.name,
        plane: "Color",
        buffer: got.picture,
        rod: query.inputRods[i],
        values: got.values,
      };
      request.inputs.push(plane);
      for (const [id, buffer] of got.planes) {
        request.inputs.push({ ...plane, plane: id, buffer });
      }
    });
    const upstream = passed as Rendered | null;

    if (effect.isIdentity(request)) {
      return upstream ?? NOTHING(rod);
    }

    const gpu = this.gpu;
    const out = placed(gpu.scratch(rectWidth(area), rectHeight(area)), area);
    request.outputs.push({ plane: "Color", buffer: out });
    const planes = new Map<string, FxBuffer>();
    for (const id of node.planes ?? []) {
      if (!desc.outputs.some((p) => p.id === id) || id === "Color") continue;
      const b = placed(gpu.scratch(out.width, out.height), area);
      planes.set(id, b);
      request.outputs.push({ plane: id, buffer: b });
    }
    if (!effect.process(request)) {
      throw new Error(
        `${desc.identifier} (${request.instance}) refused to render${request.complaint ? `: ${request.complaint}` : ""}`,
      );
    }
    const before = upstream?.picture;
    const mask = node.channels;
    if (mask && !mask.every(Boolean) && before) {
      this.restoreChannels(out, before, mask);
    }

    const values = new Map(upstream?.values ?? []);
    for (const [k, v] of request.produced) {
      if (v instanceof Promise) pending.set(k, v);
      else values.set(k, v);
    }
    return { picture: out, rod, planes, values, pending };
  }

  private restoreChannels(
    out: FxBuffer,
    before: FxBuffer,
    mask: readonly boolean[],
  ) {
    const u = UniformWriter.for(bridgeModule).setAll({
      width: out.width,
      height: out.height,
      stride: out.stride,
      srcOffsetX: out.rect.x1 - before.rect.x1,
      srcOffsetY: out.rect.y1 - before.rect.y1,
      srcWidth: before.width,
      srcHeight: before.height,
      srcStride: before.stride,
      channelMask: mask.reduce((m, on, i) => (on ? m | (1 << i) : m), 0),
    });
    this.gpu.run(
      this.gpu.load(bridgeModule, "restoreChannels"),
      [out.width, out.height, 1],
      { src: before, picture: out },
      u,
    );
  }
}
