import { describe, expect, it } from "vitest";
import {
  usedBindings,
  validateDispatch,
  workgroupCount,
} from "../../src/webgpu/KernelRegistry";
import cornerpin from "../../src/webgpu/generated/fx/cornerpin";
import invert from "../../src/webgpu/generated/fx/invert";

const fakeBuffer = (size: number) => ({ size }) as unknown as GPUBuffer;

describe("generated reflection", () => {
  it("has both cornerpin entries", () => {
    const names = cornerpin.reflection.entries.map((e) => e.name);
    expect(names).toEqual(["pinWarp", "pinBlur"]);
  });

  it("lists invert's bindings in binding order", () => {
    const entry = invert.reflection.entries[0];
    expect(usedBindings(invert, entry).map((b) => [b.name, b.kind])).toEqual([
      ["src", "read-only-storage"],
      ["mask", "read-only-storage"],
      ["dst", "storage"],
      ["params", "uniform"],
    ]);
  });
});

describe("validateDispatch", () => {
  const entry = invert.reflection.entries[0];
  const params = invert.reflection.bindings.find((b) => b.name === "params");
  const good = {
    grid: [100, 50] as const,
    buffers: {
      src: fakeBuffer(16 * 100),
      mask: fakeBuffer(16),
      dst: fakeBuffer(16 * 100),
    },
    uniforms: new Uint8Array(params?.bytes ?? 0),
  };

  it("accepts a matching dispatch", () => {
    expect(validateDispatch(invert, entry, good)).toEqual([]);
  });

  it("refuses a wrong uniform size", () => {
    const errors = validateDispatch(invert, entry, {
      ...good,
      uniforms: new Uint8Array(4),
    });
    expect(errors.join()).toMatch(/uniform block 'params'/);
  });

  it("refuses a missing buffer and a ragged one", () => {
    const errors = validateDispatch(invert, entry, {
      ...good,
      buffers: { src: fakeBuffer(17), dst: fakeBuffer(16) },
    });
    expect(errors.join("\n")).toMatch(/'src' is 17 bytes/);
    expect(errors.join("\n")).toMatch(/missing buffer 'mask'/);
  });
});

describe("workgroupCount", () => {
  it("rounds threads up to whole workgroups", () => {
    const entry = invert.reflection.entries[0];
    expect(workgroupCount(entry, [100, 50])).toEqual([7, 4, 1]);
  });
});
