// Hooks where a kernel runs dyno shader graphs (src/dyno), which users build
// at runtime and so can't be compiled offline. As compiled here they pass the
// splat through; src/webgpu/dyno/patchKernel.ts swaps their bodies in the WGSL
// for the graph's generated code and builds a pipeline per distinct graph.
// [noinline] keeps each a function of its own for the patcher to find.

module dyno_hooks;

import core.splat_types;

// A generator: the splat for an output index, in place of the source read.
[noinline]
public Gsplat dynoSource(int index) {
    Gsplat s;
    s.flags = 0u;
    s.index = index;
    return s;
}

// SplatMesh.objectModifiers, in object space after the source read.
[noinline]
public Gsplat dynoObjectModifier(Gsplat s) {
    return s;
}

// SplatMesh.worldModifiers, in world space after the transform and recolor.
[noinline]
public Gsplat dynoWorldModifier(Gsplat s) {
    return s;
}
