// The CPU sort path (sort: "cpu"), and the reference the GPU sort is
// tested against.

/**
 * Indices of finite metrics, largest first: an LSD radix sort on 16-bit
 * digits of the inverted float bits (non-negative floats order like their
 * bits). The CPU path until the GPU sort replaces it.
 */
export function sortBackToFront(
  metric: Float32Array,
  count: number,
): Uint32Array {
  const bits = new Uint32Array(metric.buffer, metric.byteOffset, count);
  let keys = new Uint32Array(count);
  let order = new Uint32Array(count);
  let n = 0;
  for (let i = 0; i < count; i++) {
    if (Number.isFinite(metric[i])) {
      keys[n] = ~bits[i] >>> 0;
      order[n] = i;
      n++;
    }
  }
  let keys2 = new Uint32Array(n);
  let order2 = new Uint32Array(n);
  const counts = new Uint32Array(65536);
  for (const shift of [0, 16]) {
    counts.fill(0);
    for (let k = 0; k < n; k++) counts[(keys[k] >>> shift) & 0xffff]++;
    let sum = 0;
    for (let d = 0; d < 65536; d++) {
      const c = counts[d];
      counts[d] = sum;
      sum += c;
    }
    for (let k = 0; k < n; k++) {
      const at = counts[(keys[k] >>> shift) & 0xffff]++;
      keys2[at] = keys[k];
      order2[at] = order[k];
    }
    [keys, keys2] = [keys2, keys];
    [order, order2] = [order2, order];
  }
  return order.slice(0, n);
}
