Relit Corvette validation in float (thread BD).

  node tools/validate-corvette/serve.mjs            # :8121 (VALIDATE_PORT, also for run.mjs), /r2/ = publish-r2/sparkwebgpu (R2_DIR)
  node tools/validate-corvette/run.mjs <set> <outdir> "<query>" all|close,far [--time] [--bg] [--masks]
  node tools/validate-corvette/compare.mjs spec.json

run.mjs opens examples/webgpu/validate-corvette.html (createCorvette, the public
page's scene) at 1920x1080, puts the camera on fixed views (close = athenea's
/World/Camera, default 2.2x, side, glass, rear, far 4x), and reads the HalfFloat
target back BEFORE the fx chain (linear light) into RGB PFMs. --bg also renders
the dome alone (car mask), --masks the car with each of paint/body/glass/trim
hidden (part masks). --time times 40 whole page frames, each waited on the queue.
compare.mjs: relMSE = mean (x-r)^2/(r^2+0.01) over a mask, values clamped to
[0,16]; trim999 drops the worst 0.1% pixels; ratio = sum x / sum r; flies =
pixels with lum > 1 and > 4 lum(ref) + 0.25. Writes PNG previews and diff maps.
Run every GPU job through the scratchpad gpu-run.sh (one at a time, cool Mac).
