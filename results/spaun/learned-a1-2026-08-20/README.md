# Learned Spaun A1 gate: 2026-08-20

The registered spiking-convolution A1 model trained for five epochs on the
MNIST training split and reached **98.36%** on the 10,000-example held-out
MNIST test split (9,836 correct). Its checkpoint was then loaded into the
Spaun task loop.

The initial full-canvas 5×7 glyph rasterization was incompatible with the
MNIST training geometry. The learned path now preserves glyph aspect ratio and
centres the raster on the 28×28 canvas. With that preprocessing, the real
`A1[7]?` task-loop stimulus produced `7` (100%). This is a checkpoint-level
perception integration result; it does not claim a shared latent-spike
representation between visual and cognitive modules.

`manifest.json` retains the two source revisions, exact commands, dataset
digests, checkpoint digest, held-out result, and task-loop output. The model
checkpoint and MNIST data remain generated local artifacts.
