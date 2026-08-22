# CIFAR-10 temporal SEW-ResNet: five seeds, 2026-08-20

The artifact-backed five-seed CUDA protocol completed all 100 epochs per seed
on the complete CIFAR-10 training and test splits. Test accuracy was
**90.582% ± 0.120%** (sample standard deviation): 90.46%, 90.59%, 90.47%,
90.65%, and 90.74% for seeds 42–46 respectively.

All five runs used the same temporal executable, whose SHA-256 is retained in
`manifest.json`. Seeds 42–43 recorded the clean published experiments revision
`cf7377b`; seeds 44–46 recorded clean published revision `427cbb0`. The
intervening changes affect only Spaun results/documentation and glyph
preprocessing; `experiments/cifar10`, the temporal runner, CMake temporal
wiring, and the linked library revision were unchanged. This is disclosed to
make the provenance boundary explicit rather than implying a single source
revision where there was not one.

`summary.csv` preserves each selected-validation checkpoint hash and log hash.
The full per-seed manifests, logs, selected checkpoints, and final checkpoints
are retained locally under the ignored artifact directory named by the protocol.
The result is below the external TET comparison target previously cited in the
roadmap and should be treated as this implementation's reproducible baseline,
not a state-of-the-art claim.
