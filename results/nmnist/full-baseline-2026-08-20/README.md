# N-MNIST native-event baseline: 2026-08-20

This is a complete 60,000-train / 10,000-test run of the project’s
`timestamp-binned-prototype-snn` baseline. It preserves native event timestamp
and polarity information across ten time bins. The measured test accuracy is
**33.91%**.

This result establishes a reproducible event-stream baseline, not a competitive
N-MNIST result. The committed `manifest.json` records the exact source
revisions, command, hardware/toolchain, and SHA-256/MD5 digests of the
official archives. The 1.5 GB extracted dataset and archives are deliberately
excluded from source control.
