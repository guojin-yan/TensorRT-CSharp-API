# TensorRT Compatible-Host Source Runtime Evidence

| Field | Value |
|---|---|
| validation state | `passed-compatible-host-source-runtime` |
| source commit | `1062d45693b9724332e306db49eeb0507036e419` |
| preset | `win-x64-trt10-cuda11-release` |
| TensorRT | `10.13.0` |
| CUDA | `11.8` |
| GPU | `NVIDIA GeForce RTX 3060 Laptop GPU` |
| import libraries inside selected root | `true` |
| runtime / builder | `true / true` |
| serialized network / minimal chain | `true / true` |
| high-level chain / enqueue | `true / true` |
| exact package matrix evidence | `false` |
| package consumer proof | `false` |
| performs publish | `false` |

## Artifacts

- Transcript: `artifacts/real-case/tensorrt10-compatible-host-source-runtime/tensorrt-smoke-transcript.txt`
- Screenshot: `docs/images/tensorrt10-compatible-host-source-runtime-terminal.png`
- Bridge SHA256: `ab99da598130d5d11added18a4b410403e6d8c1f4103ab40ab9a9fbe20cb2196`

## Boundary

This proves the current source bridge on one TensorRT same-major compatible host. It does not prove an exact release package key, a package consumer, a public package, a release, or permission to publish.
