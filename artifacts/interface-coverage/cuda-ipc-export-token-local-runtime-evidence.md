# CUDA IPC Export Token Local Runtime Evidence

Generated: 2026-07-20

## Result

| Check | Result |
| --- | --- |
| CUDA runtime / driver | `12090` / `12090` |
| GPU | `NVIDIA GeForce RTX 3060 Laptop GPU` |
| event export token | `Kind=Event Length=64` |
| memory export token | `Kind=Memory Length=64` |
| default event rejected | `True` |
| managed memory rejected | `True` |
| token contents recorded | `False` |

The local `CudaSmokeRunner` execution completed with the CUDA 12.9 TRT10 bridge. The bridge SHA256
is `136a312766f03188d0a7f870eea6624b6b6f014040dddda38e8020600ed7469b`.

## Classification

| Boundary | Value |
| --- | --- |
| local runtime evidence | `True` |
| cross-process runtime proof | `False` |
| package consumer runtime proof | `False` |
| can promote runtime proof | `False` |
| can publish publicly | `False` |
| can close release issue | `False` |

This ProjectReference run verifies export-token creation and the local fail-closed guards only. It
does not open the token in a second process, verify imported-resource lifetime, consume packed NuGet
artifacts, authorize package publication, or authorize release-issue closure.
