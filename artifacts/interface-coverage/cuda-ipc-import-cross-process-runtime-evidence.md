# CUDA IPC Import Cross-Process Runtime Evidence

Classification: `project-reference-local-cross-process-runtime`

The TRT10/CUDA12.9 bridge ran `CudaIpcImportSmokeRunner` on an NVIDIA GeForce RTX 3060 Laptop GPU
with driver 576.02 in WDDM mode. The exporting process created the CUDA IPC share handles before
issuing the writes that the importing process must observe, then initialized and verified a 64-byte
allocation. A real child process imported the event and memory mapping, synchronized the event,
verified all 64 bytes, wrote a second byte pattern, rejected `FreeAsync`, and closed the mapping via
`cudaIpcCloseMemHandle`. The exporter remained alive and verified the child's write after exit.

```text
CudaIpcImportSmokeRunner Passed=True CrossProcess=True Device=0 EventImported=True MemoryImported=True ImportedClose=True SourceOwnerAlive=True IpcEventSupport=True BytesVerified=64 ChildMarker=True
```

Tokens were transported to the child through redirected standard input, not command-line arguments.
The evidence stores no token contents, device address, or full smoke output. It explicitly keeps
`isPackageConsumerRuntimeProof=false`, `canPromoteRuntimeProof=false`, and
`canPublishPublicly=false`; this is not clean-consumer, post-publish, or release-close evidence.
