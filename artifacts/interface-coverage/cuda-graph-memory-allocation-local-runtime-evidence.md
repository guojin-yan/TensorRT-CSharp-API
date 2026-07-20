# CUDA Graph Memory Allocation Local Runtime Evidence

Classification: `project-reference-local-runtime`

The TRT10/CUDA12.9 bridge ran `CudaGraphSmokeRunner` on an NVIDIA GeForce RTX 3060 Laptop GPU
with driver 576.02. The graph allocated 64 bytes, filled them with `0x6B`, copied them to pinned
host memory, added the matching free node, instantiated, launched, synchronized, and verified all
64 bytes.

```text
CudaGraphMemoryAllocation Bytes=64 Pattern=0x6B GraphDisposeRejected=True CrossGraphRejected=True SecondFreeRejected=True Nodes=4
```

The smoke also confirmed that an active allocation wrapper prevents graph disposal, a different
graph cannot consume the wrapper, and a second free node is rejected.

This record deliberately stores no device address and no full smoke log. It sets
`isPackageConsumerRuntimeProof=false`, `canPromoteRuntimeProof=false`, and
`canPublishPublicly=false`; it is not clean consumer, post-publish, or release-close evidence.
