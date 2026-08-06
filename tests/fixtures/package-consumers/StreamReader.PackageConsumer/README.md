# StreamReader package consumer

This repository-external consumer validates the public `TensorRtStreamReader` API through exactly two local package references:

- `JYPPX.TensorRT.CSharp.API`
- the runtime-key-specific bridge-only package

The program builds an in-memory identity engine, compares ordinary buffer deserialization with real TensorRT `IStreamReaderV2` deserialization, executes the returned engine, reuses the reader sequentially, and verifies deferred disposal plus truncated-input failure. CUDA and TensorRT remain host-installed dependencies.

Run it through `eng/Test-StreamReaderLocalPackageConsumer.ps1`; the harness creates and removes a workspace outside the repository. This sample does not download a model and does not publish packages or releases.
