# Application Shared Source

English | [简体中文](README.zh-CN.md)

This directory contains application-layer source shared by complete workflows under `applications/`. It does not contain user-facing samples and none of its projects are published as NuGet packages.

`JYPPX.TensorRtSharp.ApplicationTools` links the current `JYPPX.TensorRtSharp.Tools` source for `OnnxToEngine` and `TensorRtExec`, but compiles that source against the published 4-series `JYPPX.TensorRT.CSharp.API` package. This keeps the applications on the public managed API while the source-backed Tools project remains available for core-library development and tests.

The shared project must not add `ProjectReference` entries to `src/JYPPX.CudaSharp` or `src/JYPPX.TensorRtSharp`. CUDA, cuDNN, TensorRT, and NVRTC remain user-installed prerequisites.
