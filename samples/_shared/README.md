# Shared Sample Sources

English | [简体中文](README.zh-CN.md)

This directory contains implementation files compiled into multiple samples and applications through
`build/JYPPX.SampleSupport.props`. It is not a runnable project and does not produce a NuGet package.

Keep reusable command-line, tensor-input, reference-validation, and image-decoding support here. User-facing cases
belong in the numbered module directories under `samples`; validation-only templates belong in `tests/fixtures`.
