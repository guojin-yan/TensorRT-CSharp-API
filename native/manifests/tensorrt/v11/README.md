# TensorRT 11 manifests

This folder contains TensorRT 11 specific bridge API manifests.

TensorRT 11 changed several API surfaces compared with TensorRT 8/10, so this
line is intentionally kept in a dedicated manifest path instead of reusing the
8/10 adapter entries blindly.

Current scope:

- minimal deployment object chain
- serialized engine build/deserialization
- host memory copy/size query
- engine I/O tensor metadata

APIs that need TensorRT 11-specific signatures, callbacks, plugin plumbing, or
complex ownership rules should be marked as manual/semi-generated in the
manifest before being lifted into managed wrappers.
