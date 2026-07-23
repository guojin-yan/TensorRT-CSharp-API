# TRT8 Legacy Parser Readonly Runtime Evidence

State: `local-project-reference-copied-readonly-diagnostics`

## Results

- TensorRT 8 coverage now scans `880` interfaces per local package, including `NvCaffeParser.h` and
  `NvUffParser.h`: `760 implemented / 120 deferred-only`.
- Seven vendor rows are `implemented-with-deferred-history`: the three UFF required-version getters,
  `ICaffeParser::parseBinaryProto`, and the blob data/type/dimensions getters. Neighboring parse and destroy rows
  remain deferred-only.
- Bindings are `197 manifests / 3975 records`; repeated generation and output validation passed.
- ABI/PE parity is TRT8 `993/993`, TRT10 `1087/1087`, TRT11 `1234/1234`; all missing counts are zero.
- TRT8/CUDA12.1 runtime returned UFF `0.6.9`. The MNIST mean binaryproto produced shape `[1,1,28,28]`, `Float`,
  and `3136` copied bytes. Copied payload SHA256 is
  `DF7D560B482098FAC1C6122C22BD0A54499ED9F8EC3AC6BAE8FC917D3A01774A`.
- The source protobuf container is `3147` bytes with SHA256
  `337CF38DD3A69F25BA7E732D25CA3176576CA8F208B0A916FA3AC2A669A894BE`; it is intentionally different from
  the copied float payload hash.
- Independent managed data copies and the managed pre-dispatch guard for TRT10/11 both passed. Native parser/blob
  objects do not escape, and the bridge does not call process-global `shutdownProtobufLibrary`.
- TRT8/10/11 bridge-only packages restored and built in consumers without `ProjectReference`, 0 warning / 0 error.
  This remains `compile-surface-proof`, not package-consumer runtime proof.

## Boundary

This local runtime record proves the TRT8 copied diagnostic path on one host. It does not authorize package push,
GitHub Release upload, GitHub Actions dispatch, public publication, issue closure, or deletion of deferred history.
