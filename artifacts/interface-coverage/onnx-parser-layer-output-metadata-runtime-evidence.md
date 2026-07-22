# ONNX Parser Layer Output Metadata Runtime Evidence

State: `local-project-reference-copied-readonly-diagnostics`

## Results

- Deferred inventory: `598` TensorRT rows; low `0`, medium `112`, high `486`.
- Vendor ABI: TRT8 header absent; TRT10/11 pure virtual method present. Named LIB/DLL symbol matches are `0`
  because dispatch uses the parser vtable.
- Bindings: `196 manifests / 3973 records`, repeated generation and validation passed.
- ABI/PE: TRT8 `991/991`, TRT10 `1087/1087`, TRT11 `1234/1234`; missing `0`.
- TRT8/CUDA12.1: runtime and builder creation passed, but this bridge was built without the ONNX parser dependency.
  Parser construction produced a controlled skip with exit code `0`; metadata was not queried.
- TRT10/CUDA12.9: metadata runtime passed for `identity[0]`; tensor `output`, shape `[-1, 4]`, `Float`,
  `Device`, dynamic execution/network-output flags true. `TryGet` and `Get` matched, missing layer returned false,
  and the enclosing enqueue/output comparison passed.
- TRT11/CUDA12.9: bridge/vendor DLL/version/registry probes passed, but vendor builder/runtime creation returned
  null. Metadata was not queried; state remains `dependency-runtime-probe-only`.
- TRT8/10/11 bridge-only package consumers restored and built without `ProjectReference`. Evidence remains
  `compile-surface-proof` and includes `onnx-parser-layer-output-copied-metadata`.

## Boundary

The TRT8 parser-dependency skip and TRT11 dependency/runtime probe are not metadata runtime proof. This record is not
model accuracy, public clean package consumer, post-publish or release-close proof. It does not permit deleting
deferred history, publishing packages, uploading a GitHub Release or closing an issue.
