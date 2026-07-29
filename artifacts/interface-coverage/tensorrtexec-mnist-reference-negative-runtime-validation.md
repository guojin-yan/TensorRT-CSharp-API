# TensorRtExec MNIST Reference Negative Runtime Validation

- strict: `True`
- runtime artifacts required: `True`
- checks: `72`
- passed: `72`
- failed: `0`

| Check | Passed | Actual |
| --- | --- | --- |
| `schema` | `True` | tensorrtexec-mnist-reference-negative-runtime-evidence.v1 |
| `state` | `True` | controlled-reference-negative-runtime-passed |
| `classification` | `True` | controlled-negative-runtime |
| `runtime-key` | `True` | win-x64-trt10.11-cuda12.9-cudnn9.22/10 |
| `build-processes` | `True` | 0/0/0 |
| `base-reference` | `True` | 1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571/onnxruntime-cpu-1.23.2-derived-unreviewed/False |
| `engine-input-hashes` | `True` | 14044b5d345a68bebe7b01c3a48ce10c1665bde40088fbbfd61ae236f1221a89/81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564 |
| `consumer-isolation` | `True` | True/False/True/True/False/True |
| `consumer-package-contract` | `True` | JYPPX.TensorRT.CSharp.API/4.0.0-rtc-local.20260728/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge/4.0.0 |
| `case-count` | `True` | 5/5 |
| `case-ids` | `True` | name-mismatch,shape-mismatch,value-count-mismatch,nan-reject,infinity-reject |
| `fail-closed-counts` | `True` | 5/5 |
| `path-free-compact-evidence` | `True` | absolute-windows-path-present=False |
| `base-reference-file-hash` | `True` | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7.onnxruntime-cpu.reference.json/1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571 |
| `name-mismatch-source-execution` | `True` | 2/True/True/False |
| `name-mismatch-source-validation` | `True` | False/False/0/0/-1 |
| `name-mismatch-source-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/f9614e8c7a1c2b5ee521ac0f3c312cbb7a5e3eef61817a91a615f387b8d2f2ac |
| `name-mismatch-consumer-execution` | `True` | 1/True/True/False |
| `name-mismatch-consumer-validation` | `True` | False/False/0/0/-1 |
| `name-mismatch-consumer-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/f9614e8c7a1c2b5ee521ac0f3c312cbb7a5e3eef61817a91a615f387b8d2f2ac |
| `name-mismatch-diagnostic` | `True` | reference tensorName does not match engine output name/reference tensorName does not match engine output name; actual=Plus214_Output_0; expected=Wrong_Output_0 |
| `name-mismatch-runtime-files` | `True` |  |
| `name-mismatch-runtime-file-hashes` | `True` | all-recorded-hashes-match |
| `name-mismatch-source-runtime-contract` | `True` | False/True/False/True/False |
| `name-mismatch-consumer-runtime-contract` | `True` | package-reference/enqueue/fail-closed/owner-exit |
| `shape-mismatch-source-execution` | `True` | 2/True/True/False |
| `shape-mismatch-source-validation` | `True` | False/False/0/0/-1 |
| `shape-mismatch-source-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/3aea37791dc16e63c77bcadc5183d45b6203387367a79f14ea0ec8cddc7315fa |
| `shape-mismatch-consumer-execution` | `True` | 1/True/True/False |
| `shape-mismatch-consumer-validation` | `True` | False/False/0/0/-1 |
| `shape-mismatch-consumer-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/3aea37791dc16e63c77bcadc5183d45b6203387367a79f14ea0ec8cddc7315fa |
| `shape-mismatch-diagnostic` | `True` | reference shape does not match runtime output shape/reference shape does not match runtime output shape; actual=[1,10]; expected=[1,2,5] |
| `shape-mismatch-runtime-files` | `True` |  |
| `shape-mismatch-runtime-file-hashes` | `True` | all-recorded-hashes-match |
| `shape-mismatch-source-runtime-contract` | `True` | False/True/False/True/False |
| `shape-mismatch-consumer-runtime-contract` | `True` | package-reference/enqueue/fail-closed/owner-exit |
| `value-count-mismatch-source-execution` | `True` | 2/True/True/False |
| `value-count-mismatch-source-validation` | `True` | False/False/0/0/-1 |
| `value-count-mismatch-source-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/9ca5337e678c5941a8ed3cb88dde96ca0f44c8abf97ac35c487785c425e81ae8 |
| `value-count-mismatch-consumer-execution` | `True` | 1/True/True/False |
| `value-count-mismatch-consumer-validation` | `True` | False/False/0/0/-1 |
| `value-count-mismatch-consumer-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/9ca5337e678c5941a8ed3cb88dde96ca0f44c8abf97ac35c487785c425e81ae8 |
| `value-count-mismatch-diagnostic` | `True` | reference value count does not match runtime output element count/reference value count does not match runtime output element count; actual=10; expected=9 |
| `value-count-mismatch-runtime-files` | `True` |  |
| `value-count-mismatch-runtime-file-hashes` | `True` | all-recorded-hashes-match |
| `value-count-mismatch-source-runtime-contract` | `True` | False/True/False/True/False |
| `value-count-mismatch-consumer-runtime-contract` | `True` | package-reference/enqueue/fail-closed/owner-exit |
| `nan-reject-source-execution` | `True` | 2/True/True/False |
| `nan-reject-source-validation` | `True` | True/False/10/1/0 |
| `nan-reject-source-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/37a1c28eda29c01c43912301f57a89e22031ae07de6420e0a134f67e294dfd8d |
| `nan-reject-consumer-execution` | `True` | 1/True/True/False |
| `nan-reject-consumer-validation` | `True` | True/False/10/1/0 |
| `nan-reject-consumer-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/37a1c28eda29c01c43912301f57a89e22031ae07de6420e0a134f67e294dfd8d |
| `nan-reject-diagnostic` | `True` | 1 value(s) exceeded tolerance or special-value policy; first mismatch index 0/1 value(s) exceeded tolerance or special-value policy; first mismatch index 0 |
| `nan-reject-runtime-files` | `True` |  |
| `nan-reject-runtime-file-hashes` | `True` | all-recorded-hashes-match |
| `nan-reject-source-runtime-contract` | `True` | False/True/False/True/False |
| `nan-reject-consumer-runtime-contract` | `True` | package-reference/enqueue/fail-closed/owner-exit |
| `infinity-reject-source-execution` | `True` | 2/True/True/False |
| `infinity-reject-source-validation` | `True` | True/False/10/1/0 |
| `infinity-reject-source-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/6b3bd2f292bb21ab2764c8698a3d1c60c960aa9d9b67457aa413457da51e5000 |
| `infinity-reject-consumer-execution` | `True` | 1/True/True/False |
| `infinity-reject-consumer-validation` | `True` | True/False/10/1/0 |
| `infinity-reject-consumer-hashes` | `True` | 0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/6b3bd2f292bb21ab2764c8698a3d1c60c960aa9d9b67457aa413457da51e5000 |
| `infinity-reject-diagnostic` | `True` | 1 value(s) exceeded tolerance or special-value policy; first mismatch index 0/1 value(s) exceeded tolerance or special-value policy; first mismatch index 0 |
| `infinity-reject-runtime-files` | `True` |  |
| `infinity-reject-runtime-file-hashes` | `True` | all-recorded-hashes-match |
| `infinity-reject-source-runtime-contract` | `True` | False/True/False/True/False |
| `infinity-reject-consumer-runtime-contract` | `True` | package-reference/enqueue/fail-closed/owner-exit |
| `proof-runtime-boundary` | `True` | True/True/True |
| `proof-promotion-boundary` | `True` | False/False/False/False/False |
| `proof-statement` | `True` | Controlled malformed references prove source-tree and isolated local PackageReference consumers fail closed only after real TensorRT enqueue and output readback. They do not promote the unreviewed ONNX Runtime reference to an Owner golden, public-package, post-publish, or release proof. |

Controlled negative references prove fail-closed behavior after real enqueue/readback; they do not promote Owner, public-package, post-publish, or release proof.
