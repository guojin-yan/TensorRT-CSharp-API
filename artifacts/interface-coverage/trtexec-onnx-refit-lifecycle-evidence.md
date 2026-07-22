# TensorRtExec ONNX Stripped-Plan Refit Lifecycle Evidence

## Result

TensorRT 10.11 / CUDA 12.9 completed the full in-memory lifecycle:

1. Build an MNIST plan with `StripPlan + Refit`.
2. Deserialize the stripped plan and confirm `IsRefittable=True`.
3. Copy missing/all named-weight inventories.
4. Load ONNX weights through `IParserRefitter::refitFromFile`.
5. Commit loaded weights through `IRefitter::refitCudaEngine`.
6. Require parser errors `0`, missing weights `0`, engine refittable readback, and both refit calls to return true.
7. Create the execution context only after the refit gate passes.
8. Enqueue the same 784-float MNIST input used by a full-weight baseline.

The refitted and full-weight runs each produced 10 floats / 40 bytes. Their raw output SHA256 values are identical:

`6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041`

## Inventory

| Field | Before | After |
| --- | ---: | ---: |
| Missing named weights | 0 | 0 |
| All refittable named weights | 6 | 6 |
| Parser errors | 0 | 0 |

`refitFromFile` alone is not sufficient: an observed pre-commit run returned true but produced zero logits. The lifecycle therefore requires the subsequent `refitCudaEngine` commit, and the structured snapshot records both results independently.

## Version Guards

- TensorRT 8 accepts the option only in dry-run/precheck reporting and rejects non-dry execution because the ONNX parser-refitter API is unavailable.
- TensorRT 11 keeps the option parse-only on this host because runtime creation returns null; this is `dependency-probe-only`, not applied refit evidence.

## Boundary

This is local source-tree refit, copied diagnostics/inventory, bounded enqueue, and baseline comparison evidence. The saved plan remains the stripped artifact; the successful refit is in memory. This does not prove model accuracy, refitted-plan persistence, package-consumer runtime, public packages, or release readiness. No public release side effect was executed.
