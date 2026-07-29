# Cross-Task Reference Provenance Matrix

- state: `owner-action-required`
- rows: `7`
- ready rows: `0`
- owner action required rows: `7`

| Row | Sample | Task | Runtime contract | Ready / Required | Missing | Independent reference | Owner golden |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `classification` | Classification | `classification` | `implemented-managed-contract-owner-assets-required` | 6 / 26 | 20 | `not-captured-for-classification` | `False` |
| `yolo-det` | YoloVision | `det` | `not-audited-in-this-batch` | 5 / 25 | 20 | `not-captured-for-det` | `False` |
| `yolo-cls` | YoloVision | `cls` | `not-audited-in-this-batch` | 5 / 22 | 17 | `not-captured-for-cls` | `False` |
| `yolo-seg` | YoloVision | `seg` | `not-audited-in-this-batch` | 10 / 24 | 14 | `not-captured-for-seg` | `False` |
| `yolo-obb` | YoloVision | `obb` | `not-audited-in-this-batch` | 4 / 22 | 18 | `not-captured-for-obb` | `False` |
| `yolo-pose` | YoloVision | `pose` | `not-audited-in-this-batch` | 5 / 23 | 18 | `not-captured-for-pose` | `False` |
| `yolo-sem` | YoloVision | `sem` | `not-audited-in-this-batch` | 5 / 23 | 18 | `not-captured-for-sem` | `False` |

## Independent Candidates

- `mnist-onnxruntime-cpu-1.23.2`: provider `CPUExecutionProvider` / deterministic `True` / eligible only for `mnist` / reusable for matrix rows `False`.

This matrix audits provenance readiness and task semantics. It does not create reference output, approve licenses, accept an Owner golden, prove public-package consumption, or provide post-publish/release proof.
