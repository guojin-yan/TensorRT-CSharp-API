# Cross-Task Reference Provenance Validation

- strict: `True`
- checks: `94`
- passed: `94`
- failed: `0`

| Check | Passed | Actual |
| --- | --- | --- |
| `matrix-schema` | `True` | cross-task-reference-provenance-matrix.v1 |
| `matrix-state` | `True` | owner-action-required |
| `matrix-counts` | `True` | 7/0/7/7 |
| `matrix-row-order` | `True` | classification,yolo-det,yolo-cls,yolo-seg,yolo-obb,yolo-pose,yolo-sem |
| `matrix-task-order` | `True` | classification,det,cls,seg,obb,pose,sem |
| `matrix-path-free` | `True` | absolute-windows-path-present=False |
| `contract-schema` | `True` | cross-task-reference-provenance-contract.v1/cross-task-reference-provenance-contract |
| `contract-state` | `True` | owner-review-required |
| `contract-common-sections` | `True` | assetIdentity,tensorContract,executionIdentity,referenceIdentity,comparisonPolicy,ownerDecision |
| `contract-common-fields` | `True` | empty-sections=0 |
| `contract-task-profiles` | `True` | classification,yolo-det,yolo-cls,yolo-seg,yolo-obb,yolo-pose,yolo-sem |
| `contract-task-semantics` | `True` | invalid-profiles=0 |
| `contract-reuse-fingerprints` | `True` | modelSha256,inputTensorSha256,preprocessContractSha256,outputTensorContractSha256,labelsSha256,taskSemanticsSha256 |
| `contract-reuse-rules` | `True` | count=4 |
| `contract-promotion-boundary` | `True` | False/False/False/False |
| `matrix-contract-cross-check` | `True` | samples/assets/cross-task-reference-provenance-contract.json/1c48e81bbb066dd1b79a48037e92fc9cb64c6dadd7b6b3a5a4456ca9be9ad925/cross-task-reference-provenance-contract.v1 |
| `source-count` | `True` | 4 |
| `source-roles` | `True` | classification-manifest,yolovision-task-contract,yolovision-owner-input-template,independent-reference-candidate |
| `source-classification-manifest-hash` | `True` | samples/assets/classification-assets.template.json/8085e1b21802b24d4f07ac304bfe8261c847340e06b80e97981d3a0d11b4506b |
| `source-yolovision-task-contract-hash` | `True` | applications/YoloVision/yolovision-task-output-contract.json/6fb42216f9a709621d8fae8623405b3427b60cb6ee1d39024abf2994db615520 |
| `source-yolovision-owner-input-template-hash` | `True` | artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json/6f31053963344baa87188c1feb581928423bc05361291fa302dfdeb2466313dd |
| `source-independent-reference-candidate-hash` | `True` | artifacts/interface-coverage/tensorrtexec-mnist-onnxruntime-reference-evidence.json/434b56912a791c7ed1976ce24897891182754905329de69449d3b1a7e6e0909d |
| `classification-field-counts` | `True` | 6/26/20 |
| `classification-common-fields` | `True` | count=15/duplicates=0 |
| `classification-semantic-fields` | `True` | imageResizePolicy,imageCropPolicy,colorOrder,scale,mean,std,outputValueKind,scoreTransform,labelMappingSha256,topK,argmaxRule |
| `classification-field-state-contract` | `True` | invalid=0 |
| `classification-reference-state` | `True` | not-captured-for-classification/False |
| `classification-promotion-boundary` | `True` | False/False/False |
| `classification-proof-classification` | `True` | template-only/candidate-not-downloaded |
| `classification-boundary` | `True` | Generic Classification top-k output cannot reuse MNIST logits or YoloVision classification semantics unless every reuse fingerprint and Owner decision matches. |
| `yolo-det-field-counts` | `True` | 5/25/20 |
| `yolo-det-common-fields` | `True` | count=15/duplicates=0 |
| `yolo-det-semantic-fields` | `True` | imagePreprocessContract,outputLayout,boxFormat,scoreRule,hasObjectness,nmsMode,scoreThreshold,iouThreshold,coordinateSpace,sourceImageInversePolicy |
| `yolo-det-field-state-contract` | `True` | invalid=0 |
| `yolo-det-reference-state` | `True` | not-captured-for-det/False |
| `yolo-det-promotion-boundary` | `True` | False/False/False |
| `yolo-det-proof-classification` | `True` | template-only/owner-action-required |
| `yolo-det-boundary` | `True` | Detection references require exporter-specific box, score, NMS, and coordinate-transform semantics. |
| `yolo-det-yolo-contract-link` | `True` | inputShape,outputLayout,classCount,boxFormat,scoreRule,nmsMode |
| `yolo-cls-field-counts` | `True` | 5/22/17 |
| `yolo-cls-common-fields` | `True` | count=15/duplicates=0 |
| `yolo-cls-semantic-fields` | `True` | imagePreprocessContract,classCount,labelMappingSha256,classScoreField,softmaxApplied,topK,argmaxRule |
| `yolo-cls-field-state-contract` | `True` | invalid=0 |
| `yolo-cls-reference-state` | `True` | not-captured-for-cls/False |
| `yolo-cls-promotion-boundary` | `True` | False/False/False |
| `yolo-cls-proof-classification` | `True` | template-only/owner-action-required |
| `yolo-cls-boundary` | `True` | YoloVision classification remains distinct from the generic Classification sample and requires its own exporter, labels, and score semantics. |
| `yolo-cls-yolo-contract-link` | `True` | inputShape,classificationOutput,classCount,labelsPath,topK,classificationScoreMode |
| `yolo-seg-field-counts` | `True` | 10/24/14 |
| `yolo-seg-common-fields` | `True` | count=15/duplicates=0 |
| `yolo-seg-semantic-fields` | `True` | imagePreprocessContract,outputRoleMap,maskCoefficientCount,prototypeShape,maskResizePolicy,maskThreshold,maskValueKind,maskSpatialTransform,maskCropToDetection |
| `yolo-seg-field-state-contract` | `True` | invalid=0 |
| `yolo-seg-reference-state` | `True` | not-captured-for-seg/False |
| `yolo-seg-promotion-boundary` | `True` | False/False/False |
| `yolo-seg-proof-classification` | `True` | template-only/owner-action-required |
| `yolo-seg-boundary` | `True` | Segmentation references require boxes, coefficients, prototypes, mask composition, crop, resize, and inverse-transform semantics. |
| `yolo-seg-yolo-contract-link` | `True` | inputShape,outputRoleMap,classCount,maskCoefficientCount,prototypeShape,maskResizePolicy |
| `yolo-obb-field-counts` | `True` | 4/22/18 |
| `yolo-obb-common-fields` | `True` | count=15/duplicates=0 |
| `yolo-obb-semantic-fields` | `True` | imagePreprocessContract,boxFormat,angleUnit,angleRange,rotatedBoxLayout,coordinateSpace,rotatedNmsMode |
| `yolo-obb-field-state-contract` | `True` | invalid=0 |
| `yolo-obb-reference-state` | `True` | not-captured-for-obb/False |
| `yolo-obb-promotion-boundary` | `True` | False/False/False |
| `yolo-obb-proof-classification` | `True` | template-only/owner-action-required |
| `yolo-obb-boundary` | `True` | OBB references require angle units/range, rotated-box layout, coordinate mapping, and rotated NMS semantics. |
| `yolo-obb-yolo-contract-link` | `True` | inputShape,classCount,angleUnit,boxFormat,auxiliaryLayout |
| `yolo-pose-field-counts` | `True` | 5/23/18 |
| `yolo-pose-common-fields` | `True` | count=15/duplicates=0 |
| `yolo-pose-semantic-fields` | `True` | imagePreprocessContract,keypointCount,keypointStride,coordinateLayout,keypointLayout,keypointScoreField,skeletonMap,sourceImageInversePolicy |
| `yolo-pose-field-state-contract` | `True` | invalid=0 |
| `yolo-pose-reference-state` | `True` | not-captured-for-pose/False |
| `yolo-pose-promotion-boundary` | `True` | False/False/False |
| `yolo-pose-proof-classification` | `True` | template-only/owner-action-required |
| `yolo-pose-boundary` | `True` | Pose references require keypoint count/stride/layout, score fields, skeleton identity, and inverse-coordinate semantics. |
| `yolo-pose-yolo-contract-link` | `True` | inputShape,outputRoleMap,classCount,keypointCount,keypointStride,coordinateLayout |
| `yolo-sem-field-counts` | `True` | 5/23/18 |
| `yolo-sem-common-fields` | `True` | count=15/duplicates=0 |
| `yolo-sem-semantic-fields` | `True` | imagePreprocessContract,semanticOutputRole,semanticMapShape,classMapLayout,argmaxRule,paletteSha256,voidClassPolicy,sourceImageInversePolicy |
| `yolo-sem-field-state-contract` | `True` | invalid=0 |
| `yolo-sem-reference-state` | `True` | not-captured-for-sem/False |
| `yolo-sem-promotion-boundary` | `True` | False/False/False |
| `yolo-sem-proof-classification` | `True` | template-only/owner-action-required |
| `yolo-sem-boundary` | `True` | Semantic segmentation references require class-axis/argmax, map layout, palette, void-class, resize, and inverse-transform semantics. |
| `yolo-sem-yolo-contract-link` | `True` | inputShape,semanticOutput,classCount,mapWidth,mapHeight,argmaxRule |
| `classification-runtime-contract` | `True` | implemented-managed-contract-owner-assets-required/artifacts=4/exists=True/runtimeProof=False/reference=not-captured-for-classification/ownerGolden=False |
| `candidate-count` | `True` | 1 |
| `candidate-identity` | `True` | mnist-onnxruntime-cpu-1.23.2/mnist-classification/independent-framework-reference-candidate-runtime |
| `candidate-provider` | `True` | ONNX Runtime/1.23.2/CPUExecutionProvider/True/True |
| `candidate-hashes` | `True` | 2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf/81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564/1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571 |
| `candidate-task-isolation` | `True` | mnist/classification,yolo-det,yolo-cls,yolo-seg,yolo-obb,yolo-pose,yolo-sem/False |
| `candidate-owner-boundary` | `True` | False/False/False |
| `candidate-reason` | `True` | The MNIST model/input/preprocess/output/labels/task semantics do not match the generic Classification sample or any YoloVision task profile. |
| `matrix-proof-boundary` | `True` | False/False/False/False/False/False |
| `matrix-proof-statement` | `True` | This matrix audits provenance readiness and task semantics. It does not create reference output, approve licenses, accept an Owner golden, prove public-package consumption, or provide post-publish/release proof. |

The validator audits contract structure and current readiness only; it does not create or promote runtime, Owner, package, publication, or release proof.
