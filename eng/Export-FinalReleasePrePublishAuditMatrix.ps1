[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Read-TextOrEmpty {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return "" }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-AuditItem {
  param(
    [string]$Id,
    [string]$Area,
    [string]$State,
    [string]$EvidenceClass,
    [string]$Boundary,
    [string[]]$SourceArtifacts,
    [string]$NextAction,
    [bool]$Ready = $false
  )

  [pscustomobject]@{
    id = $Id
    area = $Area
    state = $State
    evidenceClass = $EvidenceClass
    ready = $Ready
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeProof = $false
    isPackageConsumerRuntimeProof = $false
    boundary = $Boundary
    sourceArtifacts = @($SourceArtifacts)
    nextAction = $NextAction
  }
}

$releaseEvidence = Read-JsonOrNull "artifacts/final-release/release-evidence-bundle.json"
$dryRun = Read-JsonOrNull "artifacts/final-release/final-release-dry-run-summary.json"
$closeBlockerDashboard = Read-JsonOrNull "artifacts/final-release/final-release-close-blocker-dashboard.json"
$featureMatrix = Read-JsonOrNull "applications/TensorRtExec/tensor-rt-exec-feature-matrix.json"
$tensorRtExecGapList = Read-JsonOrNull "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json"
$tensorRtExecParity = Read-JsonOrNull "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json"
$yoloMatrix = Read-JsonOrNull "samples/YoloVision/yolo-model-matrix.json"
$onnxParity = Read-JsonOrNull "samples/OnnxToEngine/trtexec-parity-matrix.json"
$docsReadiness = Read-JsonOrNull "artifacts/final-release/docs-publish-readiness-bundle.json"
$articleMatrix = Read-JsonOrNull "artifacts/final-release/technical-article-publication-matrix.json"
$postPublishPack = Read-JsonOrNull "artifacts/final-release/final-post-publish-audit-pack.json"
$ownerRuntimeSmokeFieldAlignment = Read-JsonOrNull "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json"
$ownerRuntimeSmokeFieldAlignmentValidation = Read-JsonOrNull "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json"

$yoloReadme = Read-TextOrEmpty "samples/YoloVision/README.md"
$onnxReadme = Read-TextOrEmpty "samples/OnnxToEngine/README.md"
$tensorRtExecReadme = Read-TextOrEmpty "applications/TensorRtExec/README.md"

$tensorRtExecGapItems = ConvertTo-Array (Get-PropertyOrDefault -Object $tensorRtExecGapList -Name "items" -DefaultValue @())
$tensorRtExecRuntimeProofItems = @($tensorRtExecGapItems | Where-Object {
  [bool](Get-PropertyOrDefault -Object $_ -Name "isRuntimeProof" -DefaultValue $false) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isPackageConsumerRuntimeProof" -DefaultValue $false)
})

$tensorRtExecParityEntries = ConvertTo-Array (Get-PropertyOrDefault -Object $tensorRtExecParity -Name "entries" -DefaultValue @())
$onnxParityEntries = ConvertTo-Array (Get-PropertyOrDefault -Object $onnxParity -Name "entries" -DefaultValue @())
$yoloFamilies = @((ConvertTo-Array (Get-PropertyOrDefault -Object $yoloMatrix -Name "families" -DefaultValue @())) | ForEach-Object { [string]$_ })
$yoloTasks = @((ConvertTo-Array (Get-PropertyOrDefault -Object $yoloMatrix -Name "tasks" -DefaultValue @())) | ForEach-Object { [string]$_ })

$requiredYoloFamilies = @("yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "custom")
$requiredYoloTasks = @("det", "cls", "seg", "obb", "pose", "sem")
$missingYoloFamilies = @($requiredYoloFamilies | Where-Object { $yoloFamilies -notcontains $_ })
$missingYoloTasks = @($requiredYoloTasks | Where-Object { $yoloTasks -notcontains $_ })

$packageArtifactsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "packageArtifactsReady" -DefaultValue $false)
$runtimePackagesReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "runtimePackagesReady" -DefaultValue $false)
$cleanOwnerInputReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "cleanOwnerInputReady" -DefaultValue $false)
$ownerInputCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerInputCanPromoteRuntimeProof" -DefaultValue $false)
$externalRuntimeProofReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofReady" -DefaultValue $false)
$postPublishVerificationReady = [bool](Get-PropertyOrDefault -Object $postPublishPack -Name "postPublishVerificationReady" -DefaultValue $false)
$docsArticlesReady = [bool](Get-PropertyOrDefault -Object $docsReadiness -Name "readyForPublish" -DefaultValue $false)
$ownerRuntimeSmokeFieldAlignmentState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "alignmentState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment")
$ownerRuntimeSmokeFieldAlignmentValidationState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "validationState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment-validation")
$ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "runtimeSmokeStatus" -DefaultValue "Smoke=missing")
$ownerRuntimeSmokeFieldAlignmentFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "fieldCount" -DefaultValue 0)
$ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "missingRequiredFieldCount" -DefaultValue -1)
$ownerRuntimeSmokeFieldAlignmentFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "failedBlockerCount" -DefaultValue -1)

$auditItems = @(
  New-AuditItem `
    -Id "owner-runtime-smoke-field-alignment" `
    -Area "release-proof" `
    -State "alignmentState=$ownerRuntimeSmokeFieldAlignmentState; validationState=$ownerRuntimeSmokeFieldAlignmentValidationState; runtimeSmokeStatus=$ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus; fieldCount=$ownerRuntimeSmokeFieldAlignmentFieldCount; missingRequiredFields=$ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount; failedBlockers=$ownerRuntimeSmokeFieldAlignmentFailedBlockerCount" `
    -EvidenceClass "field-alignment-non-proof" `
    -Boundary "Owner runtime smoke field alignment proves only that owner-facing fields are named and propagated; it is not runtime proof, publish approval, or release close approval." `
    -SourceArtifacts @("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json", "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json") `
    -NextAction "Keep this as a zero-missing owner field contract while waiting for real compatible-host runtime smoke."
  New-AuditItem `
    -Id "release-evidence-current-bundle" `
    -Area "release-proof" `
    -State ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")) `
    -EvidenceClass "readiness-audit" `
    -Boundary "Release evidence bundle is an aggregator; it does not run TensorRT, publish packages, or close the release." `
    -SourceArtifacts @("artifacts/final-release/release-evidence-bundle.json") `
    -NextAction "Keep exporting the bundle before final dry-run and close blocker dashboard checks."
  New-AuditItem `
    -Id "final-dry-run-blocked-by-owner-proof" `
    -Area "release-proof" `
    -State ([string](Get-PropertyOrDefault -Object $dryRun -Name "summaryState" -DefaultValue "missing-final-release-dry-run-summary")) `
    -EvidenceClass "dry-run-audit" `
    -Boundary "Final dry-run can show readiness gaps only; AllowRuntimeSmokeBlocked never means smoke passed." `
    -SourceArtifacts @("artifacts/final-release/final-release-dry-run-summary.json") `
    -NextAction "Import real owner proof and rerun strict dry-run only after clean external consumer evidence exists."
  New-AuditItem `
    -Id "close-blocker-dashboard-current" `
    -Area "release-close" `
    -State ([string](Get-PropertyOrDefault -Object $closeBlockerDashboard -Name "dashboardState" -DefaultValue "missing-final-release-close-blocker-dashboard")) `
    -EvidenceClass "blocker-dashboard" `
    -Boundary "Close blocker dashboard is not issue-close approval and must keep canCloseReleaseIssue=false until real post-publish proof exists." `
    -SourceArtifacts @("artifacts/final-release/final-release-close-blocker-dashboard.json") `
    -NextAction "Use this dashboard as the release issue worklist, not as a substitute for owner proof."
  New-AuditItem `
    -Id "tensorrtexec-cli-winforms-parity" `
    -Area "TensorRtExec" `
    -State ([string](Get-PropertyOrDefault -Object $featureMatrix -Name "matrixId" -DefaultValue "missing-tensor-rt-exec-feature-matrix")) `
    -EvidenceClass "tool-readiness-matrix" `
    -Boundary "TensorRtExec CLI/WinForms reports, dry-runs, GUI screenshots, and build-only outputs are not runtime proof." `
    -SourceArtifacts @("applications/TensorRtExec/tensor-rt-exec-feature-matrix.json", "applications/TensorRtExec/README.md") `
    -NextAction "Continue closing parity gaps while keeping package-consumer-runtime proof owned by strict release proof records."
  New-AuditItem `
    -Id "tensorrtexec-gap-list-non-proof" `
    -Area "TensorRtExec" `
    -State ([string](Get-PropertyOrDefault -Object $tensorRtExecGapList -Name "state" -DefaultValue "missing-tensor-rt-exec-gap-list")) `
    -EvidenceClass "gap-planning" `
    -Boundary "Gap list entries must keep isRuntimeProof=false and isPackageConsumerRuntimeProof=false." `
    -SourceArtifacts @("applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json") `
    -NextAction "Use the gap list to schedule implementation, not to promote proof."
  New-AuditItem `
    -Id "tensorrtexec-trtexec-parity-boundary" `
    -Area "TensorRtExec" `
    -State ([string](Get-PropertyOrDefault -Object $tensorRtExecParity -Name "matrixState" -DefaultValue "missing-tensor-rt-exec-trtexec-parity-matrix")) `
    -EvidenceClass "trtexec-parity-matrix" `
    -Boundary "Trtexec parity entries are implementation diagnostics; external runtime claims still require real model and clean consumer proof." `
    -SourceArtifacts @("applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json") `
    -NextAction "Keep load-engine, timing cache, plugins, and binding metadata guarded until runtime evidence exists."
  New-AuditItem `
    -Id "yolovision-family-task-matrix" `
    -Area "YoloVision" `
    -State ([string](Get-PropertyOrDefault -Object $yoloMatrix -Name "matrixState" -DefaultValue "missing-yolovision-model-matrix")) `
    -EvidenceClass "sample-support-matrix" `
    -Boundary "YoloVision matrix/README can promote only to real-model-runtime after real assets, logs, hashes, and sample-run-evidence validation; never package-consumer-runtime." `
    -SourceArtifacts @("samples/YoloVision/yolo-model-matrix.json", "samples/YoloVision/README.md") `
    -NextAction "Backfill owner real assets for representative det/seg/pose/obb/cls/sem paths."
  New-AuditItem `
    -Id "onnxtoengine-trtexec-like-boundary" `
    -Area "OnnxToEngine" `
    -State ([string](Get-PropertyOrDefault -Object $onnxParity -Name "matrixId" -DefaultValue "missing-onnx-to-engine-trtexec-parity-matrix")) `
    -EvidenceClass "build-report-parity" `
    -Boundary "OnnxToEngine parse-only, preflight-only, report-only, and build-only records are not runtime proof, real-model-runtime proof, or package-consumer-runtime proof." `
    -SourceArtifacts @("samples/OnnxToEngine/trtexec-parity-matrix.json", "samples/OnnxToEngine/README.md") `
    -NextAction "Use OnnxToEngine to produce build reports, then require separate sample runtime and owner proof records."
  New-AuditItem `
    -Id "docs-articles-readiness-boundary" `
    -Area "docs" `
    -State ([string](Get-PropertyOrDefault -Object $docsReadiness -Name "bundleState" -DefaultValue "missing-docs-publish-readiness-bundle")) `
    -EvidenceClass "documentation-readiness" `
    -Boundary "Docs and articles are release communication assets; they do not prove runtime execution or post-publish verification." `
    -SourceArtifacts @("artifacts/final-release/docs-publish-readiness-bundle.json", "artifacts/final-release/technical-article-publication-matrix.json") `
    -NextAction "Keep public articles aligned with blocked proof states until real owner evidence exists."
)

$sourceArtifacts = @(
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-release-dry-run-summary.json",
  "artifacts/final-release/final-release-close-blocker-dashboard.json",
  "applications/TensorRtExec/tensor-rt-exec-feature-matrix.json",
  "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
  "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
  "applications/TensorRtExec/README.md",
  "samples/YoloVision/yolo-model-matrix.json",
  "samples/YoloVision/README.md",
  "samples/OnnxToEngine/trtexec-parity-matrix.json",
  "samples/OnnxToEngine/README.md",
  "artifacts/final-release/docs-publish-readiness-bundle.json",
  "artifacts/final-release/technical-article-publication-matrix.json",
  "artifacts/final-release/final-post-publish-audit-pack.json",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json"
)

$remainingBlockerCount = @($auditItems | Where-Object { -not [bool]$_.ready }).Count

$record = [pscustomobject]@{
  recordKind = "final-release-pre-publish-audit-matrix"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = "blocked-final-pre-publish-owner-proof-required"
  packageArtifactsReady = $packageArtifactsReady
  runtimePackagesReady = $runtimePackagesReady
  cleanOwnerInputReady = $cleanOwnerInputReady
  ownerInputCanPromoteRuntimeProof = $ownerInputCanPromoteRuntimeProof
  externalRuntimeProofReady = $externalRuntimeProofReady
  postPublishVerificationReady = $postPublishVerificationReady
  packageConsumerOwnerRuntimeSmokeFieldAlignmentState = $ownerRuntimeSmokeFieldAlignmentState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState = $ownerRuntimeSmokeFieldAlignmentValidationState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = $ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount = $ownerRuntimeSmokeFieldAlignmentFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = $ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount = $ownerRuntimeSmokeFieldAlignmentFailedBlockerCount
  tensorRtExecParityState = [string](Get-PropertyOrDefault -Object $tensorRtExecParity -Name "matrixState" -DefaultValue "missing")
  tensorRtExecGapState = [string](Get-PropertyOrDefault -Object $tensorRtExecGapList -Name "state" -DefaultValue "missing")
  tensorRtExecRuntimeProofItems = $tensorRtExecRuntimeProofItems.Count
  tensorRtExecEntryCount = $tensorRtExecParityEntries.Count
  yoloVisionEvidenceState = [string](Get-PropertyOrDefault -Object $yoloMatrix -Name "matrixState" -DefaultValue "missing")
  yoloVisionFamilies = @($yoloFamilies)
  yoloVisionTasks = @($yoloTasks)
  yoloVisionMissingFamilies = @($missingYoloFamilies)
  yoloVisionMissingTasks = @($missingYoloTasks)
  onnxToEngineParityState = [string](Get-PropertyOrDefault -Object $onnxParity -Name "matrixId" -DefaultValue "missing")
  onnxToEngineEntryCount = $onnxParityEntries.Count
  docsArticlesReady = $docsArticlesReady
  docsArticleMatrixPresent = ($null -ne $articleMatrix)
  remainingBlockerCount = $remainingBlockerCount
  auditItems = @($auditItems)
  sourceArtifacts = @($sourceArtifacts)
  performsPublish = $false
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isRealModelRuntimeProof = $false
  isPostPublishProof = $false
  retiredSampleLiveProjectPathPresent = (Test-Path -LiteralPath (Join-Path $RepositoryRoot "samples\YoloDet") -PathType Container)
  retiredSampleLiveProjectFilePresent = (Test-Path -LiteralPath (Join-Path $RepositoryRoot "samples\YoloVision\YoloDet.csproj") -PathType Leaf)
  tensorRtExecBoundary = [string](Get-PropertyOrDefault -Object $tensorRtExecParity -Name "proofBoundary" -DefaultValue "")
  yoloVisionBoundary = [string](Get-PropertyOrDefault -Object $yoloMatrix -Name "proofBoundary" -DefaultValue "")
  onnxToEngineBoundary = [string](Get-PropertyOrDefault -Object $onnxParity -Name "proofBoundary" -DefaultValue "")
  readmeBoundarySignals = [pscustomobject]@{
    tensorRtExecMentionsBuildReportNotProof = $tensorRtExecReadme.Contains("build report", [StringComparison]::OrdinalIgnoreCase) -and $tensorRtExecReadme.Contains("不能直接声明", [StringComparison]::OrdinalIgnoreCase)
    yoloVisionMentionsRequiredFamilies = (
      (
        $yoloReadme.Contains("YOLOv5/v6/v7/v8/v9/v10/v11/v26", [StringComparison]::OrdinalIgnoreCase) -or
        $yoloReadme.Contains("YOLO v5/v6/v7/v8/v9/v10/v11/v26", [StringComparison]::OrdinalIgnoreCase)
      ) -and
      $yoloReadme.Contains("custom", [StringComparison]::OrdinalIgnoreCase)
    )
    yoloVisionMentionsRequiredTasks = ($requiredYoloTasks | Where-Object { -not $yoloReadme.Contains("--task $_", [StringComparison]::OrdinalIgnoreCase) }).Count -eq 0
    onnxToEngineMentionsBuildOnlyBoundary = $onnxReadme.Contains("build-only", [StringComparison]::OrdinalIgnoreCase) -and $onnxReadme.Contains("not", [StringComparison]::OrdinalIgnoreCase)
  }
  boundary = "Final release pre-publish audit matrix is blocked readiness evidence only; it does not publish packages, approve public release, close a release issue, promote runtime proof, or replace clean external package-consumer-runtime proof."
}

$jsonPath = Join-Path $OutputRoot "final-release-pre-publish-audit-matrix.json"
$markdownPath = Join-Path $OutputRoot "final-release-pre-publish-audit-matrix.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$itemRows = $auditItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.area)`` | ``$($_.state)`` | ``$($_.evidenceClass)`` | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.boundary) | $(ConvertTo-MarkdownCell $_.nextAction) |"
}

$sourceRows = $sourceArtifacts | ForEach-Object { "- ``$_``" }

$markdown = @"
# Final Release Pre-Publish Audit Matrix

| Field | Value |
| --- | --- |
| recordKind | ``$($record.recordKind)`` |
| auditState | ``$($record.auditState)`` |
| cleanOwnerInputReady | ``$($record.cleanOwnerInputReady)`` |
| ownerInputCanPromoteRuntimeProof | ``$($record.ownerInputCanPromoteRuntimeProof)`` |
| externalRuntimeProofReady | ``$($record.externalRuntimeProofReady)`` |
| postPublishVerificationReady | ``$($record.postPublishVerificationReady)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentState | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentState)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount)`` |
| tensorRtExecRuntimeProofItems | ``$($record.tensorRtExecRuntimeProofItems)`` |
| yoloVisionFamilies | ``$($record.yoloVisionFamilies -join ', ')`` |
| yoloVisionTasks | ``$($record.yoloVisionTasks -join ', ')`` |
| remainingBlockerCount | ``$($record.remainingBlockerCount)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |

## Audit Items

| ID | Area | State | Evidence Class | Ready | Boundary | Next Action |
| --- | --- | --- | --- | ---: | --- | --- |
$($itemRows -join "`r`n")

## Source Artifacts

$($sourceRows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final release pre-publish audit matrix written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "AuditState=$($record.auditState) Items=$($auditItems.Count) RemainingBlockers=$remainingBlockerCount"
