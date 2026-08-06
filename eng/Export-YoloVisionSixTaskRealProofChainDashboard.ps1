[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [string]$EvidenceRoot = "samples/assets"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\yolovision"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolvedPath = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$contractPath = "applications/YoloVision/yolovision-task-output-contract.json"
$candidateValidationPath = "artifacts/yolovision/yolovision-real-asset-candidate-validation.json"
$ownerBackfillValidationPath = "artifacts/yolovision/yolovision-real-asset-owner-backfill-pack-validation.json"
$ownerProofInputValidationPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json"
$ownerProofImportReportPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-import-report.json"
$candidateEvidencePath = "artifacts/user-acceptance/yolovision-real-asset-owner-sample-run-evidence.candidate.json"
$finalFreezePath = "artifacts/final-release/release-candidate-final-evidence-freeze.json"
$realModelEvidencePaths = @{
  det = Join-Path $EvidenceRoot "yolovision-yolov8n-det-real-model-runtime-evidence.json"
  cls = Join-Path $EvidenceRoot "yolovision-yolov8n-cls-real-model-runtime-evidence.json"
  seg = Join-Path $EvidenceRoot "yolovision-yolov8n-seg-real-model-runtime-evidence.json"
  obb = Join-Path $EvidenceRoot "yolovision-yolov8n-obb-real-model-runtime-evidence.json"
  pose = Join-Path $EvidenceRoot "yolovision-yolov8n-pose-real-model-runtime-evidence.json"
  sem = Join-Path $EvidenceRoot "yolovision-torchvision-lraspp-real-model-runtime-evidence.json"
}

$contract = Read-JsonOrNull $contractPath
if ($null -eq $contract) {
  throw "YoloVision task output contract not found: $contractPath"
}

$candidateValidation = Read-JsonOrNull $candidateValidationPath
$ownerBackfillValidation = Read-JsonOrNull $ownerBackfillValidationPath
$ownerProofInputValidation = Read-JsonOrNull $ownerProofInputValidationPath
$ownerProofImportReport = Read-JsonOrNull $ownerProofImportReportPath
$candidateEvidence = Read-JsonOrNull $candidateEvidencePath
$finalFreeze = Read-JsonOrNull $finalFreezePath

$candidateRecords = @()
if ($null -ne $candidateValidation -and $candidateValidation.PSObject.Properties.Name -contains "records") {
  $candidateRecords = @($candidateValidation.records)
}

$ownerBackfillRecords = @()
if ($null -ne $ownerBackfillValidation -and $ownerBackfillValidation.PSObject.Properties.Name -contains "records") {
  $ownerBackfillRecords = @($ownerBackfillValidation.records)
}

$candidateEvidenceCases = @()
if ($null -ne $candidateEvidence -and $candidateEvidence.PSObject.Properties.Name -contains "cases") {
  $candidateEvidenceCases = @($candidateEvidence.cases)
}

$ownerValidationItems = @()
if ($null -ne $ownerProofInputValidation -and $ownerProofInputValidation.PSObject.Properties.Name -contains "validationItems") {
  $ownerValidationItems = @($ownerProofInputValidation.validationItems)
}

$tasks = @($contract.tasks)
$dashboardItems = @()

foreach ($taskContract in $tasks) {
  $task = [string]$taskContract.task
  $realModelEvidencePath = [string]$realModelEvidencePaths[$task]
  $realModelEvidence = Read-JsonOrNull $realModelEvidencePath
  $runtimeReferenceValidation = Get-PropertyOrDefault -Object $realModelEvidence -Name "runtimeReferenceValidation" -DefaultValue $null
  $controlledNegativeValidation = Get-PropertyOrDefault -Object $realModelEvidence -Name "controlledNegativeValidation" -DefaultValue $null
  $proofBoundary = Get-PropertyOrDefault -Object $realModelEvidence -Name "proofBoundary" -DefaultValue $null
  $candidateRecord = @($candidateRecords | Where-Object { [string]$_.task -eq $task } | Select-Object -First 1)
  $ownerBackfillRecord = @($ownerBackfillRecords | Where-Object { [string]$_.task -eq $task } | Select-Object -First 1)
  $candidateEvidenceRecord = @($candidateEvidenceCases | Where-Object { [string]$_.task -eq $task } | Select-Object -First 1)

  $ownerCaseItems = @($ownerValidationItems | Where-Object {
      $id = [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
      $id -like "case-yolov8n-$task-*"
    })
  $ownerProofInputAligned = ($ownerCaseItems.Count -gt 0) -and (@($ownerCaseItems | Where-Object { [string]$_.severity -eq "blocker" -and -not [bool]$_.passed }).Count -eq 0)
  $ownerActionRequiredCount = @($ownerCaseItems | Where-Object { [string]$_.severity -eq "owner-action-required" -and -not [bool]$_.passed }).Count

  $requiredMetadata = @($taskContract.requiredMetadata | ForEach-Object { [string]$_ })
  $candidateMetadata = @()
  if ($candidateRecord.Count -gt 0) {
    $candidateMetadata = @($candidateRecord[0].contractRequiredMetadata | ForEach-Object { [string]$_ })
  }
  $ownerBackfillMetadata = @()
  if ($ownerBackfillRecord.Count -gt 0) {
    $ownerBackfillMetadata = @($ownerBackfillRecord[0].contractRequiredMetadata | ForEach-Object { [string]$_ })
  }

  $candidateTemplateAligned = $candidateRecord.Count -gt 0 -and
    [int](Get-PropertyOrDefault -Object $candidateRecord[0] -Name "failedBlockerCount" -DefaultValue 1) -eq 0 -and
    -not [bool](Get-PropertyOrDefault -Object $candidateRecord[0] -Name "canPromoteRealModelRuntime" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $candidateRecord[0] -Name "canPromotePackageConsumerRuntime" -DefaultValue $true) -and
    (@(Compare-Object -ReferenceObject $requiredMetadata -DifferenceObject $candidateMetadata).Count -eq 0)

  $ownerBackfillAligned = $ownerBackfillRecord.Count -gt 0 -and
    [int](Get-PropertyOrDefault -Object $ownerBackfillRecord[0] -Name "failedBlockerCount" -DefaultValue 1) -eq 0 -and
    -not [bool](Get-PropertyOrDefault -Object $ownerBackfillRecord[0] -Name "canPromoteRealModelRuntime" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $ownerBackfillRecord[0] -Name "canPromotePackageConsumerRuntime" -DefaultValue $true) -and
    (@(Compare-Object -ReferenceObject $requiredMetadata -DifferenceObject $ownerBackfillMetadata).Count -eq 0)

  $candidateEvidenceAligned = $candidateEvidenceRecord.Count -gt 0 -and
    -not [bool](Get-PropertyOrDefault -Object $candidateEvidenceRecord[0] -Name "canPromoteRealModelRuntime" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $candidateEvidenceRecord[0] -Name "canPromotePackageConsumerRuntime" -DefaultValue $true) -and
    [string](Get-PropertyOrDefault -Object $candidateEvidenceRecord[0] -Name "proofBoundary" -DefaultValue "") -match "never package-consumer-runtime"

  $runtimeReferenceValidated = $null -ne $runtimeReferenceValidation -and
    [bool](Get-PropertyOrDefault -Object $runtimeReferenceValidation -Name "passed" -DefaultValue $false)
  $controlledNegativeValidated = $null -ne $controlledNegativeValidation -and
    -not [bool](Get-PropertyOrDefault -Object $controlledNegativeValidation -Name "passed" -DefaultValue $true) -and
    [int](Get-PropertyOrDefault -Object $controlledNegativeValidation -Name "exitCode" -DefaultValue 0) -ne 0 -and
    [int](Get-PropertyOrDefault -Object $controlledNegativeValidation -Name "mismatchCount" -DefaultValue 0) -gt 0
  $releaseBoundaryHeld = $null -ne $proofBoundary -and
    [bool](Get-PropertyOrDefault -Object $proofBoundary -Name "sourceTreeRealModelRuntime" -DefaultValue $false) -and
    -not [bool](Get-PropertyOrDefault -Object $proofBoundary -Name "publicRedistributionApproved" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $proofBoundary -Name "packageConsumerRuntimeProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $proofBoundary -Name "publicPackageProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $proofBoundary -Name "postPublishProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $proofBoundary -Name "ownerReleaseAcceptance" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $proofBoundary -Name "releaseProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $proofBoundary -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $proofBoundary -Name "uploadsAssets" -DefaultValue $true)
  $realModelEvidenceReady = $null -ne $realModelEvidence -and
    [string](Get-PropertyOrDefault -Object $realModelEvidence -Name "recordKind" -DefaultValue "") -eq "sample-run-evidence-record" -and
    [string](Get-PropertyOrDefault -Object $realModelEvidence -Name "proofClassification" -DefaultValue "") -eq "real-model-runtime" -and
    [string](Get-PropertyOrDefault -Object $realModelEvidence -Name "validatorState" -DefaultValue "") -eq "real-model-runtime" -and
    -not [bool](Get-PropertyOrDefault -Object $realModelEvidence -Name "templateOnly" -DefaultValue $true) -and
    [bool](Get-PropertyOrDefault -Object $realModelEvidence -Name "isSmokePassed" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $realModelEvidence -Name "canPromoteRealModelRuntime" -DefaultValue $false) -and
    -not [bool](Get-PropertyOrDefault -Object $realModelEvidence -Name "canPromotePackageConsumerRuntime" -DefaultValue $false) -and
    $runtimeReferenceValidated -and
    $controlledNegativeValidated -and
    $releaseBoundaryHeld

  $evidenceRecordId = [string](Get-PropertyOrDefault -Object $realModelEvidence -Name "recordId" -DefaultValue "")
  if ([string]::IsNullOrWhiteSpace($evidenceRecordId)) {
    $evidenceRecordId = [string](Get-PropertyOrDefault -Object $realModelEvidence -Name "recordName" -DefaultValue "")
  }

  $dashboardItems += [pscustomobject]@{
      task = $task
      contractRequiredMetadata = $requiredMetadata
      candidateTemplateAligned = [bool]$candidateTemplateAligned
      ownerBackfillAligned = [bool]$ownerBackfillAligned
      ownerProofInputAligned = [bool]$ownerProofInputAligned
      legacyOwnerIntakeMissingFieldCount = [int]$ownerActionRequiredCount
      candidateEvidenceAligned = [bool]$candidateEvidenceAligned
      realModelEvidencePath = $realModelEvidencePath.Replace("\", "/")
      realModelEvidenceRecordId = $evidenceRecordId
      runtimeReferenceValidated = [bool]$runtimeReferenceValidated
      controlledNegativeValidated = [bool]$controlledNegativeValidated
      releaseBoundaryHeld = [bool]$releaseBoundaryHeld
      sourceTreeRealModelEvidenceReady = [bool]$realModelEvidenceReady
      realOwnerEvidenceReady = [bool]$realModelEvidenceReady
      ownerActionRequiredCount = if ($realModelEvidenceReady) { 0 } else { 1 }
      canPromoteRealModelRuntime = [bool]$realModelEvidenceReady
      canPromotePackageConsumerRuntime = $false
      remainingRequirement = if ($realModelEvidenceReady) {
        "Source-tree real-model runtime proof is ready. Clean package-consumer, public-package, post-publish, and owner release evidence remain required."
      }
      else {
        "A complete fail-closed source-tree real-model runtime record is still required for task $task."
      }
      blockingReason = if ($realModelEvidenceReady) {
        "Package-consumer/public/release promotion remains blocked; source-tree real-model runtime proof is ready."
      }
      else {
        "Source-tree real-model runtime proof is incomplete or violates its proof boundary."
      }
    }
}

$taskItems = @($dashboardItems)
$failedAlignmentCount = @($taskItems | Where-Object { -not $_.candidateTemplateAligned -or -not $_.ownerBackfillAligned -or -not $_.ownerProofInputAligned -or -not $_.candidateEvidenceAligned }).Count
$realModelRuntimeMissingTaskCount = @($taskItems | Where-Object { -not $_.sourceTreeRealModelEvidenceReady }).Count
$ownerActionRequiredTaskCount = $realModelRuntimeMissingTaskCount
$canPromoteRealModelRuntime = $realModelRuntimeMissingTaskCount -eq 0
$dashboardState = if ($canPromoteRealModelRuntime) {
  "source-tree-real-model-runtime-ready-package-proof-required"
}
else {
  "blocked-source-tree-real-model-runtime-proof-required"
}

$dashboard = [pscustomobject]@{
  recordKind = "yolovision-six-task-real-proof-chain-dashboard"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  dashboardState = $dashboardState
  taskCount = $taskItems.Count
  failedAlignmentCount = $failedAlignmentCount
  legacyAlignmentReadyTaskCount = @($taskItems | Where-Object { $_.candidateTemplateAligned -and $_.ownerBackfillAligned -and $_.ownerProofInputAligned -and $_.candidateEvidenceAligned }).Count
  legacyAlignmentMissingOrFailedTaskCount = $failedAlignmentCount
  realModelRuntimeReadyTaskCount = @($taskItems | Where-Object { $_.sourceTreeRealModelEvidenceReady }).Count
  realModelRuntimeMissingTaskCount = $realModelRuntimeMissingTaskCount
  ownerActionRequiredTaskCount = $ownerActionRequiredTaskCount
  packageConsumerProofRequiredTaskCount = @($taskItems | Where-Object { -not $_.canPromotePackageConsumerRuntime }).Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRealModelRuntime = [bool]$canPromoteRealModelRuntime
  canPromotePackageConsumerRuntime = $false
  proofBoundary = "This dashboard fail-closed validates committed source-tree real-model runtime records for all six tasks. Optional ignored legacy candidate/Owner artifacts are diagnostic only and cannot block or promote source-tree proof. It does not itself run models, does not prove clean package consumption, does not approve asset redistribution, does not publish, and cannot close release authorization."
  sourceArtifacts = @(
    $contractPath,
    $candidateValidationPath,
    $ownerBackfillValidationPath,
    $ownerProofInputValidationPath,
    $ownerProofImportReportPath,
    $candidateEvidencePath,
    $finalFreezePath
  ) + @($realModelEvidencePaths.Values | ForEach-Object { ([string]$_).Replace("\", "/") } | Sort-Object)
  finalFreezeState = [string](Get-PropertyOrDefault -Object $finalFreeze -Name "freezeState" -DefaultValue "missing")
  ownerProofInputValidationState = [string](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "validationState" -DefaultValue "missing")
  ownerProofImportValidationState = [string](Get-PropertyOrDefault -Object $ownerProofImportReport -Name "validationState" -DefaultValue "missing")
  tasks = $taskItems
}

$jsonPath = Join-Path $OutputRoot "yolovision-six-task-real-proof-chain-dashboard.json"
$markdownPath = Join-Path $OutputRoot "yolovision-six-task-real-proof-chain-dashboard.md"

$dashboard | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $taskItems) {
  "| $(ConvertTo-MarkdownCell $item.task) | $(ConvertTo-MarkdownCell ($item.contractRequiredMetadata -join ', ')) | ``$($item.candidateTemplateAligned)`` | ``$($item.ownerBackfillAligned)`` | ``$($item.ownerProofInputAligned)`` | ``$($item.candidateEvidenceAligned)`` | ``$($item.sourceTreeRealModelEvidenceReady)`` | ``$($item.runtimeReferenceValidated)`` | ``$($item.controlledNegativeValidated)`` | ``$($item.releaseBoundaryHeld)`` | ``$($item.canPromoteRealModelRuntime)`` | $(ConvertTo-MarkdownCell $item.remainingRequirement) |"
}

$markdown = @"
# YoloVision Six Task Real Proof Chain Dashboard

| Field | Value |
| --- | --- |
| dashboardState | ``$($dashboard.dashboardState)`` |
| taskCount | ``$($dashboard.taskCount)`` |
| failedAlignmentCount | ``$($dashboard.failedAlignmentCount)`` |
| legacyAlignmentReadyTaskCount | ``$($dashboard.legacyAlignmentReadyTaskCount)`` |
| legacyAlignmentMissingOrFailedTaskCount | ``$($dashboard.legacyAlignmentMissingOrFailedTaskCount)`` |
| realModelRuntimeReadyTaskCount | ``$($dashboard.realModelRuntimeReadyTaskCount)`` |
| realModelRuntimeMissingTaskCount | ``$($dashboard.realModelRuntimeMissingTaskCount)`` |
| ownerActionRequiredTaskCount | ``$($dashboard.ownerActionRequiredTaskCount)`` |
| packageConsumerProofRequiredTaskCount | ``$($dashboard.packageConsumerProofRequiredTaskCount)`` |
| performsPublish | ``$($dashboard.performsPublish)`` |
| canPublishPublicly | ``$($dashboard.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($dashboard.canCloseReleaseIssue)`` |
| canPromoteRealModelRuntime | ``$($dashboard.canPromoteRealModelRuntime)`` |
| canPromotePackageConsumerRuntime | ``$($dashboard.canPromotePackageConsumerRuntime)`` |
| finalFreezeState | ``$($dashboard.finalFreezeState)`` |

## Proof Boundary

$($dashboard.proofBoundary)

## Task Chain

| Task | Contract Metadata | Candidate Template | Owner Backfill | Owner Proof Input | Candidate Evidence | Real Model Evidence | Raw Reference | Controlled Negative | Boundary Held | Can Promote Runtime | Remaining Requirement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
$($rows -join "`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "YoloVision six-task real proof chain dashboard written:"
Write-Output "  $jsonPath"
Write-Output "  $markdownPath"
Write-Output "TaskCount=$($dashboard.taskCount) FailedAlignmentCount=$($dashboard.failedAlignmentCount) RealModelRuntimeReadyTaskCount=$($dashboard.realModelRuntimeReadyTaskCount) RealModelRuntimeMissingTaskCount=$($dashboard.realModelRuntimeMissingTaskCount)"
