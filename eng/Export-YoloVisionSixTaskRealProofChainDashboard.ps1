[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
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

$contractPath = "samples/YoloVision/yolovision-task-output-contract.json"
$candidateValidationPath = "artifacts/yolovision/yolovision-real-asset-candidate-validation.json"
$ownerBackfillValidationPath = "artifacts/yolovision/yolovision-real-asset-owner-backfill-pack-validation.json"
$ownerProofInputValidationPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json"
$ownerProofImportReportPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-import-report.json"
$candidateEvidencePath = "artifacts/user-acceptance/yolovision-real-asset-owner-sample-run-evidence.candidate.json"
$finalFreezePath = "artifacts/final-release/release-candidate-final-evidence-freeze.json"

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

  $dashboardItems += [pscustomobject]@{
      task = $task
      contractRequiredMetadata = $requiredMetadata
      candidateTemplateAligned = [bool]$candidateTemplateAligned
      ownerBackfillAligned = [bool]$ownerBackfillAligned
      ownerProofInputAligned = [bool]$ownerProofInputAligned
      ownerActionRequiredCount = [int]$ownerActionRequiredCount
      candidateEvidenceAligned = [bool]$candidateEvidenceAligned
      realOwnerEvidenceReady = $false
      canPromoteRealModelRuntime = $false
      canPromotePackageConsumerRuntime = $false
      blockingReason = "Owner must provide real logs, output JSON, hashes, host metadata, package metadata, and review for yolov8n-$task. Template/candidate/report evidence remains non-proof."
    }
}

$taskItems = @($dashboardItems)
$failedAlignmentCount = @($taskItems | Where-Object { -not $_.candidateTemplateAligned -or -not $_.ownerBackfillAligned -or -not $_.ownerProofInputAligned -or -not $_.candidateEvidenceAligned }).Count
$ownerActionRequiredTaskCount = @($taskItems | Where-Object { -not $_.realOwnerEvidenceReady }).Count

$dashboard = [pscustomobject]@{
  recordKind = "yolovision-six-task-real-proof-chain-dashboard"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  dashboardState = "blocked-real-owner-proof-required"
  taskCount = $taskItems.Count
  failedAlignmentCount = $failedAlignmentCount
  ownerActionRequiredTaskCount = $ownerActionRequiredTaskCount
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  proofBoundary = "This dashboard proves six-task template/contract alignment only. It does not run models, does not run package consumers, does not publish, and cannot replace owner-filled real logs, hashes, host metadata, package metadata, or review."
  sourceArtifacts = @(
    $contractPath,
    $candidateValidationPath,
    $ownerBackfillValidationPath,
    $ownerProofInputValidationPath,
    $ownerProofImportReportPath,
    $candidateEvidencePath,
    $finalFreezePath
  )
  finalFreezeState = [string](Get-PropertyOrDefault -Object $finalFreeze -Name "freezeState" -DefaultValue "missing")
  ownerProofInputValidationState = [string](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "validationState" -DefaultValue "missing")
  ownerProofImportValidationState = [string](Get-PropertyOrDefault -Object $ownerProofImportReport -Name "validationState" -DefaultValue "missing")
  tasks = $taskItems
}

$jsonPath = Join-Path $OutputRoot "yolovision-six-task-real-proof-chain-dashboard.json"
$markdownPath = Join-Path $OutputRoot "yolovision-six-task-real-proof-chain-dashboard.md"

$dashboard | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $taskItems) {
  "| $(ConvertTo-MarkdownCell $item.task) | $(ConvertTo-MarkdownCell ($item.contractRequiredMetadata -join ', ')) | ``$($item.candidateTemplateAligned)`` | ``$($item.ownerBackfillAligned)`` | ``$($item.ownerProofInputAligned)`` | ``$($item.candidateEvidenceAligned)`` | ``$($item.realOwnerEvidenceReady)`` | ``$($item.canPromoteRealModelRuntime)`` | $(ConvertTo-MarkdownCell $item.blockingReason) |"
}

$markdown = @"
# YoloVision Six Task Real Proof Chain Dashboard

| Field | Value |
| --- | --- |
| dashboardState | ``$($dashboard.dashboardState)`` |
| taskCount | ``$($dashboard.taskCount)`` |
| failedAlignmentCount | ``$($dashboard.failedAlignmentCount)`` |
| ownerActionRequiredTaskCount | ``$($dashboard.ownerActionRequiredTaskCount)`` |
| performsPublish | ``$($dashboard.performsPublish)`` |
| canPublishPublicly | ``$($dashboard.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($dashboard.canCloseReleaseIssue)`` |
| canPromoteRealModelRuntime | ``$($dashboard.canPromoteRealModelRuntime)`` |
| canPromotePackageConsumerRuntime | ``$($dashboard.canPromotePackageConsumerRuntime)`` |
| finalFreezeState | ``$($dashboard.finalFreezeState)`` |

## Proof Boundary

$($dashboard.proofBoundary)

## Task Chain

| Task | Contract Metadata | Candidate Template | Owner Backfill | Owner Proof Input | Candidate Evidence | Real Owner Evidence Ready | Can Promote Runtime | Blocking Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
$($rows -join "`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "YoloVision six-task real proof chain dashboard written:"
Write-Output "  $jsonPath"
Write-Output "  $markdownPath"
Write-Output "TaskCount=$($dashboard.taskCount) FailedAlignmentCount=$($dashboard.failedAlignmentCount) OwnerActionRequiredTaskCount=$($dashboard.ownerActionRequiredTaskCount)"
