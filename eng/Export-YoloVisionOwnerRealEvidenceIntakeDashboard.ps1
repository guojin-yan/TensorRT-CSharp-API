[CmdletBinding()]
param(
  [string]$SixTaskDashboardPath = "artifacts/yolovision/yolovision-six-task-real-proof-chain-dashboard.json",
  [string]$ValidationPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json",
  [string]$RepairPackPath = "artifacts/user-acceptance/yolovision-owner-proof-field-delta-repair-pack.json",
  [string]$ExecutionPackPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-execution-pack.json",
  [string]$CandidateEvidencePath = "artifacts/user-acceptance/yolovision-real-asset-owner-sample-run-evidence.candidate.json",
  [string]$FinalGatePath = "artifacts/final-release/final-publish-proof-gate-report.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\user-acceptance"
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
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
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

function Get-CaseId {
  param([string]$Task)
  return "yolov8n-$Task"
}

function Get-FieldsForCaseAndGroup {
  param([object[]]$Deltas, [string]$CaseId, [string[]]$Categories, [string[]]$PathHints)

  $items = @($Deltas | Where-Object {
      [string]$_.caseId -eq $CaseId -and (
        $Categories -contains [string]$_.category -or
        @($PathHints | Where-Object { ([string]$_.Length -gt 0) -and ([string]$_.jsonPath -match $_) }).Count -gt 0
      )
    })

  return @($items | ForEach-Object { [string]$_.jsonPath } | Sort-Object -Unique)
}

function Get-GlobalFieldsForGroup {
  param([object[]]$Deltas, [string[]]$Categories, [string[]]$PathHints)

  $items = @($Deltas | Where-Object {
      [string]$_.caseId -eq "global" -and (
        $Categories -contains [string]$_.category -or
        @($PathHints | Where-Object { ([string]$_.Length -gt 0) -and ([string]$_.jsonPath -match $_) }).Count -gt 0
      )
    })

  return @($items | ForEach-Object { [string]$_.jsonPath } | Sort-Object -Unique)
}

$sixTaskDashboard = Read-JsonOrNull $SixTaskDashboardPath
$validation = Read-JsonOrNull $ValidationPath
$repairPack = Read-JsonOrNull $RepairPackPath
$executionPack = Read-JsonOrNull $ExecutionPackPath
$candidateEvidence = Read-JsonOrNull $CandidateEvidencePath
$finalGate = Read-JsonOrNull $FinalGatePath

if ($null -eq $sixTaskDashboard) { throw "Six-task dashboard missing: $SixTaskDashboardPath" }
if ($null -eq $validation) { throw "YoloVision owner proof validation missing: $ValidationPath" }
if ($null -eq $repairPack) { throw "YoloVision owner repair pack missing: $RepairPackPath" }
if ($null -eq $executionPack) { throw "YoloVision owner execution pack missing: $ExecutionPackPath" }

$fieldDeltas = @($repairPack.fieldDeltas)
$caseSummaries = @($repairPack.caseSummaries)
$executionCases = @($executionPack.cases)
$candidateCases = if ($null -ne $candidateEvidence -and $candidateEvidence.PSObject.Properties.Name -contains "cases") { @($candidateEvidence.cases) } else { @() }
$finalGateActionItems = if ($null -ne $finalGate -and $finalGate.PSObject.Properties.Name -contains "validationItems") {
  @($finalGate.validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "action-required" })
}
else {
  @()
}

$taskRows = @()
foreach ($taskItem in @($sixTaskDashboard.tasks)) {
  $task = [string]$taskItem.task
  $caseId = Get-CaseId -Task $task
  $caseSummary = @($caseSummaries | Where-Object { [string]$_.caseId -eq $caseId } | Select-Object -First 1)
  $executionCase = @($executionCases | Where-Object { [string]$_.caseId -eq $caseId } | Select-Object -First 1)
  $candidateCase = @($candidateCases | Where-Object { [string]$_.caseId -eq $caseId } | Select-Object -First 1)

  $taskRows += [pscustomobject]@{
    task = $task
    caseId = $caseId
    missingModelFields = Get-FieldsForCaseAndGroup -Deltas $fieldDeltas -CaseId $caseId -Categories @("owner-input") -PathHints @("model\.")
    missingLabelsInputFields = Get-FieldsForCaseAndGroup -Deltas $fieldDeltas -CaseId $caseId -Categories @("owner-input") -PathHints @("labels\.", "input\.")
    missingTensorRtExecFields = Get-FieldsForCaseAndGroup -Deltas $fieldDeltas -CaseId $caseId -Categories @("hash") -PathHints @("tensorRtExec\.", "engine")
    missingYoloVisionFields = Get-FieldsForCaseAndGroup -Deltas $fieldDeltas -CaseId $caseId -Categories @("expected-evidence-line") -PathHints @("yoloVision\.", "outputJson", "stdoutSummary", "stderrSummary")
    missingSha256Fields = Get-FieldsForCaseAndGroup -Deltas $fieldDeltas -CaseId $caseId -Categories @("hash") -PathHints @("sha256", "Sha256", "SHA256")
    missingHostMetadata = Get-GlobalFieldsForGroup -Deltas $fieldDeltas -Categories @("global-host-metadata") -PathHints @("host")
    missingPackageMetadata = Get-GlobalFieldsForGroup -Deltas $fieldDeltas -Categories @("global-host-metadata") -PathHints @("package")
    missingOwnerReview = @(
      (Get-FieldsForCaseAndGroup -Deltas $fieldDeltas -CaseId $caseId -Categories @("owner-review") -PathHints @("ownerReview")),
      (Get-GlobalFieldsForGroup -Deltas $fieldDeltas -Categories @("owner-review") -PathHints @("ownerReview", "owner"))
    ) | ForEach-Object { $_ } | Sort-Object -Unique
    missingFieldCount = if ($caseSummary.Count -gt 0) { [int]$caseSummary[0].missingFieldCount } else { 0 }
    recommendedOwnerCommands = if ($executionCase.Count -gt 0) { @($executionCase[0].recommendedCommandSequence | ForEach-Object { [string]$_.command }) } else { @() }
    expectedEvidenceLines = if ($executionCase.Count -gt 0) { @($executionCase[0].expectedEvidenceLines | ForEach-Object { [string]$_ }) } else { @() }
    candidateEvidencePresent = $candidateCase.Count -gt 0
    strictValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
    blockingReason = "Owner real evidence is still missing for $caseId. Real logs, output JSON, SHA256 values, host/package metadata, and owner review must pass strict validation before any runtime proof promotion."
  }
}

$totalMissingFieldCount = [int](Get-PropertyOrDefault -Object $repairPack -Name "missingFieldCount" -DefaultValue 0)
$globalMissingFieldCount = [int](Get-PropertyOrDefault -Object $repairPack -Name "globalMissingFieldCount" -DefaultValue 0)

$dashboard = [pscustomobject]@{
  recordKind = "yolovision-owner-real-evidence-intake-dashboard"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  dashboardState = "blocked-owner-real-evidence-required"
  taskCount = @($taskRows).Count
  totalMissingFieldCount = $totalMissingFieldCount
  globalMissingFieldCount = $globalMissingFieldCount
  failedBlockerCount = [int](Get-PropertyOrDefault -Object $validation -Name "failedBlockerCount" -DefaultValue 0)
  ownerActionRequiredTaskCount = @($taskRows | Where-Object { [int]$_.missingFieldCount -gt 0 }).Count
  finalGateActionRequiredCount = @($finalGateActionItems).Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  sourceArtifacts = @(
    $SixTaskDashboardPath,
    $ValidationPath,
    $RepairPackPath,
    $ExecutionPackPath,
    $CandidateEvidencePath,
    $FinalGatePath
  )
  forbiddenSubstitutes = @($executionPack.forbiddenSubstitutes)
  finalGateActionRequired = @($finalGateActionItems | ForEach-Object {
      [pscustomobject]@{
        id = [string]$_.id
        detail = [string]$_.detail
      }
    })
  tasks = @($taskRows)
  boundary = "This intake dashboard is owner evidence receiving guidance only. It cannot publish, close release issues, or promote real-model-runtime/package-consumer-runtime proof without real files, logs, hashes, host/package metadata, owner review, and strict validators."
}

$jsonPath = Join-Path $OutputRoot "yolovision-owner-real-evidence-intake-dashboard.json"
$markdownPath = Join-Path $OutputRoot "yolovision-owner-real-evidence-intake-dashboard.md"
$dashboard | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($task in $taskRows) {
  "| ``$(ConvertTo-MarkdownCell $task.caseId)`` | ``$(ConvertTo-MarkdownCell $task.task)`` | $($task.missingFieldCount) | $(@($task.missingSha256Fields).Count) | $(@($task.missingHostMetadata).Count) | $(@($task.missingPackageMetadata).Count) | $(@($task.missingOwnerReview).Count) | ``False`` |"
}

$actionLines = $dashboard.finalGateActionRequired | ForEach-Object { "- ``$($_.id)``: $(ConvertTo-MarkdownCell $_.detail)" }

$markdown = @"
# YoloVision Owner Real Evidence Intake Dashboard

Generated at: ``$($dashboard.generatedAtUtc)``

## Summary

- dashboardState: ``$($dashboard.dashboardState)``
- taskCount: ``$($dashboard.taskCount)``
- totalMissingFieldCount: ``$($dashboard.totalMissingFieldCount)``
- globalMissingFieldCount: ``$($dashboard.globalMissingFieldCount)``
- failedBlockerCount: ``$($dashboard.failedBlockerCount)``
- ownerActionRequiredTaskCount: ``$($dashboard.ownerActionRequiredTaskCount)``
- finalGateActionRequiredCount: ``$($dashboard.finalGateActionRequiredCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRealModelRuntime: ``False``
- canPromotePackageConsumerRuntime: ``False``

## Task Intake

| Case | Task | Missing Fields | SHA256 Missing | Host Metadata | Package Metadata | Owner Review | Can Promote Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
$($rows -join "`r`n")

## Final Gate Action Required

$($actionLines -join "`r`n")

## Boundary

$($dashboard.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "YoloVision owner real evidence intake dashboard written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "DashboardState=$($dashboard.dashboardState) TaskCount=$($dashboard.taskCount) TotalMissingFieldCount=$($dashboard.totalMissingFieldCount) FinalGateActionRequiredCount=$($dashboard.finalGateActionRequiredCount)"
