[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-proof-report-pack.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-ArrayHasItems {
  param([AllowNull()][object]$Value)

  return @($Value).Count -gt 0
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner real proof report pack not found: $resolvedInputPath"
}

$pack = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $pack -Name "recordKind" -DefaultValue "")
$reportItems = @(Get-PropertyOrDefault -Object $pack -Name "proofReportItems" -DefaultValue @())
$requiredProofLanes = @(Get-PropertyOrDefault -Object $pack -Name "requiredProofLanes" -DefaultValue @())
$forbiddenSubstitutes = @(Get-PropertyOrDefault -Object $pack -Name "forbiddenSubstitutes" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-real-proof-report-pack") -Severity "blocker" -Detail "recordKind must be owner-real-proof-report-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "report-item-count" -Passed ($reportItems.Count -eq 6) -Severity "blocker" -Detail "Report pack must include exactly 6 proof report items.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-pack-state" -Passed ([string](Get-PropertyOrDefault -Object $pack -Name "packState" -DefaultValue "") -eq "blocked-owner-real-proof-report-input-required") -Severity "blocker" -Detail "Report pack must remain blocked until owner fills real proof reports.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $pack -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Report pack must not promote runtime proof, release close proof, or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $pack -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Report pack must not publish or approve public publication.")) | Out-Null
$items.Add((New-ValidationItem -Id "not-ready-for-review" -Passed ([int](Get-PropertyOrDefault -Object $pack -Name "readyForOwnerReviewCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Default report pack must not claim readyForOwnerReview.")) | Out-Null
$items.Add((New-ValidationItem -Id "not-ready-for-promotion" -Passed ([int](Get-PropertyOrDefault -Object $pack -Name "readyForPromotionCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Default report pack must not claim readyForPromotion.")) | Out-Null

foreach ($lane in @("package-consumer-runtime", "post-publish-verification", "linux-runner-proof", "real-model-runtime", "release-close-owner-input", "strict-close-validation")) {
  $items.Add((New-ValidationItem -Id "lane-$lane" -Passed ($requiredProofLanes -contains $lane -and @($reportItems | Where-Object { [string]$_.proofLane -eq $lane }).Count -eq 1) -Severity "blocker" -Detail "Required proof lane must be present exactly once: $lane.")) | Out-Null
}

foreach ($required in @("local feed", "ProjectReference", "direct .nupkg", "DependencyProbe", "build-only", "template", "Windows handoff for Linux proof", "hash-only audit")) {
  $items.Add((New-ValidationItem -Id "forbidden-$($required.Replace(' ', '-').Replace('.', '').ToLowerInvariant())" -Passed ($forbiddenSubstitutes -contains $required) -Severity "blocker" -Detail "Forbidden substitute must be listed: $required.")) | Out-Null
}

foreach ($reportItem in $reportItems) {
  $reportItemId = [string](Get-PropertyOrDefault -Object $reportItem -Name "reportItemId" -DefaultValue "unknown-report-item")
  $ownerInputs = @(Get-PropertyOrDefault -Object $reportItem -Name "requiredOwnerInputs" -DefaultValue @())
  $evidenceFiles = @(Get-PropertyOrDefault -Object $reportItem -Name "requiredEvidenceFiles" -DefaultValue @())
  $hashes = @(Get-PropertyOrDefault -Object $reportItem -Name "requiredHashes" -DefaultValue @())
  $commands = @(Get-PropertyOrDefault -Object $reportItem -Name "requiredCommands" -DefaultValue @())
  $validators = @(Get-PropertyOrDefault -Object $reportItem -Name "requiredValidators" -DefaultValue @())
  $forbiddenChecklist = @(Get-PropertyOrDefault -Object $reportItem -Name "forbiddenSubstituteChecklist" -DefaultValue @())
  $reviewChecklist = @(Get-PropertyOrDefault -Object $reportItem -Name "reviewChecklist" -DefaultValue @())
  $promotionFlags = Get-PropertyOrDefault -Object $reportItem -Name "promotionFlags" -DefaultValue $null

  $items.Add((New-ValidationItem -Id "$reportItemId-source-record" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $reportItem -Name "sourceRecordId" -DefaultValue ""))) -Severity "blocker" -Detail "Each report item must have a source record id.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-proof-lane" -Passed ($requiredProofLanes -contains [string](Get-PropertyOrDefault -Object $reportItem -Name "proofLane" -DefaultValue "")) -Severity "blocker" -Detail "Each report item must map to a required proof lane.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-owner-inputs-shape" -Passed (Test-ArrayHasItems -Value $ownerInputs) -Severity "blocker" -Detail "Each report item must include required owner inputs.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-evidence-files-shape" -Passed (Test-ArrayHasItems -Value $evidenceFiles) -Severity "blocker" -Detail "Each report item must include required evidence files.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-hashes-shape" -Passed (Test-ArrayHasItems -Value $hashes) -Severity "blocker" -Detail "Each report item must include required hashes.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-commands-shape" -Passed (Test-ArrayHasItems -Value $commands) -Severity "blocker" -Detail "Each report item must include required commands.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-validators-shape" -Passed (Test-ArrayHasItems -Value $validators) -Severity "blocker" -Detail "Each report item must include required validators.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-forbidden-checklist-shape" -Passed (Test-ArrayHasItems -Value $forbiddenChecklist) -Severity "blocker" -Detail "Each report item must include forbidden substitute checklist.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-review-checklist-shape" -Passed (Test-ArrayHasItems -Value $reviewChecklist) -Severity "blocker" -Detail "Each report item must include review checklist.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-promotion-flags-shape" -Passed ($null -ne $promotionFlags) -Severity "blocker" -Detail "Each report item must include promotionFlags.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-not-ready-for-owner-review" -Passed (-not [bool](Get-PropertyOrDefault -Object $reportItem -Name "readyForOwnerReview" -DefaultValue $true)) -Severity "blocker" -Detail "Default report item must not claim readyForOwnerReview.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$reportItemId-not-ready-for-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $reportItem -Name "readyForPromotion" -DefaultValue $true)) -Severity "blocker" -Detail "Default report item must not claim readyForPromotion.")) | Out-Null

  foreach ($ownerInput in $ownerInputs) {
    $name = [string](Get-PropertyOrDefault -Object $ownerInput -Name "name" -DefaultValue "owner-input")
    $items.Add((New-ValidationItem -Id "$reportItemId-owner-input-$name-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $ownerInput -Name "currentValue" -DefaultValue "")) -and [bool](Get-PropertyOrDefault -Object $ownerInput -Name "ready" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must fill and mark ready input: $name.")) | Out-Null
  }

  foreach ($evidenceFile in $evidenceFiles) {
    $name = [string](Get-PropertyOrDefault -Object $evidenceFile -Name "name" -DefaultValue "evidence-file")
    $items.Add((New-ValidationItem -Id "$reportItemId-evidence-$name-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $evidenceFile -Name "path" -DefaultValue "")) -and -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $evidenceFile -Name "sha256" -DefaultValue "")) -and [bool](Get-PropertyOrDefault -Object $evidenceFile -Name "exists" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $evidenceFile -Name "hashReady" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must provide evidence file path and SHA256: $name.")) | Out-Null
  }

  foreach ($hash in $hashes) {
    $name = [string](Get-PropertyOrDefault -Object $hash -Name "name" -DefaultValue "hash")
    $items.Add((New-ValidationItem -Id "$reportItemId-hash-$name-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hash -Name "path" -DefaultValue "")) -and -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $hash -Name "sha256" -DefaultValue "")) -and [bool](Get-PropertyOrDefault -Object $hash -Name "matches" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must provide matching hash evidence: $name.")) | Out-Null
  }

  foreach ($command in $commands) {
    $items.Add((New-ValidationItem -Id "$reportItemId-command-captured-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $command -Name "logPath" -DefaultValue "")) -and -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $command -Name "logSha256" -DefaultValue "")) -and [bool](Get-PropertyOrDefault -Object $command -Name "captured" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must capture command execution and log.")) | Out-Null
  }

  foreach ($validator in $validators) {
    $items.Add((New-ValidationItem -Id "$reportItemId-validator-pass-required" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $validator -Name "command" -DefaultValue "")) -and [bool](Get-PropertyOrDefault -Object $validator -Name "passed" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must provide passing validator output.")) | Out-Null
  }

  foreach ($forbidden in $forbiddenChecklist) {
    $name = [string](Get-PropertyOrDefault -Object $forbidden -Name "name" -DefaultValue "forbidden-substitute")
    $items.Add((New-ValidationItem -Id "$reportItemId-forbidden-$($name.Replace(' ', '-').Replace('.', '').ToLowerInvariant())-absent-required" -Passed ([bool](Get-PropertyOrDefault -Object $forbidden -Name "checked" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $forbidden -Name "present" -DefaultValue $true) -and [bool](Get-PropertyOrDefault -Object $forbidden -Name "passed" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must check forbidden substitute absent: $name.")) | Out-Null
  }

  foreach ($review in $reviewChecklist) {
    $id = [string](Get-PropertyOrDefault -Object $review -Name "id" -DefaultValue "review")
    $items.Add((New-ValidationItem -Id "$reportItemId-review-$id-required" -Passed ([bool](Get-PropertyOrDefault -Object $review -Name "passed" -DefaultValue $false)) -Severity "action-required" -Detail "Owner review checklist item must pass: $id.")) | Out-Null
  }

  $items.Add((New-ValidationItem -Id "$reportItemId-no-proof-flags" -Passed (-not [bool](Get-PropertyOrDefault -Object $reportItem -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $reportItem -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $reportItem -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $reportItem -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Report item proof flags must remain false.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-owner-real-proof-report-pack"
}
else {
  "blocked-owner-real-proof-report-input-required"
}

$validation = [pscustomobject]@{
  recordKind = "owner-real-proof-report-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  reportItemCount = $reportItems.Count
  blockedReportItemCount = [int](Get-PropertyOrDefault -Object $pack -Name "blockedReportItemCount" -DefaultValue $reportItems.Count)
  readyForOwnerReviewCount = [int](Get-PropertyOrDefault -Object $pack -Name "readyForOwnerReviewCount" -DefaultValue 0)
  readyForPromotionCount = [int](Get-PropertyOrDefault -Object $pack -Name "readyForPromotionCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates an owner report pack only. It is not runtime proof, not publication approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-real-proof-report-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-real-proof-report-pack-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Owner Real Proof Report Pack Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| reportItemCount | ``$($validation.reportItemCount)`` |
| blockedReportItemCount | ``$($validation.blockedReportItemCount)`` |
| readyForOwnerReviewCount | ``$($validation.readyForOwnerReviewCount)`` |
| readyForPromotionCount | ``$($validation.readyForPromotionCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPromoteRuntimeProof | ``False`` |
| canCloseReleaseIssue | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isReleaseCloseProof | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join [Environment]::NewLine)

## Boundary

$($validation.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner real proof report pack validation written to $jsonPath"
Write-Host "Owner real proof report pack validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState ReportItems=$($validation.reportItemCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Owner real proof report pack validation failed with $($failedBlockers.Count) blocker(s)."
}
