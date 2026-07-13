[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-repair-checklist.json",
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function Test-ArrayContainsText {
  param([string[]]$Values, [string]$Needle)
  return (($Values -join "`n").IndexOf($Needle, [StringComparison]::OrdinalIgnoreCase) -ge 0)
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final owner execution repair checklist not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$repairItems = @((Get-PropertyOrDefault -Object $record -Name "repairItems" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-repair-checklist") -Severity "blocker" -Detail "recordKind must be final-owner-execution-repair-checklist.")) | Out-Null
$items.Add((New-ValidationItem -Id "checklist-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "checklistState" -DefaultValue "") -eq "blocked-final-owner-execution-repair-real-owner-evidence-required") -Severity "blocker" -Detail "Checklist must remain blocked until real Owner evidence is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "eight-repair-items" -Passed ($repairItems.Count -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "executionStepCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "repairItemCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedRepairItemCount" -DefaultValue 0) -eq 8 -and [int](Get-PropertyOrDefault -Object $record -Name "readyRepairItemCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Repair checklist must mirror all eight final owner execution package steps.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPackagePush" -DefaultValue $true))) -Severity "blocker" -Detail "Repair checklist must not publish, promote proof, or close release.")) | Out-Null

foreach ($summaryField in @("requiredFileFieldCount", "requiredSha256FieldCount", "requiredIdentityFieldCount", "requiredNonSubstituteConfirmationCount")) {
  $items.Add((New-ValidationItem -Id "$summaryField-positive" -Passed ([int](Get-PropertyOrDefault -Object $record -Name $summaryField -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "$summaryField must be positive.")) | Out-Null
}

$sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
foreach ($artifact in @(
  "artifacts/final-release/final-owner-execution-package.json",
  "artifacts/final-release/final-owner-execution-package.md",
  "artifacts/final-release/final-owner-execution-package-validation.json",
  "artifacts/final-release/final-owner-execution-package-validation.md",
  "artifacts/final-release/final-owner-proof-action-worklist.json",
  "artifacts/final-release/final-owner-proof-action-worklist.md",
  "artifacts/final-release/final-publish-proof-gate-report.json"
)) {
  $items.Add((New-ValidationItem -Id "source-$($artifact.Replace('/', '-').Replace('.', '-'))" -Passed ($sourceArtifacts -contains $artifact) -Severity "blocker" -Detail "Source artifact $artifact must be listed.")) | Out-Null
}

foreach ($marker in @("local feed", "ProjectReference", "direct .nupkg", "template", "draft", "dry-run", "dashboard", "candidate", "build-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "cannot-use-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Cannot-use marker $marker must be present.")) | Out-Null
}

$allFileFields = @()
$allShaFields = @()
$allIdentityFields = @()
foreach ($item in $repairItems) {
  $id = [string](Get-PropertyOrDefault -Object $item -Name "id" -DefaultValue "")
  $ownerCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "ownerCommands" -DefaultValue @())
  $validatorCommands = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "validatorCommands" -DefaultValue @())
  $expectedArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "expectedResultArtifacts" -DefaultValue @())
  $fileFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "requiredFileFields" -DefaultValue @())
  $shaFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "requiredSha256Fields" -DefaultValue @())
  $identityFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "requiredIdentityFields" -DefaultValue @())
  $nonSubstituteConfirmations = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "requiredNonSubstituteConfirmations" -DefaultValue @())
  $cannotUseMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "cannotUseMarkers" -DefaultValue @())
  $boundary = [string](Get-PropertyOrDefault -Object $item -Name "boundary" -DefaultValue "")

  $allFileFields += $fileFields
  $allShaFields += $shaFields
  $allIdentityFields += $identityFields

  $items.Add((New-ValidationItem -Id "$id-state" -Passed ([string](Get-PropertyOrDefault -Object $item -Name "repairState" -DefaultValue "") -eq "blocked-real-owner-evidence-required") -Severity "blocker" -Detail "$id must remain blocked for real owner evidence.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-owner-command" -Passed ($ownerCommands.Count -ge 1) -Severity "blocker" -Detail "$id must include an owner command.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-validator-command" -Passed ($validatorCommands.Count -ge 1) -Severity "blocker" -Detail "$id must include a validator command.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-expected-artifact" -Passed ($expectedArtifacts.Count -ge 1) -Severity "blocker" -Detail "$id must include expected result artifacts.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-file-fields" -Passed ($fileFields.Count -ge 3) -Severity "blocker" -Detail "$id must include required file fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-sha-fields" -Passed ($shaFields.Count -ge 3) -Severity "blocker" -Detail "$id must include required SHA256 fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-identity-fields" -Passed ($identityFields.Count -ge 3) -Severity "blocker" -Detail "$id must include required identity fields.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-non-substitute-confirmations" -Passed ($nonSubstituteConfirmations.Count -ge 8) -Severity "blocker" -Detail "$id must include non-substitute confirmations.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-cannot-use-markers" -Passed ((Test-ArrayContainsText -Values $cannotUseMarkers -Needle "local feed") -and (Test-ArrayContainsText -Values $cannotUseMarkers -Needle "ProjectReference") -and (Test-ArrayContainsText -Values $cannotUseMarkers -Needle "direct .nupkg") -and (Test-ArrayContainsText -Values $cannotUseMarkers -Needle "blocked-by-cuda-driver")) -Severity "blocker" -Detail "$id must preserve cannot-use markers.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $item -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $item -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "$id must remain non-proof guidance.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-boundary" -Passed ($boundary.Contains("Repair guidance only") -and $boundary.Contains("not runtime proof") -and $boundary.Contains("not package push")) -Severity "blocker" -Detail "$id must state repair-only non-proof boundary.")) | Out-Null
}

foreach ($field in @("stdoutPath", "stderrPath", "mergedTranscriptPath")) {
  $items.Add((New-ValidationItem -Id "file-field-$field-present" -Passed (Test-ArrayContainsText -Values $allFileFields -Needle $field) -Severity "blocker" -Detail "Required file field $field must be present.")) | Out-Null
}

foreach ($field in @("stdoutSha256", "stderrSha256", "mergedTranscriptSha256")) {
  $items.Add((New-ValidationItem -Id "sha-field-$field-present" -Passed (Test-ArrayContainsText -Values $allShaFields -Needle $field) -Severity "blocker" -Detail "Required SHA256 field $field must be present.")) | Out-Null
}

foreach ($field in @("exitCode", "hostIdentity", "packageIdentity", "ownerReviewer")) {
  $items.Add((New-ValidationItem -Id "identity-field-$field-present" -Passed (Test-ArrayContainsText -Values $allIdentityFields -Needle $field) -Severity "blocker" -Detail "Required identity field $field must be present.")) | Out-Null
}

$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$failedActionRequired = @($repairItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "repairState" -DefaultValue "") -eq "blocked-real-owner-evidence-required" }).Count
$validationState = if ($failedBlockers -eq 0) { "blocked-final-owner-execution-repair-real-owner-evidence-required" } else { "invalid-final-owner-execution-repair-checklist" }
$validationItems = @($items.ToArray())
$failedValidationItems = @($failedItems)

$validation = [ordered]@{
  recordKind = "final-owner-execution-repair-checklist-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  checklistState = [string](Get-PropertyOrDefault -Object $record -Name "checklistState" -DefaultValue "")
  executionStepCount = [int](Get-PropertyOrDefault -Object $record -Name "executionStepCount" -DefaultValue 0)
  repairItemCount = $repairItems.Count
  blockedRepairItemCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedRepairItemCount" -DefaultValue 0)
  readyRepairItemCount = [int](Get-PropertyOrDefault -Object $record -Name "readyRepairItemCount" -DefaultValue 0)
  requiredFileFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredFileFieldCount" -DefaultValue 0)
  requiredSha256FieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredSha256FieldCount" -DefaultValue 0)
  requiredIdentityFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredIdentityFieldCount" -DefaultValue 0)
  requiredNonSubstituteConfirmationCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredNonSubstituteConfirmationCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers
  failedActionRequiredCount = $failedActionRequired
  findingCount = $failedItems.Count
  performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $false)
  canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $false)
  canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $false)
  canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $false)
  isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $false)
  isPostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $false)
  isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $false)
  validationItems = $validationItems
  failedItems = $failedValidationItems
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-repair-checklist-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-repair-checklist-validation.md"
$validation | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Final Owner Execution Repair Checklist Validation",
  "",
  "- validationState: ``$validationState``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$failedActionRequired``",
  "- repairItemCount: ``$($repairItems.Count)``",
  "- requiredFileFieldCount: ``$($validation.requiredFileFieldCount)``",
  "- requiredSha256FieldCount: ``$($validation.requiredSha256FieldCount)``",
  "- requiredIdentityFieldCount: ``$($validation.requiredIdentityFieldCount)``"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final owner execution repair checklist validation failed with $failedBlockers blocker(s)."
}
