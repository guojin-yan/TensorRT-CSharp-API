[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-external-execution-result-backfill-kit.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner external execution result backfill kit not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$kitState = [string](Get-PropertyOrDefault -Object $record -Name "kitState" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$backfillLines = @((Get-PropertyOrDefault -Object $record -Name "backfillLines" -DefaultValue @()))
$placeholderFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "placeholderFieldCount" -DefaultValue 0)
$missingExternalLogCount = [int](Get-PropertyOrDefault -Object $record -Name "missingExternalLogCount" -DefaultValue 0)
$missingSha256Count = [int](Get-PropertyOrDefault -Object $record -Name "missingSha256Count" -DefaultValue 0)
$strictValidators = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "strictValidators" -DefaultValue @())
$sourceArtifacts = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
$forbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())
$requiredResultArtifactPaths = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredResultArtifactPaths" -DefaultValue @())
$backfillJsonFieldPaths = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "backfillJsonFieldPaths" -DefaultValue @())
$ownerResultInputSchemaSummary = Get-PropertyOrDefault -Object $record -Name "ownerResultInputSchemaSummary" -DefaultValue $null
$ownerResultRequiredPaths = Convert-ToStringArray (Get-PropertyOrDefault -Object $ownerResultInputSchemaSummary -Name "requiredPaths" -DefaultValue @())
$ownerResultHashProofPaths = Convert-ToStringArray (Get-PropertyOrDefault -Object $ownerResultInputSchemaSummary -Name "hashProofPaths" -DefaultValue @())
$ownerResultReadyConditions = Convert-ToStringArray (Get-PropertyOrDefault -Object $ownerResultInputSchemaSummary -Name "requiredReadyConditions" -DefaultValue @())
$cleanExternalConsumerContract = Get-PropertyOrDefault -Object $record -Name "cleanExternalConsumerContract" -DefaultValue $null
$proofPromotionBoundary = Get-PropertyOrDefault -Object $record -Name "proofPromotionBoundary" -DefaultValue $null

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-external-execution-result-backfill-kit") -Severity "blocker" -Detail "recordKind must be owner-external-execution-result-backfill-kit.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-kit-state" -Passed ($kitState -eq "blocked-owner-external-execution-results-required") -Severity "blocker" -Detail "Kit must remain blocked until real owner external execution results are supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Kit must not publish, approve publication, or close the release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "backfill-lines-present" -Passed ($backfillLines.Count -ge 4) -Severity "blocker" -Detail "Kit must include package-consumer, post-publish, release-close, and final decision lines.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validators-present" -Passed (($strictValidators -join "`n").Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and ($strictValidators -join "`n").Contains("Test-PostPublishVerificationRecord.ps1", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Kit must preserve strict release close and post-publish validators.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts-present" -Passed ($sourceArtifacts.Count -ge 10 -and ($sourceArtifacts -contains "artifacts/final-release/owner-external-proof-execution-result.input.template.json") -and ($sourceArtifacts -contains "artifacts/final-release/owner-external-proof-execution-result-input-template-validation.json")) -Severity "blocker" -Detail "Kit must preserve source artifacts, including owner external proof execution result input template and validation.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes-present" -Passed (($forbiddenSubstitutes -join "`n").Contains("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -and ($forbiddenSubstitutes -join "`n").Contains("direct .nupkg", [StringComparison]::OrdinalIgnoreCase) -and ($forbiddenSubstitutes -join "`n").Contains("blocked-by-cuda-driver", [StringComparison]::OrdinalIgnoreCase) -and ($forbiddenSubstitutes -join "`n").Contains("dashboard", [StringComparison]::OrdinalIgnoreCase) -and ($forbiddenSubstitutes -join "`n").Contains("dry-run", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Kit must reject common proof substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-command-and-artifacts" -Passed (@($backfillLines | Where-Object { [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "ownerCommand" -DefaultValue "")) -or -not [bool](Get-PropertyOrDefault -Object $_ -Name "stdoutPathRequired" -DefaultValue $false) -or -not [bool](Get-PropertyOrDefault -Object $_ -Name "stderrPathRequired" -DefaultValue $false) -or -not [bool](Get-PropertyOrDefault -Object $_ -Name "mergedTranscriptPathRequired" -DefaultValue $false) -or -not [bool](Get-PropertyOrDefault -Object $_ -Name "hashProofRequired" -DefaultValue $false) -or [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteDirectlyToProof" -DefaultValue $true) }).Count -eq 0) -Severity "blocker" -Detail "Every lane must include owner command text and stdout/stderr/transcript/hash proof requirements, and cannot promote directly to proof.")) | Out-Null
$jsonPathText = $backfillJsonFieldPaths -join "`n"
$schemaPathText = $ownerResultRequiredPaths -join "`n"
$hashPathText = $ownerResultHashProofPaths -join "`n"
$readyConditionText = $ownerResultReadyConditions -join "`n"
$legacyObjectPaths = @($backfillJsonFieldPaths | Where-Object { $_ -match '^\$\.(packageConsumerRuntimeProof|postPublishVerification|releaseIssueCloseRecordOwnerInput|releaseIssueFinalCloseDecision)\.' })
$items.Add((New-ValidationItem -Id "artifact-and-json-field-paths" -Passed ($requiredResultArtifactPaths.Count -ge 8 -and $backfillJsonFieldPaths.Count -ge 20 -and $legacyObjectPaths.Count -eq 0 -and $jsonPathText.Contains('$.resultInputs[].packageIdentity.nupkgPath', [StringComparison]::OrdinalIgnoreCase) -and $jsonPathText.Contains('$.resultInputs[].validatorOutputPath', [StringComparison]::OrdinalIgnoreCase) -and $jsonPathText.Contains('$.resultInputs[].nonSubstituteConfirmations', [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Kit must expose import-aligned resultInputs[] JSON backfill paths and must not keep legacy lane-object JSON paths.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-result-input-schema-summary" -Passed ($null -ne $ownerResultInputSchemaSummary -and [string](Get-PropertyOrDefault -Object $ownerResultInputSchemaSummary -Name "inputPath" -DefaultValue "") -eq "artifacts/final-release/owner-external-proof-execution-result.input.json" -and [string](Get-PropertyOrDefault -Object $ownerResultInputSchemaSummary -Name "templatePath" -DefaultValue "") -eq "artifacts/final-release/owner-external-proof-execution-result.input.template.json" -and [string](Get-PropertyOrDefault -Object $ownerResultInputSchemaSummary -Name "templateValidationPath" -DefaultValue "") -eq "artifacts/final-release/owner-external-proof-execution-result-input-template-validation.json" -and [string](Get-PropertyOrDefault -Object $ownerResultInputSchemaSummary -Name "rootArrayPath" -DefaultValue "") -eq '$.resultInputs[]' -and $schemaPathText.Contains('$.resultInputs[].packageIdentity.nupkgSha256', [StringComparison]::OrdinalIgnoreCase) -and $schemaPathText.Contains('$.resultInputs[].passed', [StringComparison]::OrdinalIgnoreCase) -and $schemaPathText.Contains('$.resultInputs[].ownerReviewer', [StringComparison]::OrdinalIgnoreCase) -and $hashPathText.Contains('$.resultInputs[].stdoutSha256', [StringComparison]::OrdinalIgnoreCase) -and $readyConditionText.Contains('nonSubstituteConfirmations', [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Kit must summarize the actual owner-external-proof-execution-result.input.json resultInputs[] schema, fillable template, hash proof fields, and ready conditions.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-external-consumer-contract" -Passed ([bool](Get-PropertyOrDefault -Object $cleanExternalConsumerContract -Name "mustBeOutsideRepository" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $cleanExternalConsumerContract -Name "mustUsePublicPackageSource" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $cleanExternalConsumerContract -Name "projectReferenceForbidden" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $cleanExternalConsumerContract -Name "localFeedForbiddenForPostPublishProof" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $cleanExternalConsumerContract -Name "directNupkgForbiddenForPostPublishProof" -DefaultValue $false)) -Severity "blocker" -Detail "Clean external consumer proof must be outside the repo and must not use ProjectReference, local feed, or direct nupkg substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-promotion-boundary" -Passed ([bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "ownerResultImportIsInputOnly" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "candidateFromOwnerResultIsInputOnly" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "strictValidatorRequired" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "releaseCloseBridgeInputOnly" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "articlesAndDashboardsAreNotProof" -DefaultValue $false)) -Severity "blocker" -Detail "Owner result import, candidates, bridge output, articles, and dashboards must remain non-proof until strict validators accept real records.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-results-still-required" -Passed $false -Severity "action-required" -Detail "$placeholderFieldCount owner result placeholder(s), $missingExternalLogCount external log(s), and $missingSha256Count SHA256 value(s) still require real owner execution.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "owner-external-execution-results-ready-for-proof-records"
}
elseif ($failedBlockers.Count -eq 0) {
  "blocked-owner-external-execution-results-required"
}
else {
  "invalid-owner-external-execution-result-backfill-kit"
}

$validation = [pscustomobject]@{
  recordKind = "owner-external-execution-result-backfill-kit-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  kitState = $kitState
  isValidBackfillKitShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  backfillLineCount = $backfillLines.Count
  placeholderFieldCount = $placeholderFieldCount
  missingExternalLogCount = $missingExternalLogCount
  missingSha256Count = $missingSha256Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates owner execution backfill guidance only. It cannot publish packages, promote proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-external-execution-result-backfill-kit-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-external-execution-result-backfill-kit-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Owner External Execution Result Backfill Kit Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| kitState | ``$($validation.kitState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| backfillLineCount | ``$($validation.backfillLineCount)`` |
| placeholderFieldCount | ``$($validation.placeholderFieldCount)`` |
| missingExternalLogCount | ``$($validation.missingExternalLogCount)`` |
| missingSha256Count | ``$($validation.missingSha256Count)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner external execution result backfill kit validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner external execution result backfill kit validation failed with $($failedBlockers.Count) blocker(s)."
}
