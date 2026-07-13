[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\package-consumer-preflight.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-PackageConsumerPreflight.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$text = $record | ConvertTo-Json -Depth 12
$items = New-Object System.Collections.Generic.List[object]
$metadataChecks = @(Get-PropertyOrDefault -Object $record -Name "metadataChecks" -DefaultValue @())
$consumerChecks = @(Get-PropertyOrDefault -Object $record -Name "consumerBoundaryChecks" -DefaultValue @())
$passedCheckIds = @($metadataChecks + $consumerChecks | Where-Object { [bool]$_.passed } | ForEach-Object { [string]$_.id })

$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "package-consumer-preflight") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "ready-non-proof" ([string](Get-PropertyOrDefault -Object $record -Name "preflightState" -DefaultValue "") -eq "package-consumer-preflight-ready-non-proof") "blocker" "Preflight must be ready but explicitly non-proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "metadata-coverage" ([int](Get-PropertyOrDefault -Object $record -Name "metadataCheckCount" -DefaultValue 0) -ge 8 -and [int](Get-PropertyOrDefault -Object $record -Name "runtimeProjectCount" -DefaultValue 0) -ge 6 -and [int](Get-PropertyOrDefault -Object $record -Name "runtimeSplitProjectCount" -DefaultValue 0) -ge 6) "blocker" "Preflight must cover managed, runtime, and split runtime package metadata.")) | Out-Null
$items.Add((New-OwnerValidationItem "consumer-boundary" ($text.Contains("Test-PackageConsumer.ps1") -and $passedCheckIds -contains "consumer-script-uses-nuget-config" -and $passedCheckIds -contains "consumer-script-records-forbidden-substitutes" -and $passedCheckIds -contains "consumer-script-keeps-non-proof") "blocker" "Preflight must bind to package consumer validation guardrails.")) | Out-Null
$items.Add((New-OwnerValidationItem "project-reference-boundary" ($text.Contains("ReferenceOutputAssembly=false") -and $text.Contains("forbidden as package-consumer-runtime proof")) "blocker" "ProjectReference must remain a pack build input only, never proof.")) | Out-Null
$items.Add((New-OwnerValidationItem "forbidden-substitutes" ($text.Contains("local feed") -and $text.Contains("ProjectReference") -and $text.Contains("direct .nupkg") -and $text.Contains("TensorRtExec report") -and $text.Contains("sidecar-only report")) "blocker" "Forbidden proof substitutes must remain visible.")) | Out-Null
$items.Add((New-OwnerValidationItem "owner-gate-blocked" ([string](Get-PropertyOrDefault -Object $record -Name "finalOwnerGateState" -DefaultValue "") -eq "blocked-final-owner-next-decision-required" -and [string](Get-PropertyOrDefault -Object $record -Name "recommendedOwnerDefault" -DefaultValue "") -eq "keep-blocked-wait-for-owner") "blocker" "Owner gate must remain blocked by default.")) | Out-Null
$items.Add((New-OwnerValidationItem "side-effect-free" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromotePackageConsumerRuntimeProof" -DefaultValue $true)) "blocker" "Preflight must not publish or promote package-consumer runtime proof.")) | Out-Null

$failed = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$failedBlockerCount = [int]$failed.Count
$state = if ($failedBlockerCount -eq 0) { "package-consumer-preflight-validation-ready-non-proof" } else { "blocked-package-consumer-preflight-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "package-consumer-preflight-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failedBlockerCount
  validationItemCount = [int]$items.Count
  validationItems = @($items.ToArray())
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  canPromotePackageConsumerRuntimeProof = $false
  boundary = "Package consumer preflight validation only; not package publication, GitHub Packages publish, public package-consumer proof, workflow dispatch, proof promotion, or release close approval."
}

$jsonPath = Join-Path $OutputRoot "package-consumer-preflight-validation.json"
$mdPath = Join-Path $OutputRoot "package-consumer-preflight-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Package Consumer Preflight Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$failedBlockerCount``",
  "- validationItemCount: ``$($validation.validationItemCount)``",
  "",
  $validation.boundary
)
Write-Host "PackageConsumerPreflightValidationState=$state FailedBlockers=$failedBlockerCount Items=$($validation.validationItemCount)"
if ($Strict.IsPresent -and $failedBlockerCount -gt 0) { throw "Package consumer preflight validation failed." }
