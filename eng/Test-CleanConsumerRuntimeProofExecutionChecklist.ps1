[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\clean-consumer-runtime-proof-execution-checklist.json",
  [string]$OutputRoot = "artifacts\final-release",
  [switch]$Strict,
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

function ConvertTo-Array {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-StringArrayProperty {
  param([object]$Record, [string]$Name)

  return @((ConvertTo-Array (Get-PropertyOrDefault -Object $Record -Name $Name -DefaultValue @())) | ForEach-Object { [string]$_ })
}

function Get-IdArrayProperty {
  param([object]$Record, [string]$Name)

  return @((ConvertTo-Array (Get-PropertyOrDefault -Object $Record -Name $Name -DefaultValue @())) | ForEach-Object {
    [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
  })
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Clean consumer runtime proof execution checklist not found: $resolvedInputPath"
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$executionSteps = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "executionSteps" -DefaultValue @())
$stepIds = @($executionSteps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$stepCommands = @($executionSteps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "commandTemplate" -DefaultValue "") })
$requiredInputs = Get-IdArrayProperty -Record $record -Name "requiredInputs"
$requiredHashes = Get-IdArrayProperty -Record $record -Name "requiredHashes"
$requiredLogs = Get-IdArrayProperty -Record $record -Name "requiredLogs"
$requiredHostMetadata = Get-IdArrayProperty -Record $record -Name "requiredHostMetadata"
$requiredPackageMetadata = Get-IdArrayProperty -Record $record -Name "requiredPackageMetadata"
$requiredValidators = Get-StringArrayProperty -Record $record -Name "requiredValidators"
$forbiddenSubstitutes = Get-StringArrayProperty -Record $record -Name "forbiddenSubstitutes"
$sourceArtifacts = Get-StringArrayProperty -Record $record -Name "sourceArtifacts"
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$requiredStepIds = @(
  "create-repository-external-clean-consumer",
  "configure-public-package-source",
  "install-managed-and-runtime-packages",
  "restore-clean-consumer",
  "build-clean-consumer",
  "run-smoke-runtime-command",
  "persist-stdout-stderr-logs",
  "compute-log-sha256",
  "compute-nupkg-sha256",
  "fill-owner-input",
  "import-owner-input",
  "strict-validate-proof-record",
  "run-forbidden-substitute-scan",
  "refresh-release-evidence-and-dashboard"
)
$requiredInputIds = @(
  "public-package-source-url",
  "managed-package-id-version",
  "runtime-package-id-version",
  "repository-external-consumer-path",
  "restore-build-smoke-commands",
  "smoke-exit-code-zero",
  "startedAtUtc",
  "finishedAtUtc",
  "owner-review-identity"
)
$requiredHashIds = @(
  "clean-consumer-project-hash",
  "managed-nupkg-sha256",
  "runtime-nupkg-sha256",
  "restore-log-sha256",
  "build-log-sha256",
  "stdout-log-sha256",
  "stderr-log-sha256"
)
$requiredLogIds = @(
  "restore-log",
  "build-log",
  "smoke-stdout-log",
  "smoke-stderr-log",
  "owner-import-log",
  "strict-validator-log"
)
$requiredHostIds = @(
  "host-os-architecture",
  "gpu-name",
  "nvidia-driver-version",
  "cuda-version",
  "tensorrt-version",
  "cudnn-version"
)
$requiredPackageIds = @(
  "managed-package-id",
  "managed-package-version",
  "runtime-package-id",
  "runtime-package-version",
  "runtime-package-key",
  "public-package-source"
)
$requiredForbidden = @(
  "template",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "dry-run",
  "build-only",
  "preflight-only",
  "GUI screenshot",
  "TensorRtExec report",
  "YoloVision matrix",
  "OnnxToEngine report",
  "sample manifest",
  "sidecar-only",
  "readonly diagnostics",
  "dependency probe",
  "owner input without strict validator pass"
)
$requiredSources = @(
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-proof-owner-handoff-pack.json",
  "artifacts/final-release/final-proof-owner-handoff-pack-validation.json",
  "artifacts/final-release/final-release-pre-publish-audit-matrix.json",
  "artifacts/final-release/tensor-rt-exec-gui-cli-parity-checklist.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
  "artifacts/final-release/package-consumer-runtime-proof-record-validation.json"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "clean-consumer-runtime-proof-execution-checklist") -Severity "blocker" -Detail "recordKind must be clean-consumer-runtime-proof-execution-checklist.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "checklistState" -DefaultValue "") -eq "blocked-owner-clean-consumer-runtime-proof-required") -Severity "blocker" -Detail "Checklist must remain owner-blocked until real clean consumer runtime proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-close-or-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true)) -Severity "blocker" -Detail "Checklist must not publish, close, or promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "execution-steps" -Passed ($executionSteps.Count -ge 14 -and @($requiredStepIds | Where-Object { $stepIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must include the full clean consumer execution path.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-dotnet-nuget-push" -Passed (@($stepCommands | Where-Object { $_.Contains("dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) }).Count -eq 0) -Severity "blocker" -Detail "Checklist must never include dotnet nuget push.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-inputs" -Passed (@($requiredInputIds | Where-Object { $requiredInputs -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must identify owner input fields for public source, package ids, clean consumer, commands, smoke exit code, and owner review.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-hashes" -Passed (@($requiredHashIds | Where-Object { $requiredHashes -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must require project, package, build, restore, stdout, and stderr hashes.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-logs" -Passed (@($requiredLogIds | Where-Object { $requiredLogs -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must require restore/build/smoke/import/validator logs.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-host-metadata" -Passed (@($requiredHostIds | Where-Object { $requiredHostMetadata -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must require OS, GPU, driver, CUDA, TensorRT, and cuDNN host metadata.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-package-metadata" -Passed (@($requiredPackageIds | Where-Object { $requiredPackageMetadata -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must require managed/runtime package metadata and public source.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-validators" -Passed (@($requiredValidators | Where-Object { $_.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-Strict", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-RequireExistingLog", [StringComparison]::OrdinalIgnoreCase) -and $_.Contains("-FailOnNotProof", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($requiredValidators | Where-Object { $_.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($requiredValidators | Where-Object { $_.Contains("Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1 -and @($requiredValidators | Where-Object { $_.Contains("Export-ReleaseEvidenceBundle.ps1", [StringComparison]::OrdinalIgnoreCase) }).Count -ge 1) -Severity "blocker" -Detail "Checklist must link strict proof, import, forbidden scan, and release refresh validators.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed (@($requiredForbidden | Where-Object { $forbiddenSubstitutes -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must enumerate every forbidden substitute proof source.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts" -Passed (@($requiredSources | Where-Object { $sourceArtifacts -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checklist must source the release evidence, final handoff, prepublish, parity, owner schema, and proof validator artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($boundary.Contains("owner execution guidance", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("real clean external package-consumer-runtime", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("strict validator pass", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must clearly state guidance-only and strict real proof requirements.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "clean-consumer-runtime-proof-execution-checklist-ready" } else { "blocked-clean-consumer-runtime-proof-execution-checklist-invalid" }

$validation = [pscustomobject]@{
  recordKind = "clean-consumer-runtime-proof-execution-checklist-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  validationItems = $validationItems
  boundary = "Validation confirms the checklist shape and proof boundaries only; it is not package-consumer-runtime proof."
}

$jsonPath = Join-Path $OutputRoot "clean-consumer-runtime-proof-execution-checklist-validation.json"
$markdownPath = Join-Path $OutputRoot "clean-consumer-runtime-proof-execution-checklist-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Clean Consumer Runtime Proof Execution Checklist Validation

| Field | Value |
| --- | --- |
| validationState | ``$($validation.validationState)`` |
| validationItemCount | ``$($validation.validationItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean consumer runtime proof execution checklist validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
