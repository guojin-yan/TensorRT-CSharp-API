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

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Rule {
  param([int]$Order, [string]$Id, [string]$Requirement, [string]$Rejects)

  [pscustomobject]@{
    order = $Order
    id = $Id
    requirement = $Requirement
    rejects = $Rejects
    ownerActionRequired = $true
  }
}

function New-Evidence {
  param([int]$Order, [string]$Id, [string]$Description, [string]$PathField, [string]$Sha256Field)

  [pscustomobject]@{
    order = $Order
    id = $Id
    description = $Description
    pathField = $PathField
    sha256Field = $Sha256Field
    required = $true
    mustExistForProof = $true
    mustMatchSha256ForProof = -not [string]::IsNullOrWhiteSpace($Sha256Field)
  }
}

$requiredWorkspaceRules = @(
  New-Rule 1 "repository-external-workspace" "CleanConsumer workspace root must resolve outside this repository root." "repo-internal workspace, samples, smoke projects, or source checkout reuse"
  New-Rule 2 "no-project-reference" "CleanConsumer csproj must not contain ProjectReference." "ProjectReference"
  New-Rule 3 "no-local-feed" "Package source must be a real public or owner-approved remote package source." "local feed"
  New-Rule 4 "no-direct-nupkg" "CleanConsumer project must use PackageReference restore, not direct .nupkg file references." "direct .nupkg; direct nupkg"
  New-Rule 5 "real-package-source-required" "Managed/runtime package identities, versions, source URL, and downloaded package SHA256 must be recorded." "build-only, dependency-probe, local package folder"
  New-Rule 6 "logs-and-hashes-required" "Restore, build, run, stdout, stderr, native asset listing, and package logs must carry paths and SHA256 values." "hashless log, summary-only output, dashboard"
  New-Rule 7 "host-metadata-required" "OS, RID, GPU, NVIDIA driver, CUDA, TensorRT, cuDNN, dotnet SDK, and architecture metadata must be recorded." "host metadata without runtime smoke"
  New-Rule 8 "owner-review-required" "Owner must review the external workspace and explicitly confirm no forbidden substitutes were used." "template, candidate, runbook, command pack"
)

$requiredEvidence = @(
  New-Evidence 1 "repositoryExternalWorkspaceRoot" "Absolute root of the repository-external CleanConsumer workspace." "repositoryExternalWorkspaceRoot" ""
  New-Evidence 2 "cleanConsumerCsprojPath" "CleanConsumer project file path used for package restore/build/run." "cleanConsumerCsprojPath" "cleanConsumerCsprojSha256"
  New-Evidence 3 "packageSourceUrl" "Real public or owner-approved remote package source URL." "packageSourceUrl" ""
  New-Evidence 4 "restoreLog" "dotnet restore transcript from the external workspace." "restoreLogPath" "restoreLogSha256"
  New-Evidence 5 "buildLog" "dotnet build transcript from the external workspace." "buildLogPath" "buildLogSha256"
  New-Evidence 6 "runLog" "dotnet run or smoke runner transcript from the external workspace." "runLogPath" "runLogSha256"
  New-Evidence 7 "smokeStdout" "Captured smoke stdout file." "smokeStdoutPath" "smokeStdoutSha256"
  New-Evidence 8 "smokeStderr" "Captured smoke stderr file, even when empty." "smokeStderrPath" "smokeStderrSha256"
  New-Evidence 9 "nativeAssetListing" "Native runtime asset listing captured from restored package output." "nativeAssetListingPath" "nativeAssetListingSha256"
  New-Evidence 10 "managedPackage" "Downloaded managed package SHA256 and package identity." "managedPackagePath" "managedPackageSha256"
  New-Evidence 11 "runtimePackage" "Downloaded runtime package SHA256 and runtime package identity." "runtimePackagePath" "runtimePackageSha256"
  New-Evidence 12 "hostMetadata" "Host metadata JSON captured next to logs." "hostMetadataPath" "hostMetadataSha256"
)

$forbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "direct nupkg",
  "pre-publish smoke reused as post-publish proof",
  "local smoke",
  "build-only",
  "dependency-probe",
  "dashboard",
  "runbook",
  "template",
  "candidate",
  "command pack",
  "failedBlockerCount=0 without real logs",
  "package hash without existing log validation",
  "native asset listing without runtime smoke",
  "host metadata without runtime smoke"
)

$requiredSha256Inputs = @(
  "cleanConsumerCsprojSha256",
  "restoreLogSha256",
  "buildLogSha256",
  "runLogSha256",
  "smokeStdoutSha256",
  "smokeStderrSha256",
  "nativeAssetListingSha256",
  "managedPackageSha256",
  "runtimePackageSha256",
  "hostMetadataSha256"
)

$requiredHostMetadata = @(
  "os",
  "arch",
  "rid",
  "machineName",
  "dotnetSdk",
  "gpuName",
  "nvidiaDriver",
  "cudaRuntimeToolkit",
  "tensorrt",
  "cudnn"
)

$validatorCommands = @(
  "eng/Import-ExternalCleanConsumerExecutionResult.ps1 -OwnerInputPath <owner-input.json> -RequireExistingFiles -RequireHashMatch",
  "eng/Test-ExternalCleanConsumerExecutionResult.ps1 -Strict -RequireExistingFiles -RequireHashMatch -FailOnNotProof",
  "eng/Export-ReleaseEvidenceBundle.ps1",
  "eng/Test-ReleaseEvidenceClassificationAudit.ps1 -Strict",
  "eng/Test-FinalOwnerExecutionCloseReadinessFromRealInput.ps1 -Strict"
)

$record = [pscustomobject]@{
  recordKind = "external-clean-consumer-execution-workspace-contract"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  contractState = "blocked-external-clean-consumer-workspace-contract-required"
  requiredWorkspaceRuleCount = $requiredWorkspaceRules.Count
  requiredEvidenceCount = $requiredEvidence.Count
  forbiddenSubstituteCount = $forbiddenSubstitutes.Count
  requiredSha256InputCount = $requiredSha256Inputs.Count
  requiredHostMetadataCount = $requiredHostMetadata.Count
  requiredWorkspaceRules = @($requiredWorkspaceRules)
  requiredEvidence = @($requiredEvidence)
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  requiredSha256Inputs = @($requiredSha256Inputs)
  requiredHostMetadata = @($requiredHostMetadata)
  validatorCommands = @($validatorCommands)
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "eng/Import-ExternalCleanConsumerExecutionResult.ps1",
    "eng/Test-ExternalCleanConsumerExecutionResult.ps1",
    "artifacts/final-release/final-owner-execution-real-input.template.json"
  )
  boundary = "External CleanConsumer workspace contract only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. It requires real repository-external restore/build/run/smoke logs, stdout/stderr, native asset listing, SHA256 values, package source, and host metadata before any proof candidate can be considered."
}

$jsonPath = Join-Path $OutputRoot "external-clean-consumer-execution-workspace-contract.json"
$markdownPath = Join-Path $OutputRoot "external-clean-consumer-execution-workspace-contract.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$ruleRows = foreach ($rule in $requiredWorkspaceRules) {
  "| ``$($rule.id)`` | $(ConvertTo-MarkdownCell $rule.requirement) | $(ConvertTo-MarkdownCell $rule.rejects) |"
}

$evidenceRows = foreach ($item in $requiredEvidence) {
  "| ``$($item.id)`` | $(ConvertTo-MarkdownCell $item.description) | ``$(ConvertTo-MarkdownCell $item.pathField)`` | ``$(ConvertTo-MarkdownCell $item.sha256Field)`` |"
}

$markdown = @"
# External CleanConsumer Execution Workspace Contract

| Field | Value |
|---|---|
| contractState | ``$($record.contractState)`` |
| ownerActionRequired | ``$($record.ownerActionRequired)`` |
| passed | ``$($record.passed)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Workspace Rules

| ID | Requirement | Rejects |
|---|---|---|
$($ruleRows -join "`r`n")

## Required Evidence

| ID | Description | Path Field | SHA256 Field |
|---|---|---|---|
$($evidenceRows -join "`r`n")

## Forbidden Substitutes

$(@($forbiddenSubstitutes | ForEach-Object { "- ``$_``" }) -join "`r`n")

## Boundary

$($record.boundary)
"@

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
