[CmdletBinding()]
param(
  [string]$RuntimeProofInputPath = "artifacts\final-release\runtime-proof-execution-input-record.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-RequiredResultFields {
  @(
    "hostMetadata.os",
    "hostMetadata.arch",
    "hostMetadata.gpu",
    "hostMetadata.driverVersion",
    "hostMetadata.cudaVersion",
    "hostMetadata.tensorRtVersion",
    "hostMetadata.dotnetVersion",
    "packageIdentity.packageId",
    "packageIdentity.packageVersion",
    "packageIdentity.nupkgPath",
    "packageIdentity.nupkgSha256",
    "packageIdentity.packageSource",
    "executedCommandLine",
    "workingDirectory",
    "stdoutPath",
    "stderrPath",
    "stdoutSha256",
    "stderrSha256",
    "mergedTranscriptPath",
    "mergedTranscriptSha256",
    "validatorOutputPath",
    "validatorOutputSha256",
    "exitCode",
    "startedAtUtc",
    "endedAtUtc",
    "ownerReviewer",
    "ownerReviewTimestampUtc",
    "nonSubstituteConfirmations"
  )
}

function New-ResultInput {
  param([object]$ExecutionInput)

  $executionInputId = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "executionInputId" -DefaultValue "unknown-execution-input")
  $candidateId = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "candidateId" -DefaultValue "unknown-candidate")
  $proofLane = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "proofLane" -DefaultValue "unknown-proof-lane")
  $runtimePackageKey = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "runtimePackageKey" -DefaultValue "unknown-runtime-package-key")
  $requiredFields = @(New-RequiredResultFields)

  [pscustomobject]@{
    resultInputId = "$executionInputId-owner-result-input"
    executionInputId = $executionInputId
    candidateId = $candidateId
    proofLane = $proofLane
    runtimePackageKey = $runtimePackageKey
    sourceCommandLine = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "commandLine" -DefaultValue "")
    hostMetadata = Get-PropertyOrDefault -Object $ExecutionInput -Name "hostMetadata" -DefaultValue ([pscustomobject]@{})
    packageIdentity = Get-PropertyOrDefault -Object $ExecutionInput -Name "packageIdentity" -DefaultValue ([pscustomobject]@{})
    executedCommandLine = "<owner-fill-executed-command-line>"
    workingDirectory = "<owner-fill-working-directory>"
    stdoutPath = "<owner-fill-existing-stdout-log-path>"
    stderrPath = "<owner-fill-existing-stderr-log-path>"
    stdoutSha256 = "<owner-fill-stdout-sha256>"
    stderrSha256 = "<owner-fill-stderr-sha256>"
    mergedTranscriptPath = "<owner-fill-existing-merged-transcript-path>"
    mergedTranscriptSha256 = "<owner-fill-merged-transcript-sha256>"
    validatorOutputPath = "<owner-fill-existing-validator-output-path>"
    validatorOutputSha256 = "<owner-fill-validator-output-sha256>"
    exitCode = "<owner-fill-exit-code>"
    startedAtUtc = "<owner-fill-started-at-utc>"
    endedAtUtc = "<owner-fill-ended-at-utc>"
    ownerReviewer = "<owner-fill-reviewer>"
    ownerReviewTimestampUtc = "<owner-fill-owner-review-timestamp-utc>"
    nonSubstituteConfirmations = @(
      "public package source used where required",
      "no local feed substituted for public package proof",
      "no ProjectReference substituted for package-consumer proof",
      "no direct nupkg substituted for post-publish proof",
      "runtime command executed, not DependencyProbe-only",
      "runtime command executed, not sidecar-only",
      "runtime command executed, not build-only",
      "runtime command executed, not precheck-only",
      "run was not skipped",
      "hashes match the referenced files"
    )
    validatorCommands = @(Get-PropertyOrDefault -Object $ExecutionInput -Name "validatorCommands" -DefaultValue @())
    expectedArtifacts = @(Get-PropertyOrDefault -Object $ExecutionInput -Name "expectedArtifacts" -DefaultValue @())
    requiredResultFields = $requiredFields
    missingResultFields = $requiredFields
    resultInputState = "blocked-owner-runtime-proof-result-input-required"
    readyForRuntimeProofValidation = $false
    proofClassification = "template-only-not-proof"
    nonSubstituteBoundary = "This owner runtime proof result input is a fillable template only. It is not runtime proof until real existing files, SHA256 hashes, host metadata, package identity, exit code, owner review, and strict validators pass."
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$resolvedInputPath = Resolve-InputPath -Path $RuntimeProofInputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Runtime proof execution input record not found: $resolvedInputPath"
}

$inputRecord = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$executionInputs = @(Get-PropertyOrDefault -Object $inputRecord -Name "executionInputs" -DefaultValue @())
$resultInputs = @($executionInputs | ForEach-Object { New-ResultInput -ExecutionInput $_ })
$blockedResultInputCount = @($resultInputs | Where-Object { [string]$_.resultInputState -eq "blocked-owner-runtime-proof-result-input-required" }).Count
$readyResultInputCount = @($resultInputs | Where-Object { [bool]$_.readyForRuntimeProofValidation }).Count

$record = [pscustomobject]@{
  recordKind = "owner-runtime-proof-result-input-template"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  runtimeProofInputPath = $resolvedInputPath
  templateState = "blocked-owner-runtime-proof-result-input-required"
  resultInputCount = $resultInputs.Count
  blockedResultInputCount = $blockedResultInputCount
  readyResultInputCount = $readyResultInputCount
  resultInputs = @($resultInputs)
  sourceArtifacts = @(
    "artifacts/final-release/runtime-proof-execution-input-record.json",
    "artifacts/final-release/runtime-proof-execution-input-record-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This template is an owner-fill result input surface only. It cannot substitute real runtime proof, package publish, post-publish verification, rollback approval, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-runtime-proof-result-input.template.json"
$markdownPath = Join-Path $OutputRoot "owner-runtime-proof-result-input.template.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 18)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Runtime Proof Result Input Template")
$lines.Add("")
$lines.Add("`owner-runtime-proof-result-input.template` 将 runtime proof execution input 转为 Owner 真实执行结果回填模板。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| templateState | ``$(ConvertTo-MarkdownCell $record.templateState)`` |")
$lines.Add("| resultInputCount | ``$($record.resultInputCount)`` |")
$lines.Add("| blockedResultInputCount | ``$($record.blockedResultInputCount)`` |")
$lines.Add("| readyResultInputCount | ``$($record.readyResultInputCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Result Inputs")
$lines.Add("")
$lines.Add("| Result Input | Lane | Runtime Package | State | Missing Fields |")
$lines.Add("| --- | --- | --- | --- | ---: |")
foreach ($input in $resultInputs) {
  $lines.Add("| $(ConvertTo-MarkdownCell $input.resultInputId) | $(ConvertTo-MarkdownCell $input.proofLane) | $(ConvertTo-MarkdownCell $input.runtimePackageKey) | $(ConvertTo-MarkdownCell $input.resultInputState) | ``$(@($input.missingResultFields).Count)`` |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner runtime proof result input template written to $jsonPath"
Write-Host "Owner runtime proof result input template markdown written to $markdownPath"
Write-Host "TemplateState=$($record.templateState) Items=$($record.resultInputCount) Blocked=$($record.blockedResultInputCount) Ready=$($record.readyResultInputCount)"
