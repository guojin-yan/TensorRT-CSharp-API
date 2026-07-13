[CmdletBinding()]
param(
  [string]$OwnerRuntimeProofResultInputTemplatePath = "artifacts\final-release\owner-runtime-proof-result-input.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
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

function New-ExternalResultInput {
  param([object]$TemplateInput)

  $requiredFields = @(Get-PropertyOrDefault -Object $TemplateInput -Name "requiredResultFields" -DefaultValue @())
  if ($requiredFields -notcontains "passed") {
    $requiredFields += "passed"
  }

  [pscustomobject]@{
    resultInputId = [string](Get-PropertyOrDefault -Object $TemplateInput -Name "resultInputId" -DefaultValue "unknown-result-input")
    executionInputId = [string](Get-PropertyOrDefault -Object $TemplateInput -Name "executionInputId" -DefaultValue "unknown-execution-input")
    candidateId = [string](Get-PropertyOrDefault -Object $TemplateInput -Name "candidateId" -DefaultValue "unknown-candidate")
    proofLane = [string](Get-PropertyOrDefault -Object $TemplateInput -Name "proofLane" -DefaultValue "unknown-proof-lane")
    runtimePackageKey = [string](Get-PropertyOrDefault -Object $TemplateInput -Name "runtimePackageKey" -DefaultValue "unknown-runtime-package-key")
    sourceCommandLine = [string](Get-PropertyOrDefault -Object $TemplateInput -Name "sourceCommandLine" -DefaultValue "")
    hostMetadata = Get-PropertyOrDefault -Object $TemplateInput -Name "hostMetadata" -DefaultValue ([pscustomobject]@{})
    packageIdentity = Get-PropertyOrDefault -Object $TemplateInput -Name "packageIdentity" -DefaultValue ([pscustomobject]@{})
    executedCommandLine = "<owner-fill-executed-command-line>"
    workingDirectory = "<owner-fill-working-directory>"
    stdoutPath = "<owner-fill-existing-stdout-log-path>"
    stdoutSha256 = "<owner-fill-stdout-sha256>"
    stderrPath = "<owner-fill-existing-stderr-log-path>"
    stderrSha256 = "<owner-fill-stderr-sha256>"
    mergedTranscriptPath = "<owner-fill-existing-merged-transcript-path>"
    mergedTranscriptSha256 = "<owner-fill-merged-transcript-sha256>"
    validatorOutputPath = "<owner-fill-existing-validator-output-path>"
    validatorOutputSha256 = "<owner-fill-validator-output-sha256>"
    exitCode = "<owner-fill-exit-code>"
    passed = "<owner-fill-true-after-real-command-exit-0>"
    startedAtUtc = "<owner-fill-started-at-utc>"
    endedAtUtc = "<owner-fill-ended-at-utc>"
    ownerReviewer = "<owner-fill-reviewer>"
    ownerReviewTimestampUtc = "<owner-fill-owner-review-timestamp-utc>"
    nonSubstituteConfirmations = @(
      "public package source used where required",
      "no local feed substituted for public package proof",
      "no ProjectReference substituted for package-consumer proof",
      "no direct .nupkg substituted for post-publish proof",
      "runtime command executed, not DependencyProbe-only",
      "runtime command executed, not sidecar-only",
      "runtime command executed, not build-only",
      "runtime command executed, not dry-run",
      "run was not skipped",
      "hashes match the referenced files"
    )
    validatorCommands = @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-OwnerExternalProofExecutionResult.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofRecordImportValidator.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofRecordImportValidator.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRecordCandidateFromOwnerResultImport.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict"
    )
    requiredResultFields = $requiredFields
    ownerFillStatus = "owner-external-real-execution-required"
    readyForRealProofRecordImport = $false
    strictValidatorInputOnly = $true
    canPromoteLaneResult = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "This resultInputs[] item is a fillable owner external execution result template only. It is not proof until real files exist, hashes match, exitCode is 0, passed is true, owner review is complete, non-substitute confirmations are present, and strict validators accept the imported result."
  }
}

$resolvedTemplatePath = Resolve-RepositoryPath -Path $OwnerRuntimeProofResultInputTemplatePath
if (-not (Test-Path -LiteralPath $resolvedTemplatePath -PathType Leaf)) {
  throw "Owner runtime proof result input template not found: $resolvedTemplatePath"
}

$runtimeTemplate = Get-Content -LiteralPath $resolvedTemplatePath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeResultInputs = @(Get-PropertyOrDefault -Object $runtimeTemplate -Name "resultInputs" -DefaultValue @())
$externalResultInputs = @($runtimeResultInputs | ForEach-Object { New-ExternalResultInput -TemplateInput $_ })

$record = [pscustomobject]@{
  recordKind = "owner-external-proof-execution-result-input-template"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  templateState = "blocked-owner-external-proof-execution-result-required"
  importTarget = "artifacts/final-release/owner-external-proof-execution-result.input.json"
  sourceOwnerRuntimeProofResultInputTemplate = "artifacts/final-release/owner-runtime-proof-result-input.template.json"
  resultInputCount = $externalResultInputs.Count
  blockedResultInputCount = @($externalResultInputs | Where-Object { [string]$_.ownerFillStatus -eq "owner-external-real-execution-required" }).Count
  readyForRealProofRecordImportCount = @($externalResultInputs | Where-Object { [bool]$_.readyForRealProofRecordImport }).Count
  resultInputs = @($externalResultInputs)
  requiredImportCommands = @(
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-OwnerExternalProofExecutionResult.ps1",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofRecordImportValidator.ps1",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofRecordImportValidator.ps1 -Strict",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRecordCandidateFromOwnerResultImport.ps1",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict"
  )
  forbiddenSubstitutes = @(
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "candidate",
    "draft",
    "dashboard",
    "dry-run",
    "dependency-probe-only",
    "blocked-by-cuda-driver",
    "build-only",
    "schema-only",
    "template-only"
  )
  requiredReadyConditions = @(
    "Owner copies this template to artifacts/final-release/owner-external-proof-execution-result.input.json",
    "resultInputId matches owner-runtime-proof-result-input.template.json",
    "packageIdentity.nupkgPath, stdoutPath, stderrPath, mergedTranscriptPath, and validatorOutputPath point to existing files",
    "all SHA256 values are 64 hex characters and match referenced files",
    "exitCode is 0 and passed is true",
    "ownerReviewer and ownerReviewTimestampUtc are filled",
    "nonSubstituteConfirmations has at least 10 entries",
    "forbidden substitute markers are absent"
  )
  strictValidatorInputOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This template is owner input scaffolding only. It cannot publish, prove runtime execution, prove post-publish verification, or close a release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-external-proof-execution-result.input.template.json"
$markdownPath = Join-Path $OutputRoot "owner-external-proof-execution-result.input.template.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner External Proof Execution Result Input Template")
$lines.Add("")
$lines.Add("该模板用于生成 ``owner-external-proof-execution-result.input.json`` 的填写起点。它不是 proof，不会执行发布，也不会关闭 release issue。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| templateState | ``$(ConvertTo-MarkdownCell $record.templateState)`` |")
$lines.Add("| importTarget | ``$(ConvertTo-MarkdownCell $record.importTarget)`` |")
$lines.Add("| resultInputCount | ``$($record.resultInputCount)`` |")
$lines.Add("| blockedResultInputCount | ``$($record.blockedResultInputCount)`` |")
$lines.Add("| readyForRealProofRecordImportCount | ``$($record.readyForRealProofRecordImportCount)`` |")
$lines.Add("| strictValidatorInputOnly | ``$($record.strictValidatorInputOnly)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Result Inputs")
$lines.Add("")
$lines.Add("| Result Input | Lane | Runtime Package | Status |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($input in $externalResultInputs) {
  $lines.Add("| $(ConvertTo-MarkdownCell $input.resultInputId) | $(ConvertTo-MarkdownCell $input.proofLane) | $(ConvertTo-MarkdownCell $input.runtimePackageKey) | $(ConvertTo-MarkdownCell $input.ownerFillStatus) |")
}
$lines.Add("")
$lines.Add("## Required Ready Conditions")
$lines.Add("")
foreach ($condition in $record.requiredReadyConditions) {
  $lines.Add("- ``$(ConvertTo-MarkdownCell $condition)``")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner external proof execution result input template written to $jsonPath"
Write-Host "Owner external proof execution result input template markdown written to $markdownPath"
Write-Host "TemplateState=$($record.templateState) Items=$($record.resultInputCount) Blocked=$($record.blockedResultInputCount) Ready=$($record.readyForRealProofRecordImportCount)"
