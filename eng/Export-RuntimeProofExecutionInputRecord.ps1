[CmdletBinding()]
param(
  [string]$ClosurePackPath = "artifacts\final-release\owner-real-proof-execution-closure-pack.json",
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

function Get-RuntimePackageKeyForLane {
  param([string]$ProofLane)

  switch -Regex ($ProofLane) {
    "linux" { return "linux-x64-trt11-cuda13" }
    "post-publish" { return "public-nuget-package" }
    "package-consumer" { return "windows-clean-consumer-runtime" }
    "real-model" { return "real-model-runtime-host" }
    "release-close|strict-close" { return "release-close-owner-host" }
    default { return "owner-runtime-host" }
  }
}

function Get-CommandLineForLane {
  param(
    [string]$ProofLane,
    [string]$FirstCommand
  )

  if (-not [string]::IsNullOrWhiteSpace($FirstCommand)) { return $FirstCommand }

  switch -Regex ($ProofLane) {
    "package-consumer" { return "dotnet restore; dotnet build -c Release; dotnet run --project <clean-consumer-smoke-project> --configuration Release" }
    "post-publish" { return "dotnet add package JYPPX.TensorRtSharp --version <public-version>; dotnet run --project <post-publish-clean-consumer>" }
    "linux" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Collect-LinuxRunnerEvidence.ps1 -RuntimePackageKey <linux-runtime-package-key>" }
    "real-model" { return "dotnet run --project samples/YoloVision -- --model <owner-model.onnx> --image <owner-image> --runtime <runtime-package-key>" }
    "release-close|strict-close" { return "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady" }
    default { return "Run the owner-approved real proof command and capture stdout, stderr, merged transcript, hashes, and validator output." }
  }
}

function New-RequiredInputFields {
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
    "commandLine",
    "workingDirectory",
    "stdoutPath",
    "stderrPath",
    "stdoutSha256",
    "stderrSha256",
    "mergedTranscriptPath",
    "mergedTranscriptSha256",
    "validatorOutputPath",
    "validatorOutputSha256",
    "ownerReviewer",
    "reviewTimestampUtc",
    "forbiddenSubstituteChecks"
  )
}

function New-ExecutionInput {
  param([object]$ClosureItem)

  $candidateId = [string](Get-PropertyOrDefault -Object $ClosureItem -Name "candidateId" -DefaultValue "unknown-candidate")
  $proofLane = [string](Get-PropertyOrDefault -Object $ClosureItem -Name "proofLane" -DefaultValue "unknown-proof-lane")
  $firstCommand = [string](Get-PropertyOrDefault -Object $ClosureItem -Name "firstCommand" -DefaultValue "")
  $requiredInputFields = @(New-RequiredInputFields)

  [pscustomobject]@{
    executionInputId = "$candidateId-runtime-proof-execution-input"
    candidateId = $candidateId
    proofLane = $proofLane
    runtimePackageKey = Get-RuntimePackageKeyForLane -ProofLane $proofLane
    hostMetadata = [pscustomobject]@{
      os = "<owner-fill-os>"
      arch = "<owner-fill-arch>"
      gpu = "<owner-fill-gpu>"
      driverVersion = "<owner-fill-driver-version>"
      cudaVersion = "<owner-fill-cuda-version>"
      tensorRtVersion = "<owner-fill-tensorrt-version>"
      dotnetVersion = "<owner-fill-dotnet-version>"
    }
    packageIdentity = [pscustomobject]@{
      packageId = "<owner-fill-package-id>"
      packageVersion = "<owner-fill-package-version>"
      nupkgPath = "<owner-fill-existing-nupkg-or-public-package-reference>"
      nupkgSha256 = "<owner-fill-nupkg-sha256>"
      packageSource = "<owner-fill-public-package-source>"
    }
    commandLine = Get-CommandLineForLane -ProofLane $proofLane -FirstCommand $firstCommand
    workingDirectory = "<owner-fill-clean-working-directory-outside-repo-when-required>"
    stdoutPath = "<owner-fill-stdout-log-path>"
    stderrPath = "<owner-fill-stderr-log-path>"
    stdoutSha256 = "<owner-fill-stdout-sha256>"
    stderrSha256 = "<owner-fill-stderr-sha256>"
    mergedTranscriptPath = "<owner-fill-merged-transcript-path>"
    mergedTranscriptSha256 = "<owner-fill-merged-transcript-sha256>"
    validatorOutputPath = "<owner-fill-validator-output-path>"
    validatorOutputSha256 = "<owner-fill-validator-output-sha256>"
    ownerReviewer = "<owner-fill-reviewer>"
    reviewTimestampUtc = "<owner-fill-review-timestamp-utc>"
    forbiddenSubstituteChecks = @(
      "local feed is not public package proof",
      "ProjectReference is not package-consumer proof",
      "direct nupkg install is not post-publish proof",
      "DependencyProbe-only is not runtime proof",
      "sidecar-only is not runtime proof",
      "build-only is not runtime proof",
      "precheck-only is not runtime proof",
      "placeholder hashes are not real evidence"
    )
    expectedArtifacts = @(Get-PropertyOrDefault -Object $ClosureItem -Name "expectedArtifacts" -DefaultValue @())
    requiredLogs = @(Get-PropertyOrDefault -Object $ClosureItem -Name "requiredLogs" -DefaultValue @())
    requiredSha256 = @(Get-PropertyOrDefault -Object $ClosureItem -Name "requiredSha256" -DefaultValue @())
    validatorCommands = @(Get-PropertyOrDefault -Object $ClosureItem -Name "validatorCommands" -DefaultValue @())
    sourceClosureItemId = [string](Get-PropertyOrDefault -Object $ClosureItem -Name "closureItemId" -DefaultValue "")
    sourceBlockedReasons = @(Get-PropertyOrDefault -Object $ClosureItem -Name "blockedReasons" -DefaultValue @())
    inputState = "blocked-runtime-proof-execution-input-required"
    requiredInputFields = $requiredInputFields
    missingInputFields = $requiredInputFields
    nonSubstituteBoundary = "This execution input record is an owner-fill template. It is not runtime proof until owner-filled real logs, SHA256 hashes, package identity, host metadata, validator outputs, reviewer, and timestamp pass strict validation."
    readyForRuntimeProofValidation = $false
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$resolvedClosurePackPath = Resolve-InputPath -Path $ClosurePackPath
if (-not (Test-Path -LiteralPath $resolvedClosurePackPath -PathType Leaf)) {
  throw "Owner real proof execution closure pack not found: $resolvedClosurePackPath"
}

$closurePack = Get-Content -LiteralPath $resolvedClosurePackPath -Raw -Encoding utf8 | ConvertFrom-Json
$closureItems = @(Get-PropertyOrDefault -Object $closurePack -Name "closureItems" -DefaultValue @())
$executionInputs = @($closureItems | ForEach-Object { New-ExecutionInput -ClosureItem $_ })
$readyExecutionInputCount = @($executionInputs | Where-Object { [bool]$_.readyForRuntimeProofValidation }).Count
$blockedExecutionInputCount = @($executionInputs | Where-Object { [string]$_.inputState -eq "blocked-runtime-proof-execution-input-required" }).Count

$record = [pscustomobject]@{
  recordKind = "runtime-proof-execution-input-record"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  closurePackPath = $resolvedClosurePackPath
  inputState = "blocked-runtime-proof-execution-input-required"
  executionInputCount = $executionInputs.Count
  blockedExecutionInputCount = $blockedExecutionInputCount
  readyExecutionInputCount = $readyExecutionInputCount
  executionInputs = @($executionInputs)
  forbiddenSubstitutePolicy = @(
    "No local feed, ProjectReference, direct nupkg, DependencyProbe-only, sidecar-only, build-only, precheck-only, placeholder path, or placeholder hash may promote runtime proof.",
    "All proof lanes require owner-filled host metadata, package identity, command capture, stdout/stderr logs, merged transcript, SHA256 values, validator output, owner reviewer, and review timestamp."
  )
  sourceArtifacts = @(
    "artifacts/final-release/owner-real-proof-execution-closure-pack.json",
    "artifacts/final-release/owner-real-proof-execution-closure-pack-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This record is an owner-fill execution input surface only. It cannot run runtime proof, publish packages, verify post-publish state, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "runtime-proof-execution-input-record.json"
$markdownPath = Join-Path $OutputRoot "runtime-proof-execution-input-record.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Runtime Proof Execution Input Record")
$lines.Add("")
$lines.Add("`runtime-proof-execution-input-record` 将 Owner closure item 转为可填写、可验证的真实执行输入槽位。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| inputState | ``$(ConvertTo-MarkdownCell $record.inputState)`` |")
$lines.Add("| executionInputCount | ``$($record.executionInputCount)`` |")
$lines.Add("| blockedExecutionInputCount | ``$($record.blockedExecutionInputCount)`` |")
$lines.Add("| readyExecutionInputCount | ``$($record.readyExecutionInputCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("| canPublishPublicly | ``$($record.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($record.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($record.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Execution Inputs")
$lines.Add("")
$lines.Add("| Input | Lane | Runtime Package | State | Missing Fields | Command |")
$lines.Add("| --- | --- | --- | --- | ---: | --- |")
foreach ($input in $executionInputs) {
  $lines.Add("| $(ConvertTo-MarkdownCell $input.executionInputId) | $(ConvertTo-MarkdownCell $input.proofLane) | $(ConvertTo-MarkdownCell $input.runtimePackageKey) | $(ConvertTo-MarkdownCell $input.inputState) | ``$(@($input.missingInputFields).Count)`` | $(ConvertTo-MarkdownCell $input.commandLine) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Runtime proof execution input record written to $jsonPath"
Write-Host "Runtime proof execution input record markdown written to $markdownPath"
Write-Host "InputState=$($record.inputState) Inputs=$($record.executionInputCount) Blocked=$($record.blockedExecutionInputCount) Ready=$($record.readyExecutionInputCount)"
