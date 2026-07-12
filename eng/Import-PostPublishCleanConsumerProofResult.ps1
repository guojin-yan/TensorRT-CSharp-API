[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\post-publish-clean-consumer-proof-result.template.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$RequireExistingFiles,
  [switch]$RequireHashMatch,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("OwnerInputPath", "OutputRoot")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $pathName).Value)) {
    Set-Variable -Name $pathName -Value (Join-Path $RepositoryRoot (Get-Variable $pathName).Value)
  }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}
function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function Resolve-InputPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Test-Placeholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text.StartsWith("<owner-", [StringComparison]::OrdinalIgnoreCase) -or $text.Contains("example-not-real-proof", [StringComparison]::OrdinalIgnoreCase)
}

function Test-Sha256 {
  param([AllowNull()][object]$Value)
  return [System.Text.RegularExpressions.Regex]::IsMatch([string]$Value, "^[0-9a-fA-F]{64}$")
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Finding {
  param([string]$Id, [string]$Severity, [string]$Category, [string]$Message)
  [pscustomobject]@{ id = $Id; severity = $Severity; category = $Category; message = $Message; ownerActionRequired = $true }
}

function New-TemplateRecord {
  [pscustomobject]@{
    recordKind = "post-publish-clean-consumer-proof-result-owner-input"
    ownerInputState = "blocked-post-publish-clean-consumer-proof-result-required"
    publicPackageSourceUrl = "<owner-public-package-source-url>"
    publicPackageUrl = "<owner-public-package-url>"
    managedPackageId = "JYPPX.TensorRtSharp"
    managedPackageVersion = "<owner-package-version>"
    runtimePackageId = "JYPPX.TensorRtSharp.Native.<runtime-key>"
    runtimePackageVersion = "<owner-package-version>"
    runtimePackageKey = "<owner-runtime-key>"
    downloadedManagedPackagePath = "<owner-downloaded-managed-package-path>"
    downloadedManagedPackageSha256 = "<owner-downloaded-managed-package-sha256>"
    downloadedRuntimePackagePath = "<owner-downloaded-runtime-package-path>"
    downloadedRuntimePackageSha256 = "<owner-downloaded-runtime-package-sha256>"
    installLogPath = "<owner-install-log-path>"
    installLogSha256 = "<owner-install-log-sha256>"
    restoreLogPath = "<owner-restore-log-path>"
    restoreLogSha256 = "<owner-restore-log-sha256>"
    buildLogPath = "<owner-build-log-path>"
    buildLogSha256 = "<owner-build-log-sha256>"
    runLogPath = "<owner-run-log-path>"
    runLogSha256 = "<owner-run-log-sha256>"
    smokeStdoutPath = "<owner-smoke-stdout-path>"
    smokeStdoutSha256 = "<owner-smoke-stdout-sha256>"
    smokeStderrPath = "<owner-smoke-stderr-path>"
    smokeStderrSha256 = "<owner-smoke-stderr-sha256>"
    nativeAssetListingPath = "<owner-native-asset-listing-path>"
    nativeAssetListingSha256 = "<owner-native-asset-listing-sha256>"
    exitCode = $null
    confirmsPostPublish = $false
    confirmsNotPrePublishSmoke = $false
    hostMetadata = [pscustomobject]@{
      os = "<owner-host-os>"
      arch = "<owner-host-arch>"
      rid = "<owner-host-rid>"
      gpuName = "<owner-gpu-name>"
      nvidiaDriver = "<owner-nvidia-driver>"
      cudaRuntimeToolkit = "<owner-cuda-runtime-toolkit>"
      tensorrt = "<owner-tensorrt-version>"
      cudnn = "<owner-cudnn-version>"
    }
    ownerReviewer = "<owner-reviewer>"
    ownerReviewedAtUtc = "<owner-reviewed-at-utc>"
  }
}

$templatePath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-result.template.json"
$templateMdPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-result.template.md"
$examplePath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-result.example.json"
if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
  $template = New-TemplateRecord
  $template | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $templatePath -Encoding utf8
  Write-Utf8File -LiteralPath $templateMdPath -InputObject @("# Post-Publish CleanConsumer Proof Result Template", "", "Owner must fill this only after public package publication. Template is not proof.")
  $example = New-TemplateRecord
  $example.ownerInputState = "example-not-real-proof"
  $example.publicPackageSourceUrl = "https://api.nuget.org/v3/index.json"
  $example | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $examplePath -Encoding utf8
}

if (-not (Test-Path -LiteralPath $OwnerInputPath -PathType Leaf)) {
  $OwnerInputPath = $templatePath
}

$input = Get-Content -LiteralPath $OwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$findings = New-Object System.Collections.Generic.List[object]
$requiredTextFields = @(
  "publicPackageSourceUrl",
  "publicPackageUrl",
  "managedPackageId",
  "managedPackageVersion",
  "runtimePackageId",
  "runtimePackageVersion",
  "runtimePackageKey",
  "downloadedManagedPackageSha256",
  "downloadedRuntimePackageSha256",
  "installLogPath",
  "installLogSha256",
  "restoreLogPath",
  "restoreLogSha256",
  "buildLogPath",
  "buildLogSha256",
  "runLogPath",
  "runLogSha256",
  "smokeStdoutPath",
  "smokeStdoutSha256",
  "smokeStderrPath",
  "smokeStderrSha256",
  "nativeAssetListingPath",
  "nativeAssetListingSha256",
  "ownerReviewer",
  "ownerReviewedAtUtc"
)

foreach ($field in $requiredTextFields) {
  $value = Get-PropertyOrDefault -Object $input -Name $field -DefaultValue $null
  if (Test-Placeholder -Value $value) {
    $findings.Add((New-Finding "$field-placeholder" "action-required" "placeholder" "Required post-publish field is missing or still a placeholder.")) | Out-Null
  }
}

foreach ($field in @("downloadedManagedPackageSha256", "downloadedRuntimePackageSha256", "installLogSha256", "restoreLogSha256", "buildLogSha256", "runLogSha256", "smokeStdoutSha256", "smokeStderrSha256", "nativeAssetListingSha256")) {
  if (-not (Test-Sha256 -Value (Get-PropertyOrDefault -Object $input -Name $field -DefaultValue ""))) {
    $findings.Add((New-Finding "$field-format" "action-required" "sha256" "SHA256 must be 64 hexadecimal characters.")) | Out-Null
  }
}

foreach ($entry in @{
  downloadedManagedPackagePath = "downloadedManagedPackageSha256"
  downloadedRuntimePackagePath = "downloadedRuntimePackageSha256"
  installLogPath = "installLogSha256"
  restoreLogPath = "restoreLogSha256"
  buildLogPath = "buildLogSha256"
  runLogPath = "runLogSha256"
  smokeStdoutPath = "smokeStdoutSha256"
  smokeStderrPath = "smokeStderrSha256"
  nativeAssetListingPath = "nativeAssetListingSha256"
}.GetEnumerator()) {
  $pathValue = [string](Get-PropertyOrDefault -Object $input -Name $entry.Key -DefaultValue "")
  if (Test-Placeholder -Value $pathValue) { continue }
  $resolved = Resolve-InputPath $pathValue
  if ($RequireExistingFiles.IsPresent -and -not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    $findings.Add((New-Finding "$($entry.Key)-exists" "action-required" "path missing" "Required post-publish evidence file does not exist: $pathValue")) | Out-Null
    continue
  }
  if ($RequireHashMatch.IsPresent -and (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    $expected = [string](Get-PropertyOrDefault -Object $input -Name $entry.Value -DefaultValue "")
    $actual = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash
    if (-not $actual.Equals($expected, [StringComparison]::OrdinalIgnoreCase)) {
      $findings.Add((New-Finding "$($entry.Key)-hash-match" "action-required" "sha256 mismatch" "SHA256 mismatch for $($entry.Key).")) | Out-Null
    }
  }
}

$sourceUrl = [string](Get-PropertyOrDefault -Object $input -Name "publicPackageSourceUrl" -DefaultValue "")
if ($sourceUrl.IndexOf("local", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or $sourceUrl.IndexOf("file:", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
  $findings.Add((New-Finding "public-package-source-local" "blocker" "forbidden substitute" "Public package source appears to be local feed or file source.")) | Out-Null
}

if (-not [bool](Get-PropertyOrDefault -Object $input -Name "confirmsPostPublish" -DefaultValue $false)) {
  $findings.Add((New-Finding "confirms-post-publish" "action-required" "post-publish proof" "Owner has not confirmed this evidence was collected after public publish.")) | Out-Null
}
if (-not [bool](Get-PropertyOrDefault -Object $input -Name "confirmsNotPrePublishSmoke" -DefaultValue $false)) {
  $findings.Add((New-Finding "confirms-not-pre-publish-smoke" "action-required" "forbidden substitute" "Owner has not confirmed pre-publish smoke was not reused.")) | Out-Null
}

$exitCode = Get-PropertyOrDefault -Object $input -Name "exitCode" -DefaultValue $null
if ($null -eq $exitCode -or [int]$exitCode -ne 0) {
  $findings.Add((New-Finding "smoke-exit-code" "action-required" "runtime smoke" "Post-publish smoke exitCode must be 0.")) | Out-Null
}

$hostMetadata = Get-PropertyOrDefault -Object $input -Name "hostMetadata" -DefaultValue ([pscustomobject]@{})
foreach ($field in @("os", "arch", "rid", "gpuName", "nvidiaDriver", "cudaRuntimeToolkit", "tensorrt", "cudnn")) {
  if (Test-Placeholder -Value (Get-PropertyOrDefault -Object $hostMetadata -Name $field -DefaultValue $null)) {
    $findings.Add((New-Finding "host-$field" "action-required" "host metadata" "Host metadata field is missing: $field")) | Out-Null
  }
}

$failedBlockers = @($findings | Where-Object { [string]$_.severity -eq "blocker" })
$failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
$proofReady = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$importState = if ($proofReady) { "post-publish-clean-consumer-proof-result-import-ready" } else { "blocked-post-publish-clean-consumer-proof-result-required" }

$candidate = [pscustomobject]@{
  recordKind = "post-publish-clean-consumer-proof-result-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($proofReady) { "post-publish-clean-consumer-proof-candidate" } else { "blocked-post-publish-clean-consumer-proof-candidate" }
  ownerInputPath = $OwnerInputPath
  proofCandidateReady = $proofReady
  findingCount = $findings.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  publicPackageSourceUrl = $sourceUrl
  exitCode = $exitCode
  ownerActionRequired = -not $proofReady
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $proofReady
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $proofReady
  isPackageConsumerRuntimeProof = $proofReady
  isPostPublishProof = $proofReady
  isReleaseCloseProof = $false
  boundary = "Post-publish CleanConsumer proof candidate only. It cannot publish packages, cannot close the release issue, and cannot substitute Owner final close approval or package push."
}

$import = [pscustomobject]@{
  recordKind = "post-publish-clean-consumer-proof-result-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = $importState
  ownerInputPath = $OwnerInputPath
  candidatePath = "artifacts/final-release/post-publish-clean-consumer-proof-result-candidate.json"
  requireExistingFiles = $RequireExistingFiles.IsPresent
  requireHashMatch = $RequireHashMatch.IsPresent
  failOnNotProof = $FailOnNotProof.IsPresent
  findingCount = $findings.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  proofCandidateReady = $proofReady
  findings = @($findings.ToArray())
  ownerActionRequired = -not $proofReady
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $proofReady
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $proofReady
  isPackageConsumerRuntimeProof = $proofReady
  isPostPublishProof = $proofReady
  isReleaseCloseProof = $false
  boundary = "Post-publish CleanConsumer proof import validates real post-publication owner evidence only. Default/template input remains blocked and non-proof; it is not publish approval, not release close approval, and not package push."
}

$importPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-result-import.json"
$importMdPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-result-import.md"
$candidatePath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-result-candidate.json"
$candidateMdPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-result-candidate.md"
$import | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $importPath -Encoding utf8
$candidate | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $candidatePath -Encoding utf8

$rows = foreach ($finding in $findings) {
  "| ``$(ConvertTo-MarkdownCell $finding.id)`` | ``$(ConvertTo-MarkdownCell $finding.severity)`` | ``$(ConvertTo-MarkdownCell $finding.category)`` | $(ConvertTo-MarkdownCell $finding.message) |"
}

Write-Utf8File -LiteralPath $importMdPath -InputObject @(
  "# Post-Publish CleanConsumer Proof Result Import",
  "",
  "- importState: ``$importState``",
  "- proofCandidateReady: ``$proofReady``",
  "- failedBlockerCount: ``$($failedBlockers.Count)``",
  "- failedActionRequiredCount: ``$($failedActionRequired.Count)``",
  "",
  "| ID | Severity | Category | Message |",
  "|---|---|---|---|",
  @($rows),
  "",
  "## Boundary",
  "",
  $import.boundary
)

Write-Utf8File -LiteralPath $candidateMdPath -InputObject @(
  "# Post-Publish CleanConsumer Proof Result Candidate",
  "",
  "- candidateState: ``$($candidate.candidateState)``",
  "- proofCandidateReady: ``$($candidate.proofCandidateReady)``",
  "- isPostPublishProof: ``$($candidate.isPostPublishProof)``",
  "- canCloseReleaseIssue: ``$($candidate.canCloseReleaseIssue)``",
  "",
  "## Boundary",
  "",
  $candidate.boundary
)

Write-Host "PostPublishCleanConsumerProofResultImportState=$importState ProofCandidateReady=$proofReady FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"
if ($FailOnNotProof.IsPresent -and -not $proofReady) {
  throw "Post-publish CleanConsumer proof result is not proof-ready."
}
