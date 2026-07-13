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

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
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

function Test-PublicHttpsSource {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if (Test-Placeholder -Value $text) { return $false }
  if (-not $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase)) { return $false }
  foreach ($forbidden in @("local", "file:", "artifacts", ".nupkg", "package-managed-dry-run", "github-actions-runs")) {
    if ($text.IndexOf($forbidden, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  }

  return $true
}

function Test-PublicDownloadedPackagePath {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  if (Test-Placeholder -Value $text) { return $false }
  foreach ($forbidden in @("package-managed-dry-run", "github-actions-runs", "\artifacts\", "/artifacts/")) {
    if ($text.IndexOf($forbidden, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  }

  return $text.EndsWith(".nupkg", [StringComparison]::OrdinalIgnoreCase)
}

function Test-CleanConsumerRootOutsideRepository {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  if (Test-Placeholder -Value $text) { return $false }
  foreach ($forbidden in @("ProjectReference", "local-feed", "localfeed", "direct-nupkg", "samples", "smoke", "TensorRtExec")) {
    if ($text.IndexOf($forbidden, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  }

  try {
    $candidateFullPath = [System.IO.Path]::GetFullPath((Resolve-InputPath $text))
    $repositoryFullPath = [System.IO.Path]::GetFullPath($RepositoryRoot)
    return -not $candidateFullPath.StartsWith($repositoryFullPath, [StringComparison]::OrdinalIgnoreCase)
  }
  catch {
    return $false
  }
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

$postPublishCleanConsumerProofRequiredFields = @(
  "publicPackageSourceUrl",
  "publicPackageUrl",
  "publicPackageSourceKind",
  "managedPackageId",
  "managedPackageVersion",
  "runtimePackageId",
  "runtimePackageVersion",
  "runtimePackageKey",
  "downloadedManagedPackagePath",
  "downloadedManagedPackageSha256",
  "downloadedRuntimePackagePath",
  "downloadedRuntimePackageSha256",
  "cleanConsumerRoot",
  "consumerProjectPath",
  "restoreCommand",
  "restoreLogPath",
  "restoreLogSha256",
  "buildCommand",
  "buildLogPath",
  "buildLogSha256",
  "runCommand",
  "runLogPath",
  "runLogSha256",
  "smokeStdoutPath",
  "smokeStdoutSha256",
  "smokeStderrPath",
  "smokeStderrSha256",
  "nativeAssetListingPath",
  "nativeAssetListingSha256",
  "dotnetInfoPath",
  "dotnetInfoSha256",
  "exitCode",
  "hostMetadata.os",
  "hostMetadata.arch",
  "hostMetadata.rid",
  "hostMetadata.gpuName",
  "hostMetadata.nvidiaDriver",
  "hostMetadata.cudaRuntimeToolkit",
  "hostMetadata.tensorrt",
  "hostMetadata.cudnn",
  "sourceGitHubActionsRunEvidenceReady",
  "sourceOwnerPublicPublishResultReady",
  "sourcePublicPackageDownloadProofReady",
  "sourceGitHubActionsRunId",
  "sourceGitHubActionsRunUrl",
  "sourceGitHubActionsHeadSha",
  "sourceOwnerPublicPackageUrl",
  "sourceOwnerPublicPackageVersion",
  "sourceOwnerPublicPackageSha256",
  "sourcePublicDownloadManagedPackageDownloadUrl",
  "sourcePublicDownloadRuntimePackageDownloadUrl",
  "ownerReviewer",
  "ownerReviewedAtUtc"
)

$postPublishCleanConsumerProofRejectedSubstitutes = @(
  "project-reference",
  "local-feed",
  "direct-local-nupkg",
  "repo-internal-consumer",
  "dependency-probe-only",
  "skipped-smoke",
  "blocked-by-cuda-driver",
  "tensorrtexec-report-only",
  "dashboard-only",
  "runbook-only",
  "template-or-candidate-only"
)

$postPublishCleanConsumerProofSourceReadinessSignals = @(
  "sourceGitHubActionsRunEvidenceReady",
  "sourceGitHubActionsRunId",
  "sourceGitHubActionsRunUrl",
  "sourceGitHubActionsHeadSha",
  "sourceOwnerPublicPublishResultReady",
  "sourceOwnerPublicPackageUrl",
  "sourceOwnerPublicPackageVersion",
  "sourceOwnerPublicPackageSha256",
  "sourcePublicPackageDownloadProofReady",
  "sourcePublicDownloadManagedPackageDownloadUrl",
  "sourcePublicDownloadRuntimePackageDownloadUrl"
)

function New-TemplateRecord {
  [pscustomobject]@{
    recordKind = "post-publish-clean-consumer-proof-result-owner-input"
    ownerInputState = "blocked-post-publish-clean-consumer-proof-result-required"
    publicPackageSourceUrl = "<owner-public-package-source-url>"
    publicPackageUrl = "<owner-public-package-url>"
    publicPackageSourceKind = "<owner-nuget.org-or-github-packages>"
    managedPackageId = "JYPPX.TensorRtSharp"
    managedPackageVersion = "<owner-package-version>"
    runtimePackageId = "JYPPX.TensorRtSharp.Native.<runtime-key>"
    runtimePackageVersion = "<owner-package-version>"
    runtimePackageKey = "<owner-runtime-key>"
    sourceGitHubActionsRunId = "<owner-source-github-actions-run-id>"
    sourceGitHubActionsRunUrl = "<owner-source-github-actions-run-url>"
    sourceGitHubActionsHeadSha = "<owner-source-github-actions-head-sha>"
    sourceOwnerPublicPackageUrl = "<owner-source-public-package-url>"
    sourceOwnerPublicPackageVersion = "<owner-source-public-package-version>"
    sourceOwnerPublicPackageSha256 = "<owner-source-public-package-sha256>"
    sourceManagedPackageDownloadUrl = "<owner-source-managed-package-download-url>"
    sourceRuntimePackageDownloadUrl = "<owner-source-runtime-package-download-url>"
    sourceGitHubReleaseAssetUrl = "<owner-source-github-release-asset-url>"
    sourceGitHubReleaseAssetSha256 = "<owner-source-github-release-asset-sha256>"
    downloadedManagedPackagePath = "<owner-downloaded-managed-package-path>"
    downloadedManagedPackageSha256 = "<owner-downloaded-managed-package-sha256>"
    downloadedRuntimePackagePath = "<owner-downloaded-runtime-package-path>"
    downloadedRuntimePackageSha256 = "<owner-downloaded-runtime-package-sha256>"
    cleanConsumerRoot = "<owner-repository-external-clean-consumer-root>"
    consumerProjectPath = "<owner-clean-consumer-csproj-path>"
    restoreCommand = "<owner-restore-command>"
    buildCommand = "<owner-build-command>"
    runCommand = "<owner-run-command>"
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
    dotnetInfoPath = "<owner-dotnet-info-path>"
    dotnetInfoSha256 = "<owner-dotnet-info-sha256>"
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
  "publicPackageSourceKind",
  "managedPackageId",
  "managedPackageVersion",
  "runtimePackageId",
  "runtimePackageVersion",
  "runtimePackageKey",
  "downloadedManagedPackagePath",
  "downloadedManagedPackageSha256",
  "downloadedRuntimePackagePath",
  "downloadedRuntimePackageSha256",
  "cleanConsumerRoot",
  "consumerProjectPath",
  "restoreCommand",
  "buildCommand",
  "runCommand",
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
  "dotnetInfoPath",
  "dotnetInfoSha256",
  "ownerReviewer",
  "ownerReviewedAtUtc"
)

foreach ($field in $requiredTextFields) {
  $value = Get-PropertyOrDefault -Object $input -Name $field -DefaultValue $null
  if (Test-Placeholder -Value $value) {
    $findings.Add((New-Finding "$field-placeholder" "action-required" "placeholder" "Required post-publish field is missing or still a placeholder.")) | Out-Null
  }
}

foreach ($field in @("downloadedManagedPackageSha256", "downloadedRuntimePackageSha256", "installLogSha256", "restoreLogSha256", "buildLogSha256", "runLogSha256", "smokeStdoutSha256", "smokeStderrSha256", "nativeAssetListingSha256", "dotnetInfoSha256")) {
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
  dotnetInfoPath = "dotnetInfoSha256"
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
if (-not (Test-Placeholder -Value $sourceUrl) -and -not (Test-PublicHttpsSource -Value $sourceUrl)) {
  $findings.Add((New-Finding "public-package-source-public-https" "blocker" "forbidden substitute" "Public package source must be HTTPS package source and not local feed, direct nupkg, artifacts path, dry-run artifact, or GitHub Actions artifact.")) | Out-Null
}

$publicPackageUrl = [string](Get-PropertyOrDefault -Object $input -Name "publicPackageUrl" -DefaultValue "")
if (-not (Test-Placeholder -Value $publicPackageUrl) -and -not (Test-PublicHttpsSource -Value $publicPackageUrl)) {
  $findings.Add((New-Finding "public-package-url-public-https" "blocker" "forbidden substitute" "Public package URL must be HTTPS package metadata/source and not local feed, direct nupkg, artifacts path, dry-run artifact, or GitHub Actions artifact.")) | Out-Null
}

foreach ($entry in @{
  downloadedManagedPackagePath = "downloaded managed package"
  downloadedRuntimePackagePath = "downloaded runtime package"
}.GetEnumerator()) {
  $packagePath = [string](Get-PropertyOrDefault -Object $input -Name $entry.Key -DefaultValue "")
  if (-not (Test-Placeholder -Value $packagePath) -and -not (Test-PublicDownloadedPackagePath -Value $packagePath)) {
    $findings.Add((New-Finding "$($entry.Key)-public-download-path" "blocker" "forbidden substitute" "$($entry.Value) path must be a downloaded .nupkg and not artifacts, dry-run, or GitHub Actions path.")) | Out-Null
  }
}

$cleanConsumerRoot = [string](Get-PropertyOrDefault -Object $input -Name "cleanConsumerRoot" -DefaultValue "")
if (-not (Test-Placeholder -Value $cleanConsumerRoot) -and -not (Test-CleanConsumerRootOutsideRepository -Value $cleanConsumerRoot)) {
  $findings.Add((New-Finding "clean-consumer-root-outside-repository" "blocker" "forbidden substitute" "cleanConsumerRoot must be outside the source repository and not a sample/smoke/local substitute path.")) | Out-Null
}

foreach ($field in @("restoreCommand", "buildCommand", "runCommand")) {
  $commandText = [string](Get-PropertyOrDefault -Object $input -Name $field -DefaultValue "")
  foreach ($forbidden in @("ProjectReference", "local feed", "local-feed", "direct .nupkg", "package-managed-dry-run", "github-actions-runs")) {
    if ($commandText.IndexOf($forbidden, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      $findings.Add((New-Finding "$field-forbidden-substitute" "blocker" "forbidden substitute" "$field must not reference forbidden substitute: $forbidden.")) | Out-Null
    }
  }
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

$githubActionsRunEvidence = Read-JsonOrNull "artifacts\final-release\github-actions-run-evidence-import-validation.json"
$ownerPublicPublishResult = Read-JsonOrNull "artifacts\final-release\owner-public-publish-execution-result-candidate-validation.json"
$publicPackageDownloadProof = Read-JsonOrNull "artifacts\final-release\public-package-download-proof-candidate-validation.json"

$sourceGitHubActionsReady = [bool](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "githubActionsRunEvidenceReady" -DefaultValue $false)
$sourceGitHubActionsRunId = [string](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "runId" -DefaultValue "")
$sourceGitHubActionsRunUrl = [string](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "runUrl" -DefaultValue "")
$sourceGitHubActionsHeadSha = [string](Get-PropertyOrDefault -Object $githubActionsRunEvidence -Name "headSha" -DefaultValue "")
$sourceOwnerPublicPublishReady = [bool](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "proofCandidateReady" -DefaultValue $false)
$sourceOwnerPublicPackageUrl = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "publicPackageUrl" -DefaultValue "")
$sourceOwnerPublicPackageVersion = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "publicPackageVersion" -DefaultValue "")
$sourceOwnerPublicPackageSha256 = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "publicPackageSha256" -DefaultValue "")
$sourceOwnerGitHubReleaseAssetUrl = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "githubReleaseAssetUrl" -DefaultValue "")
$sourceOwnerGitHubReleaseAssetSha256 = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResult -Name "githubReleaseAssetSha256" -DefaultValue "")
$sourcePublicDownloadReady = [bool](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "proofCandidateReady" -DefaultValue $false)
$sourcePublicDownloadManagedPackageUrl = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "managedPackagePageUrl" -DefaultValue "")
$sourcePublicDownloadManagedPackageDownloadUrl = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "managedPackageDownloadUrl" -DefaultValue "")
$sourcePublicDownloadRuntimePackageUrl = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "runtimePackagePageUrl" -DefaultValue "")
$sourcePublicDownloadRuntimePackageDownloadUrl = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "runtimePackageDownloadUrl" -DefaultValue "")
$sourcePublicDownloadOwnerPackageUrl = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "sourceOwnerPublicPackageUrl" -DefaultValue "")
$sourcePublicDownloadOwnerPackageVersion = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "sourceOwnerPublicPackageVersion" -DefaultValue "")
$sourcePublicDownloadOwnerPackageSha256 = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "sourceOwnerPublicPackageSha256" -DefaultValue "")
$sourcePublicDownloadGitHubReleaseAssetUrl = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "githubReleaseAssetUrl" -DefaultValue "")
$sourcePublicDownloadGitHubReleaseAssetSha256 = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProof -Name "githubReleaseAssetSha256" -DefaultValue "")

if (-not $sourceGitHubActionsReady) {
  $findings.Add((New-Finding "source-github-actions-run-evidence-ready" "action-required" "source proof" "GitHub Actions run evidence must be ready before post-publish clean consumer proof can close.")) | Out-Null
}
if (-not $sourceOwnerPublicPublishReady) {
  $findings.Add((New-Finding "source-owner-public-publish-result-ready" "action-required" "source proof" "Owner public publish result must be ready before post-publish clean consumer proof can close.")) | Out-Null
}
if (-not $sourcePublicDownloadReady) {
  $findings.Add((New-Finding "source-public-package-download-proof-ready" "action-required" "source proof" "Public package download proof must be ready before post-publish clean consumer proof can close.")) | Out-Null
}

if ($sourceOwnerPublicPublishReady -and -not (Test-Placeholder -Value $publicPackageUrl) -and -not $publicPackageUrl.Equals($sourceOwnerPublicPackageUrl, [StringComparison]::OrdinalIgnoreCase)) {
  $findings.Add((New-Finding "post-publish-owner-public-package-url-match" "blocker" "source proof mismatch" "Post-publish publicPackageUrl must match Owner public publish result publicPackageUrl.")) | Out-Null
}
if ($sourcePublicDownloadReady -and -not (Test-Placeholder -Value $publicPackageUrl) -and -not $publicPackageUrl.Equals($sourcePublicDownloadManagedPackageUrl, [StringComparison]::OrdinalIgnoreCase)) {
  $findings.Add((New-Finding "post-publish-public-download-package-url-match" "blocker" "source proof mismatch" "Post-publish publicPackageUrl must match public package download proof managedPackagePageUrl.")) | Out-Null
}

$managedPackageVersion = [string](Get-PropertyOrDefault -Object $input -Name "managedPackageVersion" -DefaultValue "")
if ($sourceOwnerPublicPublishReady -and -not (Test-Placeholder -Value $managedPackageVersion) -and -not $managedPackageVersion.Equals($sourceOwnerPublicPackageVersion, [StringComparison]::OrdinalIgnoreCase)) {
  $findings.Add((New-Finding "post-publish-owner-public-package-version-match" "blocker" "source proof mismatch" "Post-publish managedPackageVersion must match Owner public publish result publicPackageVersion.")) | Out-Null
}
if ($sourcePublicDownloadReady -and -not (Test-Placeholder -Value $managedPackageVersion) -and -not $managedPackageVersion.Equals($sourcePublicDownloadOwnerPackageVersion, [StringComparison]::OrdinalIgnoreCase)) {
  $findings.Add((New-Finding "post-publish-public-download-package-version-match" "blocker" "source proof mismatch" "Post-publish managedPackageVersion must match public package download proof sourceOwnerPublicPackageVersion.")) | Out-Null
}

$downloadedManagedPackageSha256 = [string](Get-PropertyOrDefault -Object $input -Name "downloadedManagedPackageSha256" -DefaultValue "")
if ($sourceOwnerPublicPublishReady -and (Test-Sha256 -Value $downloadedManagedPackageSha256) -and -not $downloadedManagedPackageSha256.Equals($sourceOwnerPublicPackageSha256, [StringComparison]::OrdinalIgnoreCase)) {
  $findings.Add((New-Finding "post-publish-owner-public-package-sha-match" "blocker" "source proof mismatch" "Post-publish downloadedManagedPackageSha256 must match Owner public publish result publicPackageSha256.")) | Out-Null
}
if ($sourcePublicDownloadReady -and (Test-Sha256 -Value $downloadedManagedPackageSha256) -and -not $downloadedManagedPackageSha256.Equals($sourcePublicDownloadOwnerPackageSha256, [StringComparison]::OrdinalIgnoreCase)) {
  $findings.Add((New-Finding "post-publish-public-download-package-sha-match" "blocker" "source proof mismatch" "Post-publish downloadedManagedPackageSha256 must match public package download proof sourceOwnerPublicPackageSha256.")) | Out-Null
}

$sourceProofLinkageReady = $sourceGitHubActionsReady -and $sourceOwnerPublicPublishReady -and $sourcePublicDownloadReady

$failedBlockers = @($findings | Where-Object { [string]$_.severity -eq "blocker" })
$failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
$blockedRealInputCount = @($findings | Where-Object { [string]$_.severity -eq "blocker" -or [string]$_.severity -eq "action-required" }).Count
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
  postPublishCleanConsumerProofRequiredFields = @($postPublishCleanConsumerProofRequiredFields)
  postPublishCleanConsumerProofRequiredFieldCount = $postPublishCleanConsumerProofRequiredFields.Count
  postPublishCleanConsumerProofRejectedSubstitutes = @($postPublishCleanConsumerProofRejectedSubstitutes)
  postPublishCleanConsumerProofRejectedSubstituteCount = $postPublishCleanConsumerProofRejectedSubstitutes.Count
  postPublishCleanConsumerProofBlockedRealInputCount = $blockedRealInputCount
  postPublishCleanConsumerProofSourceReadinessSignals = @($postPublishCleanConsumerProofSourceReadinessSignals)
  postPublishCleanConsumerProofSourceReadinessSignalCount = $postPublishCleanConsumerProofSourceReadinessSignals.Count
  publicPackageSourceUrl = $sourceUrl
  publicPackageUrl = $publicPackageUrl
  managedPackageVersion = $managedPackageVersion
  downloadedManagedPackageSha256 = $downloadedManagedPackageSha256
  sourceProofLinkageReady = $sourceProofLinkageReady
  sourceGitHubActionsRunEvidenceReady = $sourceGitHubActionsReady
  sourceGitHubActionsRunId = $sourceGitHubActionsRunId
  sourceGitHubActionsRunUrl = $sourceGitHubActionsRunUrl
  sourceGitHubActionsHeadSha = $sourceGitHubActionsHeadSha
  sourceOwnerPublicPublishResultReady = $sourceOwnerPublicPublishReady
  sourceOwnerPublicPackageUrl = $sourceOwnerPublicPackageUrl
  sourceOwnerPublicPackageVersion = $sourceOwnerPublicPackageVersion
  sourceOwnerPublicPackageSha256 = $sourceOwnerPublicPackageSha256
  sourcePublicPackageDownloadProofReady = $sourcePublicDownloadReady
  sourcePublicDownloadManagedPackageUrl = $sourcePublicDownloadManagedPackageUrl
  sourcePublicDownloadManagedPackageDownloadUrl = $sourcePublicDownloadManagedPackageDownloadUrl
  sourcePublicDownloadRuntimePackageUrl = $sourcePublicDownloadRuntimePackageUrl
  sourcePublicDownloadRuntimePackageDownloadUrl = $sourcePublicDownloadRuntimePackageDownloadUrl
  sourceGitHubReleaseAssetUrl = $sourceOwnerGitHubReleaseAssetUrl
  sourceGitHubReleaseAssetSha256 = $sourceOwnerGitHubReleaseAssetSha256
  cleanConsumerRoot = $cleanConsumerRoot
  exitCode = $exitCode
  ownerActionRequired = -not $proofReady
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Post-publish CleanConsumer proof candidate only. proofCandidateReady=true means the owner evidence can satisfy the remote lane, but the import record itself is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
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
  postPublishCleanConsumerProofRequiredFields = @($postPublishCleanConsumerProofRequiredFields)
  postPublishCleanConsumerProofRequiredFieldCount = $postPublishCleanConsumerProofRequiredFields.Count
  postPublishCleanConsumerProofRejectedSubstitutes = @($postPublishCleanConsumerProofRejectedSubstitutes)
  postPublishCleanConsumerProofRejectedSubstituteCount = $postPublishCleanConsumerProofRejectedSubstitutes.Count
  postPublishCleanConsumerProofBlockedRealInputCount = $blockedRealInputCount
  postPublishCleanConsumerProofSourceReadinessSignals = @($postPublishCleanConsumerProofSourceReadinessSignals)
  postPublishCleanConsumerProofSourceReadinessSignalCount = $postPublishCleanConsumerProofSourceReadinessSignals.Count
  proofCandidateReady = $proofReady
  publicPackageUrl = $publicPackageUrl
  managedPackageVersion = $managedPackageVersion
  downloadedManagedPackageSha256 = $downloadedManagedPackageSha256
  sourceProofLinkageReady = $sourceProofLinkageReady
  sourceGitHubActionsRunEvidenceReady = $sourceGitHubActionsReady
  sourceGitHubActionsRunId = $sourceGitHubActionsRunId
  sourceGitHubActionsRunUrl = $sourceGitHubActionsRunUrl
  sourceGitHubActionsHeadSha = $sourceGitHubActionsHeadSha
  sourceOwnerPublicPublishResultReady = $sourceOwnerPublicPublishReady
  sourceOwnerPublicPackageUrl = $sourceOwnerPublicPackageUrl
  sourceOwnerPublicPackageVersion = $sourceOwnerPublicPackageVersion
  sourceOwnerPublicPackageSha256 = $sourceOwnerPublicPackageSha256
  sourcePublicPackageDownloadProofReady = $sourcePublicDownloadReady
  sourcePublicDownloadManagedPackageUrl = $sourcePublicDownloadManagedPackageUrl
  sourcePublicDownloadManagedPackageDownloadUrl = $sourcePublicDownloadManagedPackageDownloadUrl
  sourcePublicDownloadRuntimePackageUrl = $sourcePublicDownloadRuntimePackageUrl
  sourcePublicDownloadRuntimePackageDownloadUrl = $sourcePublicDownloadRuntimePackageDownloadUrl
  sourceGitHubReleaseAssetUrl = $sourceOwnerGitHubReleaseAssetUrl
  sourceGitHubReleaseAssetSha256 = $sourceOwnerGitHubReleaseAssetSha256
  findings = @($findings.ToArray())
  ownerActionRequired = -not $proofReady
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Post-publish CleanConsumer proof import validates real post-publication owner evidence only. proofCandidateReady=true means the owner evidence can satisfy the remote lane, but this local import record remains not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
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
  "- postPublishCleanConsumerProofRequiredFieldCount: ``$($postPublishCleanConsumerProofRequiredFields.Count)``",
  "- postPublishCleanConsumerProofRejectedSubstituteCount: ``$($postPublishCleanConsumerProofRejectedSubstitutes.Count)``",
  "- postPublishCleanConsumerProofBlockedRealInputCount: ``$blockedRealInputCount``",
  "- postPublishCleanConsumerProofSourceReadinessSignalCount: ``$($postPublishCleanConsumerProofSourceReadinessSignals.Count)``",
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
  "- postPublishCleanConsumerProofRequiredFieldCount: ``$($candidate.postPublishCleanConsumerProofRequiredFieldCount)``",
  "- postPublishCleanConsumerProofRejectedSubstituteCount: ``$($candidate.postPublishCleanConsumerProofRejectedSubstituteCount)``",
  "- postPublishCleanConsumerProofBlockedRealInputCount: ``$($candidate.postPublishCleanConsumerProofBlockedRealInputCount)``",
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
