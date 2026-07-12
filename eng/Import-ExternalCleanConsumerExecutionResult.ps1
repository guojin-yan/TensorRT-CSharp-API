[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\external-clean-consumer-execution-result.template.json",
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
  return [string]::IsNullOrWhiteSpace($text) -or $text.StartsWith("<owner-", [StringComparison]::OrdinalIgnoreCase) -or $text.StartsWith("<external-", [StringComparison]::OrdinalIgnoreCase) -or $text.Contains("example-not-real-proof", [StringComparison]::OrdinalIgnoreCase)
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
    recordKind = "external-clean-consumer-execution-result-owner-input"
    ownerInputState = "blocked-external-clean-consumer-execution-result-required"
    repositoryExternalWorkspaceRoot = "<owner-repository-external-workspace-root>"
    cleanConsumerCsprojPath = "<owner-clean-consumer-csproj-path>"
    cleanConsumerCsprojSha256 = "<owner-clean-consumer-csproj-sha256>"
    packageSourceUrl = "<owner-real-package-source-url>"
    managedPackageId = "JYPPX.TensorRtSharp"
    managedPackageVersion = "<owner-package-version>"
    managedPackagePath = "<owner-managed-nupkg-path>"
    managedPackageSha256 = "<owner-managed-nupkg-sha256>"
    runtimePackageId = "JYPPX.TensorRtSharp.Native.<runtime-key>"
    runtimePackageVersion = "<owner-package-version>"
    runtimePackageKey = "<owner-runtime-key>"
    runtimePackagePath = "<owner-runtime-nupkg-path>"
    runtimePackageSha256 = "<owner-runtime-nupkg-sha256>"
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
    stdoutSummary = "<owner-smoke-stdout-summary>"
    stderrSummary = "<owner-smoke-stderr-summary>"
    hostMetadata = [pscustomobject]@{
      os = "<owner-host-os>"
      arch = "<owner-host-arch>"
      rid = "<owner-host-rid>"
      machineName = "<owner-machine-name>"
      dotnetSdk = "<owner-dotnet-sdk>"
      gpuName = "<owner-gpu-name>"
      nvidiaDriver = "<owner-nvidia-driver>"
      cudaRuntimeToolkit = "<owner-cuda-runtime-toolkit>"
      tensorrt = "<owner-tensorrt-version>"
      cudnn = "<owner-cudnn-version>"
    }
    nonSubstituteConfirmations = @(
      [pscustomobject]@{ marker = "local feed"; confirmedAbsent = $false },
      [pscustomobject]@{ marker = "ProjectReference"; confirmedAbsent = $false },
      [pscustomobject]@{ marker = "direct .nupkg"; confirmedAbsent = $false },
      [pscustomobject]@{ marker = "direct nupkg"; confirmedAbsent = $false },
      [pscustomobject]@{ marker = "pre-publish smoke reused as post-publish proof"; confirmedAbsent = $false }
    )
    ownerReviewer = "<owner-reviewer>"
    ownerReviewedAtUtc = "<owner-reviewed-at-utc>"
    readyForStrictValidation = $false
  }
}

$templatePath = Join-Path $OutputRoot "external-clean-consumer-execution-result.template.json"
$templateMdPath = Join-Path $OutputRoot "external-clean-consumer-execution-result.template.md"
$examplePath = Join-Path $OutputRoot "external-clean-consumer-execution-result.example.json"
if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
  $template = New-TemplateRecord
  $template | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $templatePath -Encoding utf8
  Write-Utf8File -LiteralPath $templateMdPath -InputObject @("# External CleanConsumer Execution Result Template", "", "Owner must fill this template with real repository-external logs, hashes, package source, host metadata, and confirmations. Template is not proof.")
  $example = New-TemplateRecord
  $example.ownerInputState = "example-not-real-proof"
  $example.packageSourceUrl = "https://api.nuget.org/v3/index.json"
  $example | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $examplePath -Encoding utf8
}

if (-not (Test-Path -LiteralPath $OwnerInputPath -PathType Leaf)) {
  $OwnerInputPath = $templatePath
}

$input = Get-Content -LiteralPath $OwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$findings = New-Object System.Collections.Generic.List[object]
$requiredTextFields = @(
  "repositoryExternalWorkspaceRoot",
  "cleanConsumerCsprojPath",
  "packageSourceUrl",
  "managedPackageId",
  "managedPackageVersion",
  "managedPackageSha256",
  "runtimePackageId",
  "runtimePackageVersion",
  "runtimePackageKey",
  "runtimePackageSha256",
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
  "stdoutSummary",
  "stderrSummary",
  "ownerReviewer",
  "ownerReviewedAtUtc"
)

foreach ($field in $requiredTextFields) {
  $value = Get-PropertyOrDefault -Object $input -Name $field -DefaultValue $null
  if (Test-Placeholder -Value $value) {
    $findings.Add((New-Finding "$field-placeholder" "action-required" "placeholder" "Required field is missing or still a placeholder.")) | Out-Null
  }
}

$shaFields = @("managedPackageSha256", "runtimePackageSha256", "restoreLogSha256", "buildLogSha256", "runLogSha256", "smokeStdoutSha256", "smokeStderrSha256", "nativeAssetListingSha256", "cleanConsumerCsprojSha256")
foreach ($field in $shaFields) {
  $value = Get-PropertyOrDefault -Object $input -Name $field -DefaultValue ""
  if (-not (Test-Sha256 -Value $value)) {
    $findings.Add((New-Finding "$field-format" "action-required" "sha256" "SHA256 must be 64 hexadecimal characters.")) | Out-Null
  }
}

$repoRootFull = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
$workspaceRoot = [string](Get-PropertyOrDefault -Object $input -Name "repositoryExternalWorkspaceRoot" -DefaultValue "")
if (-not (Test-Placeholder -Value $workspaceRoot)) {
  $workspaceFull = [System.IO.Path]::GetFullPath((Resolve-InputPath $workspaceRoot)).TrimEnd('\', '/')
  if ($workspaceFull.StartsWith($repoRootFull, [StringComparison]::OrdinalIgnoreCase)) {
    $findings.Add((New-Finding "repository-external-workspace" "blocker" "forbidden substitute" "Workspace root resolves inside the repository.")) | Out-Null
  }
  if ($RequireExistingFiles.IsPresent -and -not (Test-Path -LiteralPath $workspaceFull -PathType Container)) {
    $findings.Add((New-Finding "repository-external-workspace-exists" "action-required" "path missing" "Repository-external workspace root does not exist.")) | Out-Null
  }
}

$pathToHash = @{
  cleanConsumerCsprojPath = "cleanConsumerCsprojSha256"
  managedPackagePath = "managedPackageSha256"
  runtimePackagePath = "runtimePackageSha256"
  restoreLogPath = "restoreLogSha256"
  buildLogPath = "buildLogSha256"
  runLogPath = "runLogSha256"
  smokeStdoutPath = "smokeStdoutSha256"
  smokeStderrPath = "smokeStderrSha256"
  nativeAssetListingPath = "nativeAssetListingSha256"
}

foreach ($entry in $pathToHash.GetEnumerator()) {
  $pathValue = [string](Get-PropertyOrDefault -Object $input -Name $entry.Key -DefaultValue "")
  if (Test-Placeholder -Value $pathValue) { continue }
  $resolved = Resolve-InputPath $pathValue
  if ($RequireExistingFiles.IsPresent -and -not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    $findings.Add((New-Finding "$($entry.Key)-exists" "action-required" "path missing" "Required evidence file does not exist: $pathValue")) | Out-Null
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

$packageSourceUrl = [string](Get-PropertyOrDefault -Object $input -Name "packageSourceUrl" -DefaultValue "")
if ($packageSourceUrl.IndexOf("local", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or $packageSourceUrl.IndexOf("file:", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
  $findings.Add((New-Finding "package-source-local-feed" "blocker" "forbidden substitute" "Package source appears to be local feed or file source.")) | Out-Null
}

$csprojPath = [string](Get-PropertyOrDefault -Object $input -Name "cleanConsumerCsprojPath" -DefaultValue "")
if (-not (Test-Placeholder -Value $csprojPath)) {
  $resolvedCsproj = Resolve-InputPath $csprojPath
  if (Test-Path -LiteralPath $resolvedCsproj -PathType Leaf) {
    $csprojText = Get-Content -LiteralPath $resolvedCsproj -Raw -Encoding utf8
    if ($csprojText.IndexOf("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      $findings.Add((New-Finding "csproj-project-reference" "blocker" "forbidden substitute" "CleanConsumer csproj contains ProjectReference.")) | Out-Null
    }
    if ($csprojText.IndexOf(".nupkg", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      $findings.Add((New-Finding "csproj-direct-nupkg" "blocker" "forbidden substitute" "CleanConsumer csproj references direct .nupkg content.")) | Out-Null
    }
  }
}

$exitCode = Get-PropertyOrDefault -Object $input -Name "exitCode" -DefaultValue $null
if ($null -eq $exitCode -or [int]$exitCode -ne 0) {
  $findings.Add((New-Finding "smoke-exit-code" "action-required" "runtime smoke" "Smoke exitCode must be 0.")) | Out-Null
}

$hostMetadata = Get-PropertyOrDefault -Object $input -Name "hostMetadata" -DefaultValue ([pscustomobject]@{})
foreach ($field in @("os", "arch", "rid", "gpuName", "nvidiaDriver", "cudaRuntimeToolkit", "tensorrt", "cudnn")) {
  $value = Get-PropertyOrDefault -Object $hostMetadata -Name $field -DefaultValue $null
  if (Test-Placeholder -Value $value) {
    $findings.Add((New-Finding "host-$field" "action-required" "host metadata" "Host metadata field is missing: $field")) | Out-Null
  }
}

$confirmations = @(Convert-ToArray (Get-PropertyOrDefault -Object $input -Name "nonSubstituteConfirmations" -DefaultValue @()))
foreach ($marker in @("local feed", "ProjectReference", "direct .nupkg", "direct nupkg", "pre-publish smoke reused as post-publish proof")) {
  $matches = @($confirmations | Where-Object { [string](Get-PropertyOrDefault $_ "marker" "") -eq $marker -and [bool](Get-PropertyOrDefault $_ "confirmedAbsent" $false) })
  if ($matches.Count -eq 0) {
    $findings.Add((New-Finding "confirm-$($marker.Replace(' ', '-').Replace('.', 'dot'))" "action-required" "forbidden substitute" "Owner has not confirmed forbidden substitute absent: $marker")) | Out-Null
  }
}

$failedBlockers = @($findings | Where-Object { [string]$_.severity -eq "blocker" })
$failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
$proofReady = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$importState = if ($proofReady) { "external-clean-consumer-execution-result-import-ready" } else { "blocked-external-clean-consumer-execution-result-required" }

$candidate = [pscustomobject]@{
  recordKind = "external-clean-consumer-execution-result-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($proofReady) { "external-clean-consumer-runtime-proof-candidate" } else { "blocked-external-clean-consumer-runtime-proof-candidate" }
  ownerInputPath = $OwnerInputPath
  proofCandidateReady = $proofReady
  findingCount = $findings.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  packageSourceUrl = $packageSourceUrl
  exitCode = $exitCode
  ownerActionRequired = -not $proofReady
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $proofReady
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $proofReady
  isPackageConsumerRuntimeProof = $proofReady
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This External CleanConsumer record is a candidate only and cannot authorize publication, close a release, establish post-publish proof, or push packages."
}

$import = [pscustomobject]@{
  recordKind = "external-clean-consumer-execution-result-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = $importState
  ownerInputPath = $OwnerInputPath
  candidatePath = "artifacts/final-release/external-clean-consumer-execution-result-candidate.json"
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
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "External CleanConsumer execution result import validates real owner evidence only. Missing/default input remains blocked and non-proof; it is not post-publish proof, not publish approval, not release close approval, and not package push."
}

$importPath = Join-Path $OutputRoot "external-clean-consumer-execution-result-import.json"
$importMdPath = Join-Path $OutputRoot "external-clean-consumer-execution-result-import.md"
$candidatePath = Join-Path $OutputRoot "external-clean-consumer-execution-result-candidate.json"
$candidateMdPath = Join-Path $OutputRoot "external-clean-consumer-execution-result-candidate.md"
$import | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $importPath -Encoding utf8
$candidate | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $candidatePath -Encoding utf8

$rows = foreach ($finding in $findings) {
  "| ``$(ConvertTo-MarkdownCell $finding.id)`` | ``$(ConvertTo-MarkdownCell $finding.severity)`` | ``$(ConvertTo-MarkdownCell $finding.category)`` | $(ConvertTo-MarkdownCell $finding.message) |"
}

Write-Utf8File -LiteralPath $importMdPath -InputObject @(
  "# External CleanConsumer Execution Result Import",
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
  "# External CleanConsumer Execution Result Candidate",
  "",
  "- candidateState: ``$($candidate.candidateState)``",
  "- proofCandidateReady: ``$($candidate.proofCandidateReady)``",
  "- canPromoteRuntimeProof: ``$($candidate.canPromoteRuntimeProof)``",
  "- canCloseReleaseIssue: ``$($candidate.canCloseReleaseIssue)``",
  "",
  "## Boundary",
  "",
  $candidate.boundary
)

Write-Host "ExternalCleanConsumerExecutionResultImportState=$importState ProofCandidateReady=$proofReady FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"
if ($FailOnNotProof.IsPresent -and -not $proofReady) {
  throw "External CleanConsumer execution result is not proof-ready."
}
