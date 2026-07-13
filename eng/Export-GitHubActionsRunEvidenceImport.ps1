[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$RunId,
  [string]$RepositoryRoot,
  [string]$ArtifactsRoot,
  [string]$RunMetadataPath,
  [string]$ExpectedHeadSha,
  [string]$WorkflowRunLogPath,
  [string]$ArtifactManifestPath,
  [string]$OwnerReviewer,
  [string]$CapturedAtUtc,
  [switch]$SourceQualityOnly,
  [string]$OutputPath = "artifacts\final-release\github-actions-run-evidence-import.json",
  [string]$MarkdownOutputPath = "artifacts\final-release\github-actions-run-evidence-import.md"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepoPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Write-TextFile {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Value
  )

  $fullPath = Resolve-RepoPath -Path $Path
  $directory = Split-Path -Parent $fullPath
  New-Item -ItemType Directory -Force -Path $directory | Out-Null
  [System.IO.File]::WriteAllText($fullPath, $Value, $utf8)
}

function Read-JsonOrNull {
  param([Parameter(Mandatory = $true)][string]$Path)

  $fullPath = Resolve-RepoPath -Path $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $fullPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [object]$Object,
    [Parameter(Mandatory = $true)][string]$Name,
    [object]$DefaultValue = $null
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }

  return $DefaultValue
}

function Invoke-GitText {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)

  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = "git"
  $psi.WorkingDirectory = $RepositoryRoot
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  foreach ($argument in $Arguments) {
    $psi.ArgumentList.Add($argument)
  }

  $process = [System.Diagnostics.Process]::Start($psi)
  $stdout = $process.StandardOutput.ReadToEnd()
  $stderr = $process.StandardError.ReadToEnd()
  $process.WaitForExit()

  return [pscustomobject]@{
    exitCode = $process.ExitCode
    stdout = $stdout.Trim()
    stderr = $stderr.Trim()
  }
}

function New-Check {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][bool]$Passed,
    [Parameter(Mandatory = $true)][string]$Severity,
    [Parameter(Mandatory = $true)][string]$Detail
  )

  return [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)

  $stream = [System.IO.File]::OpenRead($Path)
  try {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
      return ([System.BitConverter]::ToString($sha.ComputeHash($stream)) -replace "-", "").ToLowerInvariant()
    }
    finally {
      $sha.Dispose()
    }
  }
  finally {
    $stream.Dispose()
  }
}

function Get-NupkgInfo {
  param([Parameter(Mandatory = $true)][System.IO.FileInfo]$Package)

  Add-Type -AssemblyName System.IO.Compression.FileSystem

  $entries = New-Object System.Collections.Generic.List[object]
  $frameworks = New-Object 'System.Collections.Generic.HashSet[string]'
  $archive = [System.IO.Compression.ZipFile]::OpenRead($Package.FullName)
  try {
    foreach ($entry in $archive.Entries) {
      $entries.Add([pscustomobject]@{
        fullName = $entry.FullName
        length = $entry.Length
      }) | Out-Null

      if ($entry.FullName -match '^lib/([^/]+)/') {
        [void]$frameworks.Add($Matches[1])
      }
    }
  }
  finally {
    $archive.Dispose()
  }

  $entryNames = @($entries | ForEach-Object { [string]$_.fullName })
  $dllEntries = @($entryNames | Where-Object { $_ -match '\.dll$' })
  $xmlEntries = @($entryNames | Where-Object { $_ -match '\.xml$' })
  $nativeEntries = @($entryNames | Where-Object { $_ -match '^(runtimes|native)/' })

  return [pscustomobject]@{
    fileName = $Package.Name
    fullPath = $Package.FullName
    length = $Package.Length
    sha256 = Get-Sha256 -Path $Package.FullName
    entryCount = $entryNames.Count
    hasNuspec = @($entryNames | Where-Object { $_ -match '\.nuspec$' }).Count -gt 0
    hasReadme = $entryNames -contains "README.md"
    targetFrameworks = @($frameworks | Sort-Object)
    dllEntryCount = $dllEntries.Count
    xmlEntryCount = $xmlEntries.Count
    nativeEntryCount = $nativeEntries.Count
    sampleEntries = @($entryNames | Select-Object -First 60)
  }
}

function Get-JobConclusion {
  param(
    [object[]]$Jobs,
    [Parameter(Mandatory = $true)][string]$Name
  )

  foreach ($job in $Jobs) {
    $jobName = [string](Get-PropertyOrDefault -Object $job -Name "name" -DefaultValue "")
    if ($jobName.Equals($Name, [StringComparison]::OrdinalIgnoreCase)) {
      return [string](Get-PropertyOrDefault -Object $job -Name "conclusion" -DefaultValue "")
    }
  }

  return ""
}

function Select-FirstExistingPath {
  param([Parameter(Mandatory = $true)][string[]]$Candidates)

  foreach ($candidate in $Candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
      return $candidate
    }
  }

  return $Candidates[0]
}

if ([string]::IsNullOrWhiteSpace($ArtifactsRoot)) {
  $ArtifactsRoot = Join-Path "artifacts\github-actions-runs" $RunId
}

$artifactsRootFullPath = Resolve-RepoPath -Path $ArtifactsRoot
$releaseQualitySummaryPath = Select-FirstExistingPath -Candidates @(
  (Join-Path $artifactsRootFullPath "release-quality-gate\release-quality-gate\release-quality-gate-summary.json"),
  (Join-Path $artifactsRootFullPath "release-quality-gate\release-quality-gate-summary.json")
)
$packageValidationAuditPath = Select-FirstExistingPath -Candidates @(
  (Join-Path $artifactsRootFullPath "release-quality-gate\final-release\github-actions-package-validation-audit.json"),
  (Join-Path $artifactsRootFullPath "final-release\github-actions-package-validation-audit.json")
)
$packageRoot = Join-Path $artifactsRootFullPath "package-managed-dry-run"

if ([string]::IsNullOrWhiteSpace($RunMetadataPath)) {
  foreach ($candidate in @(
      (Join-Path $artifactsRootFullPath "github-run-view.json"),
      (Join-Path $artifactsRootFullPath "run-metadata.json"),
      (Join-Path $artifactsRootFullPath "run.json")
    )) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
      $RunMetadataPath = $candidate
      break
    }
  }
}

$releaseQualitySummary = Read-JsonOrNull -Path $releaseQualitySummaryPath
$packageValidationAudit = Read-JsonOrNull -Path $packageValidationAuditPath
$runMetadata = if ([string]::IsNullOrWhiteSpace($RunMetadataPath)) { $null } else { Read-JsonOrNull -Path $RunMetadataPath }

$nupkgs = if (Test-Path -LiteralPath $packageRoot -PathType Container) {
  @(Get-ChildItem -LiteralPath $packageRoot -Filter *.nupkg -File)
}
else {
  @()
}
$nupkgInfos = @($nupkgs | ForEach-Object { Get-NupkgInfo -Package $_ })

$metadataJobs = @()
if ($null -ne $runMetadata) {
  $metadataJobs = @(Get-PropertyOrDefault -Object $runMetadata -Name "jobs" -DefaultValue @())
}

$runHeadSha = [string](Get-PropertyOrDefault -Object $runMetadata -Name "headSha" -DefaultValue "")
if ([string]::IsNullOrWhiteSpace($runHeadSha)) {
  $runHeadSha = [string](Get-PropertyOrDefault -Object $packageValidationAudit -Name "headSha" -DefaultValue "")
}

$runConclusion = [string](Get-PropertyOrDefault -Object $runMetadata -Name "conclusion" -DefaultValue "")
$runStatus = [string](Get-PropertyOrDefault -Object $runMetadata -Name "status" -DefaultValue "")
$runUrl = [string](Get-PropertyOrDefault -Object $runMetadata -Name "url" -DefaultValue "")
$runAttempt = [string](Get-PropertyOrDefault -Object $runMetadata -Name "runAttempt" -DefaultValue (Get-PropertyOrDefault -Object $runMetadata -Name "attempt" -DefaultValue ""))
$workflowName = [string](Get-PropertyOrDefault -Object $runMetadata -Name "workflowName" -DefaultValue (Get-PropertyOrDefault -Object $runMetadata -Name "name" -DefaultValue ""))
$workflowFile = [string](Get-PropertyOrDefault -Object $runMetadata -Name "workflowFile" -DefaultValue (Get-PropertyOrDefault -Object $runMetadata -Name "workflowPath" -DefaultValue ""))
$runEvent = [string](Get-PropertyOrDefault -Object $runMetadata -Name "event" -DefaultValue (Get-PropertyOrDefault -Object $runMetadata -Name "eventName" -DefaultValue ""))
$runBranch = [string](Get-PropertyOrDefault -Object $runMetadata -Name "headBranch" -DefaultValue (Get-PropertyOrDefault -Object $runMetadata -Name "branch" -DefaultValue ""))
$runRef = [string](Get-PropertyOrDefault -Object $runMetadata -Name "ref" -DefaultValue (Get-PropertyOrDefault -Object $runMetadata -Name "headRefName" -DefaultValue ""))
$startedAtUtc = [string](Get-PropertyOrDefault -Object $runMetadata -Name "startedAtUtc" -DefaultValue (Get-PropertyOrDefault -Object $runMetadata -Name "createdAt" -DefaultValue ""))
$completedAtUtc = [string](Get-PropertyOrDefault -Object $runMetadata -Name "completedAtUtc" -DefaultValue (Get-PropertyOrDefault -Object $runMetadata -Name "updatedAt" -DefaultValue ""))

$workflowRunLogFullPath = if ([string]::IsNullOrWhiteSpace($WorkflowRunLogPath)) { "" } else { Resolve-RepoPath -Path $WorkflowRunLogPath }
$artifactManifestFullPath = if ([string]::IsNullOrWhiteSpace($ArtifactManifestPath)) { "" } else { Resolve-RepoPath -Path $ArtifactManifestPath }
$workflowRunLogSha256 = if (-not [string]::IsNullOrWhiteSpace($workflowRunLogFullPath) -and (Test-Path -LiteralPath $workflowRunLogFullPath -PathType Leaf)) { Get-Sha256 -Path $workflowRunLogFullPath } else { [string](Get-PropertyOrDefault -Object $runMetadata -Name "workflowRunLogSha256" -DefaultValue "") }
$artifactManifestSha256 = if (-not [string]::IsNullOrWhiteSpace($artifactManifestFullPath) -and (Test-Path -LiteralPath $artifactManifestFullPath -PathType Leaf)) { Get-Sha256 -Path $artifactManifestFullPath } else { [string](Get-PropertyOrDefault -Object $runMetadata -Name "artifactManifestSha256" -DefaultValue "") }

if ([string]::IsNullOrWhiteSpace($OwnerReviewer)) {
  $OwnerReviewer = [string](Get-PropertyOrDefault -Object $runMetadata -Name "ownerReviewer" -DefaultValue "")
}
if ([string]::IsNullOrWhiteSpace($CapturedAtUtc)) {
  $CapturedAtUtc = [string](Get-PropertyOrDefault -Object $runMetadata -Name "capturedAtUtc" -DefaultValue "")
}

$sourceQualityConclusion = Get-JobConclusion -Jobs $metadataJobs -Name "source-quality"
$packagePackConclusion = Get-JobConclusion -Jobs $metadataJobs -Name "package-managed-dry-run / pack"
$publishNugetConclusion = Get-JobConclusion -Jobs $metadataJobs -Name "package-managed-dry-run / publish-nuget"
$publishGitHubPackagesConclusion = Get-JobConclusion -Jobs $metadataJobs -Name "package-managed-dry-run / publish-github-packages"

$head = Invoke-GitText -Arguments @("rev-parse", "HEAD")
$upstream = Invoke-GitText -Arguments @("rev-parse", "@{u}")
$currentHead = if ($head.exitCode -eq 0) { $head.stdout } else { "" }
$upstreamHead = if ($upstream.exitCode -eq 0) { $upstream.stdout } else { "" }

$expectedHeadReady = [string]::IsNullOrWhiteSpace($ExpectedHeadSha) -or $runHeadSha.Equals($ExpectedHeadSha, [StringComparison]::OrdinalIgnoreCase)
$runMatchesCurrentHead = -not [string]::IsNullOrWhiteSpace($runHeadSha) -and $runHeadSha.Equals($currentHead, [StringComparison]::OrdinalIgnoreCase)
$runMatchesUpstreamHead = -not [string]::IsNullOrWhiteSpace($runHeadSha) -and $runHeadSha.Equals($upstreamHead, [StringComparison]::OrdinalIgnoreCase)
$releaseQualityPassed = $null -ne $releaseQualitySummary -and ([string](Get-PropertyOrDefault -Object $releaseQualitySummary -Name "state" -DefaultValue "")).Equals("release-quality-gate-passed", [StringComparison]::OrdinalIgnoreCase)
$packageAuditPresent = $null -ne $packageValidationAudit
$packageContentReadable = $nupkgInfos.Count -gt 0 -and @($nupkgInfos | Where-Object { -not $_.hasNuspec -or $_.dllEntryCount -le 0 -or $_.xmlEntryCount -le 0 }).Count -eq 0
$sourceQualitySucceeded = $sourceQualityConclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)
$packagePackSucceeded = $packagePackConclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)
$publishNugetSkipped = $publishNugetConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase)
$publishGitHubPackagesSkipped = $publishGitHubPackagesConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase)
$runSucceeded = $runConclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)
$sourceQualityOnlyMode = $SourceQualityOnly.IsPresent
$packageEvidenceRequired = -not $sourceQualityOnlyMode
$packagePackSafeForSelectedMode = if ($sourceQualityOnlyMode) {
  [string]::IsNullOrWhiteSpace($packagePackConclusion) -or
  $packagePackConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase)
}
else {
  $packagePackSucceeded
}
$publishJobsSafeForSelectedMode = if ($sourceQualityOnlyMode) {
  (
    [string]::IsNullOrWhiteSpace($publishNugetConclusion) -or
    $publishNugetConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase)
  ) -and (
    [string]::IsNullOrWhiteSpace($publishGitHubPackagesConclusion) -or
    $publishGitHubPackagesConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase)
  )
}
else {
  $publishNugetSkipped -and $publishGitHubPackagesSkipped
}

$checks = @(
  New-Check -Id "run-metadata-present" -Passed ($null -ne $runMetadata) -Severity "blocker" -Detail $(if ($null -ne $runMetadata) { $RunMetadataPath } else { "Run metadata JSON is required to import job conclusions without network access." })
  New-Check -Id "release-quality-summary-present" -Passed ($null -ne $releaseQualitySummary) -Severity "blocker" -Detail $releaseQualitySummaryPath
  New-Check -Id "release-quality-summary-passed" -Passed $releaseQualityPassed -Severity "blocker" -Detail "release-quality-gate-summary state must be release-quality-gate-passed."
  New-Check -Id "package-validation-audit-present" -Passed $packageAuditPresent -Severity "blocker" -Detail $packageValidationAuditPath
  New-Check -Id "run-head-sha-matches-expected" -Passed $expectedHeadReady -Severity "blocker" -Detail "runHeadSha=$runHeadSha; expected=$ExpectedHeadSha"
  New-Check -Id "workflow-run-success" -Passed $runSucceeded -Severity "blocker" -Detail "runConclusion=$runConclusion"
  New-Check -Id "source-quality-job-success" -Passed $sourceQualitySucceeded -Severity "blocker" -Detail "source-quality=$sourceQualityConclusion"
  New-Check -Id "package-managed-dry-run-pack-success" -Passed $packagePackSafeForSelectedMode -Severity $(if ($packageEvidenceRequired) { "blocker" } else { "info" }) -Detail "package-managed-dry-run / pack=$packagePackConclusion; sourceQualityOnly=$sourceQualityOnlyMode"
  New-Check -Id "publish-nuget-skipped" -Passed $publishJobsSafeForSelectedMode -Severity $(if ($packageEvidenceRequired) { "blocker" } else { "info" }) -Detail "package-managed-dry-run / publish-nuget=$publishNugetConclusion; sourceQualityOnly=$sourceQualityOnlyMode"
  New-Check -Id "publish-github-packages-skipped" -Passed $publishJobsSafeForSelectedMode -Severity $(if ($packageEvidenceRequired) { "blocker" } else { "info" }) -Detail "package-managed-dry-run / publish-github-packages=$publishGitHubPackagesConclusion; sourceQualityOnly=$sourceQualityOnlyMode"
  New-Check -Id "managed-nupkg-present" -Passed (($nupkgInfos.Count -gt 0) -or $sourceQualityOnlyMode) -Severity $(if ($packageEvidenceRequired) { "blocker" } else { "info" }) -Detail "nupkgCount=$($nupkgInfos.Count); sourceQualityOnly=$sourceQualityOnlyMode"
  New-Check -Id "managed-nupkg-content-readable" -Passed ($packageContentReadable -or $sourceQualityOnlyMode) -Severity $(if ($packageEvidenceRequired) { "blocker" } else { "info" }) -Detail "nuspec/dll/xml entries must be readable from the package artifact when package dry-run evidence is imported."
)

$failedBlockerCount = @($checks | Where-Object { -not $_.passed -and $_.severity -eq "blocker" }).Count
$canClaimPackageDryRun = $failedBlockerCount -eq 0 -and $runSucceeded -and $sourceQualitySucceeded -and $packagePackSucceeded -and $publishNugetSkipped -and $publishGitHubPackagesSkipped
$canClaimSourceQualityRun = $failedBlockerCount -eq 0 -and $runSucceeded -and $sourceQualitySucceeded -and $releaseQualityPassed -and $packagePackSafeForSelectedMode -and $publishJobsSafeForSelectedMode
$evidenceState = if ($canClaimPackageDryRun) {
  "github-actions-run-evidence-ready"
}
elseif ($canClaimSourceQualityRun) {
  "source-quality-run-evidence-ready"
}
else {
  "blocked-github-actions-run-evidence-required"
}
$proofBoundary = if ($sourceQualityOnlyMode) {
  "This imported evidence proves only that the specified GitHub Actions run completed source-quality when all source checks pass. It does not prove package-managed dry-run pack, does not publish NuGet, does not publish GitHub Packages, does not run dotnet nuget push, is not compatible-host runtime proof, is not package-consumer runtime proof, and is not post-publish proof."
}
else {
  "This imported evidence proves only that the specified GitHub Actions run completed source-quality and package-managed dry-run pack when all checks pass. It does not publish NuGet, does not publish GitHub Packages, does not run dotnet nuget push, is not compatible-host runtime proof, is not package-consumer runtime proof, and is not post-publish proof."
}

$record = [pscustomobject]@{
  recordKind = "github-actions-run-evidence-import"
  generatedAt = (Get-Date).ToString("o")
  evidenceState = $evidenceState
  importMode = if ($sourceQualityOnlyMode) { "source-quality-only" } else { "package-dry-run" }
  repositoryRoot = $RepositoryRoot
  artifactsRoot = $artifactsRootFullPath
  runId = $RunId
  runUrl = $runUrl
  runStatus = $runStatus
  runConclusion = $runConclusion
  runAttempt = $runAttempt
  workflowName = $workflowName
  workflowFile = $workflowFile
  runEvent = $runEvent
  runBranch = $runBranch
  runRef = $runRef
  startedAtUtc = $startedAtUtc
  completedAtUtc = $completedAtUtc
  headSha = $runHeadSha
  expectedHeadSha = $ExpectedHeadSha
  currentHead = $currentHead
  upstreamHead = $upstreamHead
  runHeadMatchesCurrentHead = $runMatchesCurrentHead
  runHeadMatchesUpstreamHead = $runMatchesUpstreamHead
  sourceQualityConclusion = $sourceQualityConclusion
  packageManagedDryRunPackConclusion = $packagePackConclusion
  publishNugetConclusion = $publishNugetConclusion
  publishGitHubPackagesConclusion = $publishGitHubPackagesConclusion
  releaseQualitySummaryPath = $releaseQualitySummaryPath
  packageValidationAuditPath = $packageValidationAuditPath
  workflowRunLogPath = $workflowRunLogFullPath
  workflowRunLogSha256 = $workflowRunLogSha256
  artifactManifestPath = $artifactManifestFullPath
  artifactManifestSha256 = $artifactManifestSha256
  ownerReviewer = $OwnerReviewer
  capturedAtUtc = $CapturedAtUtc
  nupkgPackages = $nupkgInfos
  failedBlockerCount = $failedBlockerCount
  canClaimGitHubActionsSourceQualityForRun = $canClaimSourceQualityRun
  canClaimGitHubActionsPackageDryRunPackForRun = $canClaimPackageDryRun
  canClaimNuGetPublished = $false
  canClaimGitHubPackagesPublished = $false
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  proofBoundary = $proofBoundary
  checks = $checks
}

Write-TextFile -Path $OutputPath -Value ($record | ConvertTo-Json -Depth 10)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# GitHub Actions Run Evidence Import")
$lines.Add("")
$lines.Add("- Run ID: ``$RunId``")
$lines.Add("- Run URL: $runUrl")
$lines.Add("- Import mode: ``$($record.importMode)``")
$lines.Add("- Evidence state: ``$evidenceState``")
$lines.Add("- Head SHA: ``$runHeadSha``")
$lines.Add("- Run conclusion: ``$runConclusion``")
$lines.Add("- Run attempt: ``$runAttempt``")
$lines.Add("- Workflow: ``$workflowName``")
$lines.Add("- Workflow file: ``$workflowFile``")
$lines.Add("- Event: ``$runEvent``")
$lines.Add("- Ref: ``$runRef``")
$lines.Add("- Started at UTC: ``$startedAtUtc``")
$lines.Add("- Completed at UTC: ``$completedAtUtc``")
$lines.Add("- Source quality: ``$sourceQualityConclusion``")
$lines.Add("- Package dry-run pack: ``$packagePackConclusion``")
$lines.Add("- publish-nuget: ``$publishNugetConclusion``")
$lines.Add("- publish-github-packages: ``$publishGitHubPackagesConclusion``")
$lines.Add("- Can claim source-quality run: ``$canClaimSourceQualityRun``")
$lines.Add("- Can claim package dry-run pack: ``$canClaimPackageDryRun``")
$lines.Add("- Can claim NuGet published: ``False``")
$lines.Add("- Can claim GitHub Packages published: ``False``")
$lines.Add("- Is package-consumer runtime proof: ``False``")
$lines.Add("- Workflow run log SHA256: ``$workflowRunLogSha256``")
$lines.Add("- Artifact manifest SHA256: ``$artifactManifestSha256``")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.proofBoundary)
$lines.Add("")
$lines.Add("## Packages")
$lines.Add("")
$lines.Add("| File | SHA256 | Size | TFMs | DLLs | XML docs | Native entries |")
$lines.Add("| --- | --- | ---: | --- | ---: | ---: | ---: |")
foreach ($package in $nupkgInfos) {
  $lines.Add("| ``$($package.fileName)`` | ``$($package.sha256)`` | $($package.length) | ``$(($package.targetFrameworks -join ', '))`` | $($package.dllEntryCount) | $($package.xmlEntryCount) | $($package.nativeEntryCount) |")
}
if ($nupkgInfos.Count -eq 0) {
  $lines.Add("| _missing_ |  | 0 |  | 0 | 0 | 0 |")
}
$lines.Add("")
$lines.Add("## Checks")
$lines.Add("")
$lines.Add("| Check | Passed | Severity | Detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($check in $checks) {
  $lines.Add("| ``$($check.id)`` | ``$($check.passed)`` | ``$($check.severity)`` | $($check.detail) |")
}

Write-TextFile -Path $MarkdownOutputPath -Value ($lines -join "`r`n")

Write-Host "GitHub Actions run evidence import written to $(Resolve-RepoPath -Path $OutputPath)"
Write-Host "GitHub Actions run evidence import written to $(Resolve-RepoPath -Path $MarkdownOutputPath)"
