[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\github-actions-run-evidence-import.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-Sha256Format {
  param([string]$Value)
  return -not [string]::IsNullOrWhiteSpace($Value) -and $Value -match "^[0-9a-fA-F]{64}$"
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

function Test-FileHashMatches {
  param([string]$Path, [string]$Sha256)

  if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Sha256Format -Value $Sha256)) {
    return $false
  }

  $resolvedPath = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $false
  }

  return (Get-Sha256 -Path $resolvedPath).Equals($Sha256, [StringComparison]::OrdinalIgnoreCase)
}

function Test-DateTimeOffsetFormat {
  param([string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $false
  }

  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(
    $Value,
    [System.Globalization.CultureInfo]::InvariantCulture,
    [System.Globalization.DateTimeStyles]::AssumeUniversal,
    [ref]$parsed)
}

function Test-CompletedAfterStarted {
  param([string]$StartedAtUtc, [string]$CompletedAtUtc)

  $started = [DateTimeOffset]::MinValue
  $completed = [DateTimeOffset]::MinValue
  $startedOk = [DateTimeOffset]::TryParse($StartedAtUtc, [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::AssumeUniversal, [ref]$started)
  $completedOk = [DateTimeOffset]::TryParse($CompletedAtUtc, [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::AssumeUniversal, [ref]$completed)
  return $startedOk -and $completedOk -and $completed -ge $started
}

function ConvertTo-IsoDateTimeOffsetString {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  if ($Value -is [DateTimeOffset]) {
    return $Value.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ", [System.Globalization.CultureInfo]::InvariantCulture)
  }

  if ($Value -is [DateTime]) {
    return ([DateTimeOffset]$Value).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ", [System.Globalization.CultureInfo]::InvariantCulture)
  }

  $text = [string]$Value
  if ([string]::IsNullOrWhiteSpace($text)) {
    return ""
  }

  $parsed = [DateTimeOffset]::MinValue
  if ([DateTimeOffset]::TryParse($text, [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::AssumeUniversal, [ref]$parsed)) {
    return $parsed.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ", [System.Globalization.CultureInfo]::InvariantCulture)
  }

  return $text
}

function Test-ConcreteValue {
  param([string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $false
  }

  $text = $Value.Trim()
  if ($text -match '^<.*>$') {
    return $false
  }

  return $text -notmatch '(?i)\b(placeholder|todo|tbd|sample|example|dummy|fake)\b'
}

function Test-GitHubActionsRunUrl {
  param([string]$Value)

  return -not [string]::IsNullOrWhiteSpace($Value) -and
    $Value.StartsWith("https://github.com/", [StringComparison]::OrdinalIgnoreCase) -and
    $Value -match '(?i)/actions/runs/[0-9]+'
}

$githubActionsRunProofRequiredFields = @(
  "githubRunId",
  "githubRunUrl",
  "workflowName",
  "headSha",
  "headBranch",
  "status",
  "conclusion",
  "createdAtUtc",
  "updatedAtUtc",
  "logSha256",
  "artifactSha256",
  "runnerOs",
  "ownerReviewer"
)

$githubActionsRunProofRejectedStates = @(
  "queued",
  "waiting",
  "requested",
  "pending",
  "in_progress",
  "cancelled",
  "failure",
  "timed_out",
  "dashboard-only",
  "local-build-only",
  "local-test-only",
  "dry-run-only"
)

$githubActionsRunProofRejectedSubstitutes = @(
  "queued workflow",
  "dashboard-only",
  "local dotnet test",
  "local build",
  "package-managed-dry-run-only",
  "local feed",
  "direct nupkg",
  "ProjectReference",
  "manual approval",
  "missing runner"
)

function Get-ForbiddenSubstituteFindings {
  param(
    [AllowNull()][object]$Record,
    [string]$RunUrl,
    [string]$RunStatus,
    [string]$RunConclusion,
    [string]$WorkflowRunLogPath,
    [string]$ArtifactManifestPath,
    [string]$OwnerReviewer
  )

  $findings = New-Object System.Collections.Generic.List[string]
  if ($null -eq $Record) {
    return @()
  }

  if (-not [string]::IsNullOrWhiteSpace($RunStatus) -and $RunStatus -match '(?i)\b(queued|waiting|requested|pending)\b') {
    $findings.Add("queued-or-pending-workflow-status:$RunStatus") | Out-Null
  }

  if (-not [string]::IsNullOrWhiteSpace($RunUrl) -and $RunUrl.StartsWith("https://github.com/", [StringComparison]::OrdinalIgnoreCase) -and -not (Test-GitHubActionsRunUrl -Value $RunUrl)) {
    $findings.Add("dashboard-or-non-run-url:$RunUrl") | Out-Null
  }

  foreach ($pathPair in @(
      @{ name = "workflowRunLogPath"; value = $WorkflowRunLogPath },
      @{ name = "artifactManifestPath"; value = $ArtifactManifestPath }
    )) {
    $pathValue = [string]$pathPair.value
    if ([string]::IsNullOrWhiteSpace($pathValue)) {
      continue
    }

    $normalized = $pathValue.Replace("\", "/")
    if ($normalized -match '(?i)/package-managed-dry-run(/|$)') {
      $findings.Add("$($pathPair.name)-points-to-package-managed-dry-run") | Out-Null
    }

    if ($normalized -match '(?i)\.nupkg$') {
      $findings.Add("$($pathPair.name)-points-to-direct-nupkg") | Out-Null
    }
  }

  $selectedText = @(
    $RunUrl,
    $RunStatus,
    $RunConclusion,
    $WorkflowRunLogPath,
    $ArtifactManifestPath,
    $OwnerReviewer
  ) -join "`n"

  foreach ($pattern in @(
      @{ id = "local-dotnet-test"; regex = '(?i)local\s+dotnet\s+test' },
      @{ id = "missing-runner"; regex = '(?i)missing\s+runner' },
      @{ id = "manual-approval"; regex = '(?i)manual\s+approval' },
      @{ id = "local-feed"; regex = '(?i)local\s+feed|local-feed' },
      @{ id = "project-reference"; regex = '(?i)projectreference|project\s+reference' },
      @{ id = "package-managed-dry-run-only"; regex = '(?i)package-managed-dry-run[-\s]+only' },
      @{ id = "direct-nupkg"; regex = '(?i)direct\s+\.?nupkg|direct-nupkg' }
    )) {
    if ($selectedText -match $pattern.regex) {
      $findings.Add([string]$pattern.id) | Out-Null
    }
  }

  return @($findings.ToArray() | Select-Object -Unique)
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
$record = $null
if (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf) {
  $record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$runId = [string](Get-PropertyOrDefault -Object $record -Name "runId" -DefaultValue "")
$runUrl = [string](Get-PropertyOrDefault -Object $record -Name "runUrl" -DefaultValue "")
$runStatus = [string](Get-PropertyOrDefault -Object $record -Name "runStatus" -DefaultValue "")
$runConclusion = [string](Get-PropertyOrDefault -Object $record -Name "runConclusion" -DefaultValue "")
$runAttempt = [string](Get-PropertyOrDefault -Object $record -Name "runAttempt" -DefaultValue "")
$workflowName = [string](Get-PropertyOrDefault -Object $record -Name "workflowName" -DefaultValue "")
$workflowFile = [string](Get-PropertyOrDefault -Object $record -Name "workflowFile" -DefaultValue "")
$runEvent = [string](Get-PropertyOrDefault -Object $record -Name "runEvent" -DefaultValue "")
$runBranch = [string](Get-PropertyOrDefault -Object $record -Name "runBranch" -DefaultValue "")
$runRef = [string](Get-PropertyOrDefault -Object $record -Name "runRef" -DefaultValue "")
$runnerOs = [string](Get-PropertyOrDefault -Object $record -Name "runnerOs" -DefaultValue "")
$startedAtUtc = ConvertTo-IsoDateTimeOffsetString -Value (Get-PropertyOrDefault -Object $record -Name "startedAtUtc" -DefaultValue "")
$completedAtUtc = ConvertTo-IsoDateTimeOffsetString -Value (Get-PropertyOrDefault -Object $record -Name "completedAtUtc" -DefaultValue "")
$headSha = [string](Get-PropertyOrDefault -Object $record -Name "headSha" -DefaultValue "")
$expectedHeadSha = [string](Get-PropertyOrDefault -Object $record -Name "expectedHeadSha" -DefaultValue "")
$runHeadMatchesCurrentHead = [bool](Get-PropertyOrDefault -Object $record -Name "runHeadMatchesCurrentHead" -DefaultValue $false)
$runHeadMatchesUpstreamHead = [bool](Get-PropertyOrDefault -Object $record -Name "runHeadMatchesUpstreamHead" -DefaultValue $false)
$sourceQualityConclusion = [string](Get-PropertyOrDefault -Object $record -Name "sourceQualityConclusion" -DefaultValue "")
$packagePackConclusion = [string](Get-PropertyOrDefault -Object $record -Name "packageManagedDryRunPackConclusion" -DefaultValue "")
$publishNugetConclusion = [string](Get-PropertyOrDefault -Object $record -Name "publishNugetConclusion" -DefaultValue "")
$publishGitHubPackagesConclusion = [string](Get-PropertyOrDefault -Object $record -Name "publishGitHubPackagesConclusion" -DefaultValue "")
$nupkgPackages = @((Get-PropertyOrDefault -Object $record -Name "nupkgPackages" -DefaultValue @()))
$importFailedBlockerCount = [int](Get-PropertyOrDefault -Object $record -Name "failedBlockerCount" -DefaultValue 999)
$canClaimDryRunPack = [bool](Get-PropertyOrDefault -Object $record -Name "canClaimGitHubActionsPackageDryRunPackForRun" -DefaultValue $false)
$workflowRunLogPath = [string](Get-PropertyOrDefault -Object $record -Name "workflowRunLogPath" -DefaultValue "")
$workflowRunLogSha256 = [string](Get-PropertyOrDefault -Object $record -Name "workflowRunLogSha256" -DefaultValue "")
$artifactManifestPath = [string](Get-PropertyOrDefault -Object $record -Name "artifactManifestPath" -DefaultValue "")
$artifactManifestSha256 = [string](Get-PropertyOrDefault -Object $record -Name "artifactManifestSha256" -DefaultValue "")
$ownerReviewer = [string](Get-PropertyOrDefault -Object $record -Name "ownerReviewer" -DefaultValue "")
$capturedAtUtc = ConvertTo-IsoDateTimeOffsetString -Value (Get-PropertyOrDefault -Object $record -Name "capturedAtUtc" -DefaultValue "")
$importMode = [string](Get-PropertyOrDefault -Object $record -Name "importMode" -DefaultValue "package-dry-run")
$evidenceState = [string](Get-PropertyOrDefault -Object $record -Name "evidenceState" -DefaultValue "")
$canClaimSourceQuality = [bool](Get-PropertyOrDefault -Object $record -Name "canClaimGitHubActionsSourceQualityForRun" -DefaultValue $false)
$sourceQualityOnlyMode = $importMode.Equals("source-quality-only", [StringComparison]::OrdinalIgnoreCase)
$packageEvidenceRequired = -not $sourceQualityOnlyMode

$runAttemptIsPositiveInteger = $runAttempt -match '^[0-9]+$' -and [int64]$runAttempt -gt 0
$runHeadLinksSource = $headSha -match '^[0-9a-fA-F]{40}$' -and (
  $runHeadMatchesCurrentHead -or
  $runHeadMatchesUpstreamHead -or
  (-not [string]::IsNullOrWhiteSpace($expectedHeadSha) -and $headSha.Equals($expectedHeadSha, [StringComparison]::OrdinalIgnoreCase))
)
$nupkgSha256s = @($nupkgPackages | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "sha256" -DefaultValue "") })
$packageSha256sPresent = $nupkgSha256s.Count -gt 0 -and @($nupkgSha256s | Where-Object { -not (Test-Sha256Format -Value $_) }).Count -eq 0
$packagePackSafe = if ($sourceQualityOnlyMode) {
  [string]::IsNullOrWhiteSpace($packagePackConclusion) -or
  $packagePackConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase)
}
else {
  $packagePackConclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)
}
$forbiddenFindings = Get-ForbiddenSubstituteFindings `
  -Record $record `
  -RunUrl $runUrl `
  -RunStatus $runStatus `
  -RunConclusion $runConclusion `
  -WorkflowRunLogPath $workflowRunLogPath `
  -ArtifactManifestPath $artifactManifestPath `
  -OwnerReviewer $ownerReviewer

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "input-present" -Passed ($null -ne $record) -Severity "action-required" -Detail "Owner must import a real GitHub Actions run evidence artifact before this lane can be ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-required-fields-contract" -Passed ($githubActionsRunProofRequiredFields.Count -eq 13) -Severity "blocker" -Detail "GitHub Actions run proof admission contract must expose the canonical required field list.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-rejected-states-contract" -Passed ($githubActionsRunProofRejectedStates.Count -eq 12) -Severity "blocker" -Detail "GitHub Actions run proof admission contract must expose rejected run/non-proof states.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-rejected-substitutes-contract" -Passed ($githubActionsRunProofRejectedSubstitutes.Count -eq 10) -Severity "blocker" -Detail "GitHub Actions run proof admission contract must expose rejected substitute evidence sources.")) | Out-Null
$items.Add((New-ValidationItem -Id "record-kind" -Passed ($null -eq $record -or [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "github-actions-run-evidence-import") -Severity "blocker" -Detail "recordKind must be github-actions-run-evidence-import when input is present.")) | Out-Null
$items.Add((New-ValidationItem -Id "run-id-present" -Passed (-not [string]::IsNullOrWhiteSpace($runId)) -Severity "action-required" -Detail "runId must identify the GitHub Actions workflow run.")) | Out-Null
$items.Add((New-ValidationItem -Id "run-url-public-run-detail" -Passed (Test-GitHubActionsRunUrl -Value $runUrl) -Severity "action-required" -Detail "runUrl must be a GitHub workflow run detail URL under /actions/runs/<id>, not a dashboard or placeholder.")) | Out-Null
$items.Add((New-ValidationItem -Id "run-status-completed" -Passed ($runStatus.Equals("completed", [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "GitHub Actions run status must be completed.")) | Out-Null
$items.Add((New-ValidationItem -Id "run-conclusion-success" -Passed ($runConclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "GitHub Actions run conclusion must be success.")) | Out-Null
$items.Add((New-ValidationItem -Id "run-attempt-present" -Passed $runAttemptIsPositiveInteger -Severity "action-required" -Detail "runAttempt must be a positive integer captured from the workflow run.")) | Out-Null
$items.Add((New-ValidationItem -Id "workflow-name-present" -Passed (Test-ConcreteValue -Value $workflowName) -Severity "action-required" -Detail "workflowName must identify the workflow that ran.")) | Out-Null
$items.Add((New-ValidationItem -Id "workflow-file-present" -Passed (Test-ConcreteValue -Value $workflowFile) -Severity "action-required" -Detail "workflowFile must identify the workflow YAML file.")) | Out-Null
$items.Add((New-ValidationItem -Id "event-present" -Passed (Test-ConcreteValue -Value $runEvent) -Severity "action-required" -Detail "runEvent must identify the trigger event, such as workflow_dispatch or push.")) | Out-Null
$items.Add((New-ValidationItem -Id "ref-or-branch-present" -Passed ((Test-ConcreteValue -Value $runRef) -or (Test-ConcreteValue -Value $runBranch)) -Severity "action-required" -Detail "runRef or runBranch must link the run back to the release branch/ref.")) | Out-Null
$items.Add((New-ValidationItem -Id "runner-os-present" -Passed (Test-ConcreteValue -Value $runnerOs) -Severity "action-required" -Detail "runnerOs must identify the GitHub Actions runner operating system that produced the run evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "started-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value $startedAtUtc) -Severity "action-required" -Detail "startedAtUtc must be parseable as a DateTimeOffset.")) | Out-Null
$items.Add((New-ValidationItem -Id "completed-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value $completedAtUtc) -Severity "action-required" -Detail "completedAtUtc must be parseable as a DateTimeOffset.")) | Out-Null
$items.Add((New-ValidationItem -Id "completed-at-after-started-at" -Passed (Test-CompletedAfterStarted -StartedAtUtc $startedAtUtc -CompletedAtUtc $completedAtUtc) -Severity "action-required" -Detail "completedAtUtc must be equal to or later than startedAtUtc.")) | Out-Null
$items.Add((New-ValidationItem -Id "head-sha-format" -Passed ($headSha -match "^[0-9a-fA-F]{40}$") -Severity "action-required" -Detail "headSha must be a 40-character git commit SHA.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-head-link-present" -Passed $runHeadLinksSource -Severity "action-required" -Detail "headSha must match expectedHeadSha, current HEAD, or upstream HEAD to link this run to the reviewed release source.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-quality-success" -Passed ($sourceQualityConclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "source-quality job must succeed.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-quality-claim-ready" -Passed $canClaimSourceQuality -Severity "action-required" -Detail "Import must be internally ready to claim GitHub Actions source-quality evidence for the run.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-dry-run-pack-success" -Passed $packagePackSafe -Severity "action-required" -Detail "package-managed-dry-run / pack job must be success for package imports, or absent/skipped for source-only imports.")) | Out-Null
$publishJobsSafe = if ($sourceQualityOnlyMode) {
  (
    [string]::IsNullOrWhiteSpace($publishNugetConclusion) -or
    $publishNugetConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase)
  ) -and (
    [string]::IsNullOrWhiteSpace($publishGitHubPackagesConclusion) -or
    $publishGitHubPackagesConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase)
  )
}
else {
  $publishNugetConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase) -and
  $publishGitHubPackagesConclusion.Equals("skipped", [StringComparison]::OrdinalIgnoreCase)
}
$items.Add((New-ValidationItem -Id "publish-jobs-skipped" -Passed ($null -eq $record -or $publishJobsSafe) -Severity "blocker" -Detail "Publish jobs must be absent or skipped for this import; it is CI evidence, not publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "nupkg-packages-present" -Passed ((-not $packageEvidenceRequired) -or $nupkgPackages.Count -gt 0) -Severity "action-required" -Detail "At least one package dry-run nupkg artifact must be present when package dry-run evidence is imported.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-artifact-sha256s-present" -Passed ((-not $packageEvidenceRequired) -or $packageSha256sPresent) -Severity "action-required" -Detail "Every package artifact summary must include a 64-character SHA256 when package dry-run evidence is imported.")) | Out-Null
$items.Add((New-ValidationItem -Id "import-blockers-zero" -Passed ($null -eq $record -or $importFailedBlockerCount -eq 0) -Severity "blocker" -Detail "Import failedBlockerCount must be zero.")) | Out-Null
$items.Add((New-ValidationItem -Id "dry-run-pack-claim-ready" -Passed ((-not $packageEvidenceRequired) -or $canClaimDryRunPack) -Severity "action-required" -Detail "Import must be internally ready to claim GitHub Actions package dry-run pack evidence when package dry-run mode is used.")) | Out-Null
$items.Add((New-ValidationItem -Id "workflow-run-log-path-present" -Passed (Test-ConcreteValue -Value $workflowRunLogPath) -Severity "action-required" -Detail "A real GitHub Actions proof lane needs the workflow run log path.")) | Out-Null
$items.Add((New-ValidationItem -Id "workflow-run-log-sha256-present" -Passed (Test-Sha256Format -Value $workflowRunLogSha256) -Severity "action-required" -Detail "A real GitHub Actions proof lane needs the workflow run log SHA256; dry-run package evidence alone is not enough.")) | Out-Null
$items.Add((New-ValidationItem -Id "workflow-run-log-hash-match" -Passed (Test-FileHashMatches -Path $workflowRunLogPath -Sha256 $workflowRunLogSha256) -Severity "action-required" -Detail "workflowRunLogSha256 must match the imported workflow run log file.")) | Out-Null
$items.Add((New-ValidationItem -Id "artifact-manifest-path-present" -Passed (Test-ConcreteValue -Value $artifactManifestPath) -Severity "action-required" -Detail "A real GitHub Actions proof lane needs an artifact manifest path.")) | Out-Null
$items.Add((New-ValidationItem -Id "artifact-manifest-sha256-present" -Passed (Test-Sha256Format -Value $artifactManifestSha256) -Severity "action-required" -Detail "A real GitHub Actions proof lane needs an artifact manifest SHA256 tying downloaded artifacts to the run.")) | Out-Null
$items.Add((New-ValidationItem -Id "artifact-manifest-hash-match" -Passed (Test-FileHashMatches -Path $artifactManifestPath -Sha256 $artifactManifestSha256) -Severity "action-required" -Detail "artifactManifestSha256 must match the imported artifact manifest file.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-reviewer-present" -Passed (Test-ConcreteValue -Value $ownerReviewer) -Severity "action-required" -Detail "ownerReviewer must identify the person who captured or reviewed the imported run evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "captured-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value $capturedAtUtc) -Severity "action-required" -Detail "capturedAtUtc must be parseable as a DateTimeOffset.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes-absent" -Passed ($forbiddenFindings.Count -eq 0) -Severity "blocker" -Detail $(if ($forbiddenFindings.Count -eq 0) { "No queued workflow, dashboard, local feed, direct nupkg, manual approval, ProjectReference, or dry-run-only substitute was detected in owner evidence fields." } else { "Forbidden substitute(s): $($forbiddenFindings -join ', ')" }))) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ($null -eq $record -or (
      [bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $false) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $false) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isGitHubActionsProof" -DefaultValue $false)
    )) -Severity "blocker" -Detail "Validator must not classify the import as publish, runtime, post-publish, release-close, or GitHub Actions proof by itself.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$sourceQualityRunEvidenceReady = $sourceQualityOnlyMode -and $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0 -and $canClaimSourceQuality
$packageDryRunEvidenceReady = (-not $sourceQualityOnlyMode) -and $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0 -and $canClaimDryRunPack
$ready = $sourceQualityRunEvidenceReady -or $packageDryRunEvidenceReady
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-github-actions-run-evidence-import"
}
elseif ($packageDryRunEvidenceReady) {
  "github-actions-run-evidence-ready"
}
elseif ($sourceQualityRunEvidenceReady) {
  "source-quality-run-evidence-ready"
}
else {
  "blocked-github-actions-run-evidence-required"
}

$validation = [pscustomobject]@{
  recordKind = "github-actions-run-evidence-import-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  importMode = $importMode
  evidenceState = $evidenceState
  sourceQualityRunEvidenceReady = $sourceQualityRunEvidenceReady
  packageDryRunEvidenceReady = $packageDryRunEvidenceReady
  githubActionsRunEvidenceReady = $packageDryRunEvidenceReady
  githubActionsRunProofRequiredFields = @($githubActionsRunProofRequiredFields)
  githubActionsRunProofRequiredFieldCount = $githubActionsRunProofRequiredFields.Count
  githubActionsRunProofRejectedStates = @($githubActionsRunProofRejectedStates)
  githubActionsRunProofRejectedStateCount = $githubActionsRunProofRejectedStates.Count
  githubActionsRunProofRejectedSubstitutes = @($githubActionsRunProofRejectedSubstitutes)
  githubActionsRunProofRejectedSubstituteCount = $githubActionsRunProofRejectedSubstitutes.Count
  githubRunId = $runId
  githubRunUrl = $runUrl
  headBranch = $runBranch
  status = $runStatus
  conclusion = $runConclusion
  createdAtUtc = $startedAtUtc
  updatedAtUtc = $completedAtUtc
  logSha256 = $workflowRunLogSha256
  artifactSha256 = $artifactManifestSha256
  canClaimGitHubActionsSourceQualityForRun = $canClaimSourceQuality
  canClaimGitHubActionsPackageDryRunPackForRun = $canClaimDryRunPack
  runId = $runId
  runUrl = $runUrl
  runStatus = $runStatus
  runConclusion = $runConclusion
  runAttempt = $runAttempt
  workflowName = $workflowName
  workflowFile = $workflowFile
  runEvent = $runEvent
  runBranch = $runBranch
  runRef = $runRef
  runnerOs = $runnerOs
  startedAtUtc = $startedAtUtc
  completedAtUtc = $completedAtUtc
  headSha = $headSha
  expectedHeadSha = $expectedHeadSha
  runHeadMatchesCurrentHead = $runHeadMatchesCurrentHead
  runHeadMatchesUpstreamHead = $runHeadMatchesUpstreamHead
  sourceQualityConclusion = $sourceQualityConclusion
  packageManagedDryRunPackConclusion = $packagePackConclusion
  publishNugetConclusion = $publishNugetConclusion
  publishGitHubPackagesConclusion = $publishGitHubPackagesConclusion
  nupkgPackageCount = $nupkgPackages.Count
  nupkgPackageSha256s = @($nupkgSha256s)
  workflowRunLogPath = $workflowRunLogPath
  workflowRunLogSha256 = $workflowRunLogSha256
  artifactManifestPath = $artifactManifestPath
  artifactManifestSha256 = $artifactManifestSha256
  ownerReviewer = $ownerReviewer
  capturedAtUtc = $capturedAtUtc
  forbiddenSubstituteFindings = @($forbiddenFindings)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "GitHub Actions run evidence import validation is read-only. It can make the remote CI lane structurally ready only after real imported run artifacts pass, but it is not package publish proof, not runtime proof, not post-publish proof, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "github-actions-run-evidence-import-validation.json"
$markdownPath = Join-Path $OutputRoot "github-actions-run-evidence-import-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# GitHub Actions Run Evidence Import Validation

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| githubActionsRunEvidenceReady | ``$($validation.githubActionsRunEvidenceReady)`` |
| sourceQualityRunEvidenceReady | ``$($validation.sourceQualityRunEvidenceReady)`` |
| packageDryRunEvidenceReady | ``$($validation.packageDryRunEvidenceReady)`` |
| githubActionsRunProofRequiredFieldCount | ``$($validation.githubActionsRunProofRequiredFieldCount)`` |
| githubActionsRunProofRejectedStateCount | ``$($validation.githubActionsRunProofRejectedStateCount)`` |
| githubActionsRunProofRejectedSubstituteCount | ``$($validation.githubActionsRunProofRejectedSubstituteCount)`` |
| importMode | ``$($validation.importMode)`` |
| evidenceState | ``$($validation.evidenceState)`` |
| runId | ``$($validation.runId)`` |
| runUrl | ``$($validation.runUrl)`` |
| headSha | ``$($validation.headSha)`` |
| workflowName | ``$($validation.workflowName)`` |
| workflowFile | ``$($validation.workflowFile)`` |
| runAttempt | ``$($validation.runAttempt)`` |
| runnerOs | ``$($validation.runnerOs)`` |
| workflowRunLogSha256 | ``$($validation.workflowRunLogSha256)`` |
| artifactManifestSha256 | ``$($validation.artifactManifestSha256)`` |
| ownerReviewer | ``$($validation.ownerReviewer)`` |
| capturedAtUtc | ``$($validation.capturedAtUtc)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| isPostPublishProof | ``$($validation.isPostPublishProof)`` |
| isGitHubActionsProof | ``$($validation.isGitHubActionsProof)`` |

## Proof Admission Contract

### Required Fields

$($githubActionsRunProofRequiredFields | ForEach-Object { "- ``$_``" } | Out-String)

### Rejected States

$($githubActionsRunProofRejectedStates | ForEach-Object { "- ``$_``" } | Out-String)

### Rejected Substitutes

$($githubActionsRunProofRejectedSubstitutes | ForEach-Object { "- ``$_``" } | Out-String)

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "GitHub Actions run evidence import validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "GitHub Actions run evidence import validation failed with $($failedBlockers.Count) blocker(s)."
}
