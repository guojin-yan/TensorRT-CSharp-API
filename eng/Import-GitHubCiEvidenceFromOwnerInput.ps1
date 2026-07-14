[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\github-ci-evidence.owner-input.template.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($OwnerInputPath)) { $OwnerInputPath = Join-Path $RepositoryRoot $OwnerInputPath }

function Get-GitText {
  param([string[]]$Arguments)
  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = "git"
  $psi.WorkingDirectory = $RepositoryRoot
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  foreach ($argument in $Arguments) { $psi.ArgumentList.Add($argument) }
  $process = [System.Diagnostics.Process]::Start($psi)
  $stdout = $process.StandardOutput.ReadToEnd()
  $process.WaitForExit()
  if ($process.ExitCode -ne 0) { return "" }
  return $stdout.Trim()
}

function Test-ConcreteOwnerValue {
  param([AllowNull()][object]$Value)
  if (Test-OwnerPlaceholder $Value) { return $false }
  $text = [string]$Value
  return $text -notmatch '(?i)\b(todo|tbd|sample|example|dummy|fake|local-only)\b'
}

function Test-DateTimeOffsetText {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  if ([string]::IsNullOrWhiteSpace($text)) { return $false }
  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse($text, [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::AssumeUniversal, [ref]$parsed)
}

function Test-GitHubRunUrl {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return $text.StartsWith("https://github.com/", [StringComparison]::OrdinalIgnoreCase) -and $text -match '/actions/runs/[0-9]+'
}

$currentHeadSha = Get-GitText @("rev-parse", "HEAD")
$templatePath = Join-Path $OutputRoot "github-ci-evidence.owner-input.template.json"
if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
  $template = [pscustomobject]@{
    recordKind = "github-ci-evidence-owner-input"
    repository = "<owner-github-repository>"
    branch = "TensorRtSharp4.0"
    commitSha = $currentHeadSha
    workflowName = "release-quality-gate"
    runId = "<owner-github-actions-run-id>"
    runUrl = "<owner-github-actions-run-url>"
    conclusion = "<owner-conclusion>"
    createdAtUtc = "<owner-created-at-utc>"
    completedAtUtc = "<owner-completed-at-utc>"
    artifactManifestSha256 = "<owner-artifact-manifest-sha256>"
    ownerReviewer = "<owner-reviewer>"
    ownerReviewed = $false
  }
  Write-Utf8File -LiteralPath $templatePath -InputObject ($template | ConvertTo-Json -Depth 6)
}
if (-not (Test-Path -LiteralPath $OwnerInputPath -PathType Leaf)) { $OwnerInputPath = $templatePath }
$input = Get-Content -LiteralPath $OwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$findings = New-Object System.Collections.Generic.List[object]

foreach ($field in @("repository", "branch", "commitSha", "workflowName", "runId", "runUrl", "conclusion", "createdAtUtc", "completedAtUtc", "artifactManifestSha256", "ownerReviewer")) {
  if (-not (Test-ConcreteOwnerValue (Get-PropertyOrDefault -Object $input -Name $field -DefaultValue $null))) {
    $findings.Add((New-OwnerFinding "$field-missing" "action-required" "missing-field" "GitHub CI evidence field is missing or placeholder: $field")) | Out-Null
  }
}

$runUrl = [string](Get-PropertyOrDefault -Object $input -Name "runUrl" -DefaultValue "")
if (-not (Test-GitHubRunUrl $runUrl)) { $findings.Add((New-OwnerFinding "runUrl-format" "action-required" "invalid-url" "runUrl must be a GitHub Actions run URL.")) | Out-Null }
$runId = [string](Get-PropertyOrDefault -Object $input -Name "runId" -DefaultValue "")
if ($runId -notmatch '^[0-9]+$') { $findings.Add((New-OwnerFinding "runId-format" "action-required" "invalid-run-id" "runId must be numeric.")) | Out-Null }
$commitSha = [string](Get-PropertyOrDefault -Object $input -Name "commitSha" -DefaultValue "")
if ($commitSha -notmatch '^[0-9a-fA-F]{40}$') { $findings.Add((New-OwnerFinding "commitSha-format" "action-required" "invalid-commit-sha" "commitSha must be 40 hex characters.")) | Out-Null }
if (-not [string]::IsNullOrWhiteSpace($currentHeadSha) -and -not $commitSha.Equals($currentHeadSha, [StringComparison]::OrdinalIgnoreCase)) {
  $findings.Add((New-OwnerFinding "commitSha-current-head-mismatch" "action-required" "commit-mismatch" "Owner CI evidence must match current HEAD commit.")) | Out-Null
}
$conclusion = [string](Get-PropertyOrDefault -Object $input -Name "conclusion" -DefaultValue "")
if (-not $conclusion.Equals("success", [StringComparison]::OrdinalIgnoreCase)) { $findings.Add((New-OwnerFinding "conclusion-not-success" "action-required" "ci-not-success" "GitHub Actions conclusion must be success.")) | Out-Null }
foreach ($field in @("createdAtUtc", "completedAtUtc")) {
  if (-not (Test-DateTimeOffsetText (Get-PropertyOrDefault -Object $input -Name $field -DefaultValue ""))) {
    $findings.Add((New-OwnerFinding "$field-format" "action-required" "invalid-timestamp" "$field must be a timestamp.")) | Out-Null
  }
}
if (-not (Test-Sha256Text (Get-PropertyOrDefault -Object $input -Name "artifactManifestSha256" -DefaultValue ""))) {
  $findings.Add((New-OwnerFinding "artifactManifestSha256-format" "action-required" "invalid-sha256" "artifactManifestSha256 must be 64 hex characters.")) | Out-Null
}
if (-not [bool](Get-PropertyOrDefault -Object $input -Name "ownerReviewed" -DefaultValue $false)) {
  $findings.Add((New-OwnerFinding "ownerReviewed-required" "action-required" "missing-owner-review" "Owner must review GitHub CI evidence.")) | Out-Null
}

$failed = @($findings | Where-Object { [string]$_.severity -eq "blocker" -or [string]$_.severity -eq "action-required" })
$accepted = $failed.Count -eq 0
$record = [pscustomobject]@{
  recordKind = "github-ci-evidence-from-owner-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = if ($accepted) { "github-ci-evidence-owner-input-accepted-non-proof" } else { "blocked-github-ci-evidence-owner-input-required" }
  ownerInputPath = $OwnerInputPath
  currentHeadSha = $currentHeadSha
  repository = [string](Get-PropertyOrDefault -Object $input -Name "repository" -DefaultValue "")
  branch = [string](Get-PropertyOrDefault -Object $input -Name "branch" -DefaultValue "")
  commitSha = $commitSha
  workflowName = [string](Get-PropertyOrDefault -Object $input -Name "workflowName" -DefaultValue "")
  runId = $runId
  runUrl = $runUrl
  conclusion = $conclusion
  artifactManifestSha256 = [string](Get-PropertyOrDefault -Object $input -Name "artifactManifestSha256" -DefaultValue "")
  ownerReviewed = [bool](Get-PropertyOrDefault -Object $input -Name "ownerReviewed" -DefaultValue $false)
  ciEvidenceAccepted = $accepted
  findingCount = $findings.Count
  failedActionRequiredCount = $failed.Count
  findings = @($findings.ToArray())
  ownerActionRequired = -not $accepted
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "GitHub CI evidence import records Owner-reviewed completed CI metadata only; queued, in-progress, local build, local test, dashboard, dry-run, local feed, ProjectReference, and direct nupkg signals are not proof. This import does not publish, does not use tokens, does not close the release issue, and is not runtime proof, not post-publish proof, not release close approval, and not package push."
}
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "github-ci-evidence-from-owner-input.json") -InputObject ($record | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "github-ci-evidence-from-owner-input.md") -InputObject @("# GitHub CI Evidence From Owner Input", "", "- importState: ``$($record.importState)``", "- ciEvidenceAccepted: ``$accepted``", "- failedActionRequiredCount: ``$($failed.Count)``", "", $record.boundary)
Write-Host "GitHubCiEvidenceFromOwnerInputState=$($record.importState) Accepted=$accepted FailedActionRequired=$($failed.Count)"
if ($FailOnNotProof.IsPresent -and -not $accepted) { throw "GitHub CI evidence Owner input is not accepted." }
