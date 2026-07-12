[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\package-consumer-runtime-proof-owner-input.template.json",
  [string]$ConsumerProjectPath,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $Path
  }

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Test-Placeholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-LocalPathLike {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if ([string]::IsNullOrWhiteSpace($text)) {
    return $false
  }

  return $text -match "^[a-zA-Z]:[\\/]" -or
    $text.StartsWith("\\", [StringComparison]::Ordinal) -or
    $text.StartsWith("./", [StringComparison]::Ordinal) -or
    $text.StartsWith("../", [StringComparison]::Ordinal)
}

function Test-PublicPackageSourceIsLocal {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if (Test-Placeholder -Value $text) {
    return $true
  }

  if (Test-LocalPathLike -Value $text) {
    return $true
  }

  return $text.Contains("local", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
}

function Test-IsInsideRepository {
  param([AllowNull()][object]$Path)

  $text = [string]$Path
  if (Test-Placeholder -Value $text) {
    return $true
  }

  try {
    $repositoryFullPath = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $candidateFullPath = [IO.Path]::GetFullPath($text).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    return $candidateFullPath.StartsWith($repositoryFullPath, [StringComparison]::OrdinalIgnoreCase)
  }
  catch {
    return $true
  }
}

function New-ScanItem {
  param(
    [string]$Id,
    [string]$Label,
    [bool]$Detected,
    [string]$Severity,
    [string]$Evidence,
    [string]$RequiredAction
  )

  [pscustomobject]@{
    id = $Id
    label = $Label
    detected = $Detected
    passed = (-not $Detected)
    severity = $Severity
    evidence = $Evidence
    requiredAction = $RequiredAction
  }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
$ownerInput = $null
if (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf) {
  $ownerInput = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$effectiveConsumerProjectPath = $ConsumerProjectPath
if ([string]::IsNullOrWhiteSpace($effectiveConsumerProjectPath) -and $null -ne $ownerInput) {
  $effectiveConsumerProjectPath = [string](Get-PropertyOrDefault -Object $ownerInput -Name "consumerProjectPath" -DefaultValue "")
}

$resolvedConsumerProjectPath = Resolve-RepositoryPath -Path $effectiveConsumerProjectPath
$projectExists = -not [string]::IsNullOrWhiteSpace($resolvedConsumerProjectPath) -and (Test-Path -LiteralPath $resolvedConsumerProjectPath -PathType Leaf)
$projectContent = ""
if ($projectExists) {
  $projectContent = Get-Content -LiteralPath $resolvedConsumerProjectPath -Raw -Encoding utf8
}

$publicPackageSource = if ($null -eq $ownerInput) { "" } else { [string](Get-PropertyOrDefault -Object $ownerInput -Name "publicPackageSource" -DefaultValue "") }
$restoreCommand = if ($null -eq $ownerInput) { "" } else { [string](Get-PropertyOrDefault -Object $ownerInput -Name "restoreCommand" -DefaultValue "") }
$buildCommand = if ($null -eq $ownerInput) { "" } else { [string](Get-PropertyOrDefault -Object $ownerInput -Name "buildCommand" -DefaultValue "") }
$smokeCommand = if ($null -eq $ownerInput) { "" } else { [string](Get-PropertyOrDefault -Object $ownerInput -Name "smokeCommand" -DefaultValue "") }
$stdoutSummary = if ($null -eq $ownerInput) { "" } else { [string](Get-PropertyOrDefault -Object $ownerInput -Name "stdoutSummary" -DefaultValue "") }
$stderrSummary = if ($null -eq $ownerInput) { "" } else { [string](Get-PropertyOrDefault -Object $ownerInput -Name "stderrSummary" -DefaultValue "") }
$sourceRunnerQueueStatus = if ($null -eq $ownerInput) { "" } else { [string](Get-PropertyOrDefault -Object $ownerInput -Name "sourceRunnerQueueStatus" -DefaultValue "") }
$sourceRunnerInfrastructureStatus = if ($null -eq $ownerInput) { "" } else { [string](Get-PropertyOrDefault -Object $ownerInput -Name "sourceRunnerInfrastructureStatus" -DefaultValue "") }
$sourceRunnerOwnerAction = if ($null -eq $ownerInput) { "" } else { [string](Get-PropertyOrDefault -Object $ownerInput -Name "sourceRunnerOwnerAction" -DefaultValue "") }

$combinedText = @(
  $publicPackageSource,
  $restoreCommand,
  $buildCommand,
  $smokeCommand,
  $stdoutSummary,
  $stderrSummary,
  $sourceRunnerQueueStatus,
  $sourceRunnerInfrastructureStatus,
  $sourceRunnerOwnerAction,
  $projectContent
) -join "`n"

$repoEscaped = [Regex]::Escape((Resolve-Path -LiteralPath $RepositoryRoot).Path)
$usesProjectReference = $projectContent.Contains("<ProjectReference", [StringComparison]::OrdinalIgnoreCase) -or $combinedText -match "(?i)ProjectReference"
$usesLocalFeed = (Test-PublicPackageSourceIsLocal -Value $publicPackageSource) -or
  $projectContent.Contains("RestoreSources", [StringComparison]::OrdinalIgnoreCase) -and
  ($projectContent.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or $projectContent.Contains("local", [StringComparison]::OrdinalIgnoreCase)) -or
  $combinedText -match "(?i)local feed|local-feed"
$usesDirectNupkg = $combinedText.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
$cleanExternalConsumerRoot = ""
if ($null -ne $ownerInput) {
  $cleanExternalConsumerRoot = [string](Get-PropertyOrDefault -Object $ownerInput -Name "cleanExternalConsumerRoot" -DefaultValue "")
}

$repositoryPathLeakage = $combinedText -match $repoEscaped -or (Test-IsInsideRepository -Path $cleanExternalConsumerRoot)
$buildOnly = $combinedText -match "(?i)build-only|build only" -or
  ($smokeCommand -match "(?i)dotnet build|--no-run" -and -not ($smokeCommand -match "(?i)dotnet run"))
$dryRun = $combinedText -match "(?i)dry-run|dryrun|--dry"
$queuedGitHubActionsRun = $combinedText -match "(?i)queued GitHub Actions run|github actions.*queued|sourceRunnerQueueStatus.*queued" -or
  $sourceRunnerQueueStatus.Equals("queued", [StringComparison]::OrdinalIgnoreCase)
$missingSelfHostedRunner = $combinedText -match "(?i)missing self-hosted runner|self-hosted runner.*missing|no self-hosted runner|sourceRunnerInfrastructureStatus.*missing" -or
  $sourceRunnerInfrastructureStatus.Equals("missing-self-hosted-runner", [StringComparison]::OrdinalIgnoreCase) -or
  $sourceRunnerInfrastructureStatus.Equals("missing", [StringComparison]::OrdinalIgnoreCase)
$templatePlaceholder = $null -eq $ownerInput
if ($null -ne $ownerInput) {
  foreach ($property in $ownerInput.PSObject.Properties) {
    $templatePlaceholder = $templatePlaceholder -or (Test-Placeholder -Value $property.Value)
  }
}
$guiScreenshot = $combinedText -match "(?i)GUI screenshot|WinForms screenshot|screenshot|截图|\.png|\.jpg|\.jpeg"
$tensorRtExecBuildReportOnly = $combinedText -match "(?i)TensorRtExec build report|TensorRtExec.*build-only|LoadEngineDiagnostics|readonly diagnostics"

$items = @(
  New-ScanItem -Id "local-feed" -Label "local feed" -Detected $usesLocalFeed -Severity "blocker" -Evidence "publicPackageSource/RestoreSources/local-feed markers" -RequiredAction "Use a public package source that is not a local folder/feed."
  New-ScanItem -Id "project-reference" -Label "ProjectReference" -Detected $usesProjectReference -Severity "blocker" -Evidence "consumer project or text contains ProjectReference markers" -RequiredAction "Use PackageReference from the public package source only."
  New-ScanItem -Id "direct-nupkg" -Label "direct .nupkg" -Detected $usesDirectNupkg -Severity "blocker" -Evidence "owner input or project text contains .nupkg" -RequiredAction "Do not reference local/direct nupkg files as package-consumer proof."
  New-ScanItem -Id "repository-path-leakage" -Label "repository path leakage" -Detected $repositoryPathLeakage -Severity "blocker" -Evidence "cleanExternalConsumerRoot or command text points inside repository" -RequiredAction "Run the clean consumer outside the repository and remove source path dependencies."
  New-ScanItem -Id "build-only" -Label "build-only" -Detected $buildOnly -Severity "blocker" -Evidence "build-only markers or smoke command without runtime run semantics" -RequiredAction "Provide a runtime smoke command and output, not build-only evidence."
  New-ScanItem -Id "dry-run" -Label "dry-run" -Detected $dryRun -Severity "blocker" -Evidence "dry-run markers" -RequiredAction "Use a real runtime execution, not dry-run output."
  New-ScanItem -Id "queued-github-actions-run" -Label "queued GitHub Actions run" -Detected $queuedGitHubActionsRun -Severity "blocker" -Evidence "source runner queue status or text says queued" -RequiredAction "Wait for a completed run or provide real owner clean-consumer runtime evidence."
  New-ScanItem -Id "missing-self-hosted-runner" -Label "missing self-hosted runner" -Detected $missingSelfHostedRunner -Severity "blocker" -Evidence "source runner infrastructure status or text says missing runner" -RequiredAction "Fix runner infrastructure or keep this as owner-infra-action, not runtime proof."
  New-ScanItem -Id "template-placeholder" -Label "template placeholder" -Detected $templatePlaceholder -Severity "blocker" -Evidence "placeholder owner input value such as <owner-fill-...>" -RequiredAction "Replace every placeholder with real owner-provided evidence."
  New-ScanItem -Id "gui-screenshot" -Label "GUI screenshot" -Detected $guiScreenshot -Severity "blocker" -Evidence "screenshot markers" -RequiredAction "Use machine-verifiable command logs and hashes, not screenshots."
  New-ScanItem -Id "tensorrtexec-build-report-only" -Label "TensorRtExec build report only" -Detected $tensorRtExecBuildReportOnly -Severity "blocker" -Evidence "TensorRtExec build/read-only diagnostics markers" -RequiredAction "Provide external clean consumer runtime proof; TensorRtExec build/read-only reports are not enough."
)

$detectedItems = @($items | Where-Object { $_.detected })
$scanState = if ($detectedItems.Count -eq 0) {
  "forbidden-substitute-scan-clear"
}
else {
  "blocked-forbidden-substitute-detected"
}

$scan = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-forbidden-substitute-scan"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath
  resolvedInputPath = $resolvedInputPath
  consumerProjectPath = $effectiveConsumerProjectPath
  resolvedConsumerProjectPath = $resolvedConsumerProjectPath
  consumerProjectExists = $projectExists
  scanState = $scanState
  forbiddenSubstituteCount = $items.Count
  detectedForbiddenSubstituteCount = $detectedItems.Count
  scanItems = @($items)
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPromoteProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Forbidden substitute scan only. It detects non-proof substitutes and never publishes packages, closes release issues, or promotes package-consumer runtime proof."
}

$jsonPath = Join-Path $artifactRoot "package-consumer-runtime-proof-forbidden-substitute-scan.json"
$markdownPath = Join-Path $artifactRoot "package-consumer-runtime-proof-forbidden-substitute-scan.md"

$scan | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $scan.scanItems | ForEach-Object {
  "| ``$($_.id)`` | $($_.label.Replace("|", "\|")) | ``$($_.detected)`` | ``$($_.severity)`` | $($_.requiredAction.Replace("|", "\|")) |"
}

$markdown = @"
# Package Consumer Runtime Proof Forbidden Substitute Scan

生成时间：$($scan.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| scanState | ``$($scan.scanState)`` |
| inputPath | ``$($scan.inputPath)`` |
| consumerProjectPath | ``$($scan.consumerProjectPath)`` |
| consumerProjectExists | ``$($scan.consumerProjectExists)`` |
| forbiddenSubstituteCount | ``$($scan.forbiddenSubstituteCount)`` |
| detectedForbiddenSubstituteCount | ``$($scan.detectedForbiddenSubstituteCount)`` |
| canPromoteRuntimeProof | ``$($scan.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($scan.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($scan.canCloseReleaseIssue)`` |

## Scan Items

| ID | Substitute | Detected | Severity | Required Action |
|---|---|---|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($scan.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Package consumer runtime proof forbidden substitute scan written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ScanState=$($scan.scanState) Detected=$($scan.detectedForbiddenSubstituteCount) CanPromoteRuntimeProof=False"
