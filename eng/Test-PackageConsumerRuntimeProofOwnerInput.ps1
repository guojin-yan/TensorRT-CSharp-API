[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\package-consumer-runtime-proof-owner-input.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

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

function Get-BoolPropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [bool]$DefaultValue
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue
  if ($value -is [bool]) {
    return [bool]$value
  }

  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) {
    return $parsed
  }

  return $DefaultValue
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-IntZero {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = 0
  return [int]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed -eq 0
}

function Test-BoolTrue {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = $false
  return [bool]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed
}

function Test-DateTimeOffsetFormat {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed)
}

function Test-ValueInSet {
  param(
    [AllowNull()][object]$Value,
    [string[]]$AllowedValues
  )

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $text = ([string]$Value).Trim()
  foreach ($allowedValue in $AllowedValues) {
    if ($text.Equals($allowedValue, [StringComparison]::OrdinalIgnoreCase)) {
      return $true
    }
  }

  return $false
}

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Test-IsOutsideRepository {
  param([AllowNull()][object]$Path)

  $text = [string]$Path
  if (Test-IsPlaceholder -Value $text) {
    return $false
  }

  try {
    $repositoryFullPath = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $candidateFullPath = [IO.Path]::GetFullPath($text).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    return -not $candidateFullPath.StartsWith($repositoryFullPath, [StringComparison]::OrdinalIgnoreCase)
  }
  catch {
    return $false
  }
}

function Test-FileHashMatches {
  param(
    [AllowNull()][object]$Path,
    [AllowNull()][object]$Sha256
  )

  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $shaText)) {
    return $false
  }

  $resolvedPath = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $false
  }

  $actual = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

function Read-JsonOrNull {
  param([string]$Path)

  if (Test-IsPlaceholder -Value $Path) {
    return $null
  }

  $resolvedPath = Resolve-InputPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Test-ManagedNupkgPathIsNotDryRunArtifact {
  param(
    [AllowNull()][object]$ManagedPath,
    [AllowNull()][object]$DryRunPath
  )

  $managedText = [string]$ManagedPath
  $dryRunText = [string]$DryRunPath
  if (Test-IsPlaceholder -Value $managedText) {
    return $true
  }

  if (-not (Test-IsPlaceholder -Value $dryRunText)) {
    try {
      $managedFullPath = [IO.Path]::GetFullPath((Resolve-InputPath -Path $managedText))
      $dryRunFullPath = [IO.Path]::GetFullPath((Resolve-InputPath -Path $dryRunText))
      if ($managedFullPath.Equals($dryRunFullPath, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
      }
    }
    catch {
      if ($managedText.Equals($dryRunText, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
      }
    }
  }

  return -not $managedText.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase) -and
    -not $managedText.Contains("github-actions-runs", [StringComparison]::OrdinalIgnoreCase)
}

function Test-PublicPackageSourceIsLocal {
  param([AllowNull()][object]$Value)

  $text = ([string]$Value).Trim()
  if (Test-IsPlaceholder -Value $text) {
    return $true
  }

  if ($text -match "^[a-zA-Z]:[\\/]" -or $text.StartsWith("\\", [StringComparison]::Ordinal) -or $text.StartsWith("./", [StringComparison]::Ordinal) -or $text.StartsWith("../", [StringComparison]::Ordinal)) {
    return $true
  }

  return $text.Contains("local", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
}

function Test-PublicPackageSourceKind {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $text = ([string]$Value).Trim()
  return $text.Equals("nuget", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("github-packages", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("github packages", [StringComparison]::OrdinalIgnoreCase)
}

function Test-PublicUrl {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $text = ([string]$Value).Trim()
  return ($text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase) -or
    $text.StartsWith("http://", [StringComparison]::OrdinalIgnoreCase)) -and
    -not (Test-PublicPackageSourceIsLocal -Value $text)
}

function Test-RunnerQueueCompleted {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $text = ([string]$Value).Trim()
  return $text.Equals("completed", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("not-queued", [StringComparison]::OrdinalIgnoreCase)
}

function Test-RunnerInfrastructureReady {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $text = ([string]$Value).Trim()
  return $text.Equals("available", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("ready", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("not-required", [StringComparison]::OrdinalIgnoreCase)
}

function Get-ConsumerProjectReferenceFlags {
  param([AllowNull()][object]$ProjectPath)

  $pathText = [string]$ProjectPath
  $result = [ordered]@{
    projectExists = $false
    usesProjectReference = $true
    usesLocalFeed = $true
    usesDirectNupkg = $true
  }

  if (Test-IsPlaceholder -Value $pathText) {
    return [pscustomobject]$result
  }

  $resolvedPath = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return [pscustomobject]$result
  }

  $result.projectExists = $true
  $content = Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8
  $repoEscaped = [Regex]::Escape((Resolve-Path -LiteralPath $RepositoryRoot).Path)
  $result.usesProjectReference = $content.Contains("<ProjectReference", [StringComparison]::OrdinalIgnoreCase) -and
    ($content -match $repoEscaped -or $content.Contains("..\src\", [StringComparison]::OrdinalIgnoreCase) -or $content.Contains("../src/", [StringComparison]::OrdinalIgnoreCase))
  $result.usesLocalFeed = $content.Contains("RestoreSources", [StringComparison]::OrdinalIgnoreCase) -and
    ($content.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or $content.Contains("local", [StringComparison]::OrdinalIgnoreCase))
  $result.usesDirectNupkg = $content.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
  return [pscustomobject]$result
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Package consumer runtime proof owner input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$performsPublish = Get-BoolPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true
$canPublishPublicly = Get-BoolPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true
$canCloseReleaseIssue = Get-BoolPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true
$canPromoteProof = Get-BoolPropertyOrDefault -Object $record -Name "canPromoteProof" -DefaultValue $true
$isDryRunOnly = Get-BoolPropertyOrDefault -Object $record -Name "isDryRunOnly" -DefaultValue $true
$isPublishedPackageProof = Get-BoolPropertyOrDefault -Object $record -Name "isPublishedPackageProof" -DefaultValue $false
$isPackageConsumerRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $false
$packageDryRunCanClaimPack = Get-BoolPropertyOrDefault -Object $record -Name "packageDryRunCanClaimPack" -DefaultValue $false
$runtimePackageKey = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageKey" -DefaultValue "")
$smokeCommand = [string](Get-PropertyOrDefault -Object $record -Name "smokeCommand" -DefaultValue "")
$consumerProjectFlags = Get-ConsumerProjectReferenceFlags -ProjectPath (Get-PropertyOrDefault -Object $record -Name "consumerProjectPath" -DefaultValue "")
$sourceGitHubActionsRunEvidenceImportPath = [string](Get-PropertyOrDefault -Object $record -Name "sourceGitHubActionsRunEvidenceImportPath" -DefaultValue "")
$sourceGitHubActionsRunId = [string](Get-PropertyOrDefault -Object $record -Name "sourceGitHubActionsRunId" -DefaultValue "")
$sourceHeadSha = [string](Get-PropertyOrDefault -Object $record -Name "sourceHeadSha" -DefaultValue "")
$packageDryRunArtifactPath = [string](Get-PropertyOrDefault -Object $record -Name "packageDryRunArtifactPath" -DefaultValue "")
$packageDryRunManagedNupkgSha256 = [string](Get-PropertyOrDefault -Object $record -Name "packageDryRunManagedNupkgSha256" -DefaultValue "")
$sourceEvidenceImport = Read-JsonOrNull -Path $sourceGitHubActionsRunEvidenceImportPath
$sourceEvidenceImportPresent = $null -ne $sourceEvidenceImport -and ([string](Get-PropertyOrDefault -Object $sourceEvidenceImport -Name "recordKind" -DefaultValue "")).Equals("github-actions-run-evidence-import", [StringComparison]::Ordinal)
$sourceEvidenceDryRunClaim = $sourceEvidenceImportPresent -and $packageDryRunCanClaimPack -and (Test-Sha256Format -Value $packageDryRunManagedNupkgSha256) -and -not (Test-IsPlaceholder -Value $packageDryRunArtifactPath)

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "package-consumer-runtime-proof-owner-input") -Severity "blocker" -Detail "recordKind must be package-consumer-runtime-proof-owner-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue -and -not $canPromoteProof) -Severity "blocker" -Detail "Owner input validation must not publish, approve publication, close the issue, or promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-github-actions-run-evidence-import-present" -Passed $sourceEvidenceImportPresent -Severity "action-required" -Detail "sourceGitHubActionsRunEvidenceImportPath should point to github-actions-run-evidence-import.json when dry-run pack context is available.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-github-actions-dry-run-pack-claim-ready" -Passed $sourceEvidenceDryRunClaim -Severity "action-required" -Detail "GitHub Actions dry-run context must include run id, head SHA, package artifact path, package SHA256, and canClaimGitHubActionsPackageDryRunPackForRun=true.")) | Out-Null
$items.Add((New-ValidationItem -Id "dry-run-only-not-proof" -Passed $isDryRunOnly -Severity "blocker" -Detail "GitHub Actions package dry-run context must remain marked as dry-run-only and not proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "published-package-proof-false" -Passed (-not $isPublishedPackageProof) -Severity "blocker" -Detail "Owner input template and dry-run context are not published package proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "package-consumer-runtime-proof-false" -Passed (-not $isPackageConsumerRuntimeProof) -Severity "blocker" -Detail "Owner input template and dry-run context are not package-consumer runtime proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-root-outside-repository" -Passed (Test-IsOutsideRepository -Path (Get-PropertyOrDefault -Object $record -Name "cleanExternalConsumerRoot" -DefaultValue "")) -Severity "action-required" -Detail "cleanExternalConsumerRoot must be a real path outside this repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "consumer-project-exists" -Passed $consumerProjectFlags.projectExists -Severity "action-required" -Detail "consumerProjectPath must point to a real external .csproj.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-project-reference-to-repository" -Passed (-not $consumerProjectFlags.usesProjectReference) -Severity "action-required" -Detail "Clean consumer project must not use ProjectReference into this repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-source-kind" -Passed (Test-PublicPackageSourceKind -Value (Get-PropertyOrDefault -Object $record -Name "publicPackageSourceKind" -DefaultValue "")) -Severity "action-required" -Detail "publicPackageSourceKind must be NuGet or GitHub Packages; local-feed, direct-nupkg, dry-run, and queued-run source kinds are forbidden.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-source-not-local" -Passed (-not (Test-PublicPackageSourceIsLocal -Value (Get-PropertyOrDefault -Object $record -Name "publicPackageSource" -DefaultValue "")) -and -not $consumerProjectFlags.usesLocalFeed) -Severity "action-required" -Detail "publicPackageSource and consumer project restore sources must not be local folder/feed evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-feed-url" -Passed (Test-PublicUrl -Value (Get-PropertyOrDefault -Object $record -Name "publicPackageFeedUrl" -DefaultValue "")) -Severity "action-required" -Detail "publicPackageFeedUrl must be a public NuGet/GitHub Packages URL, not a local path or dry-run artifact.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-package-url" -Passed (Test-PublicUrl -Value (Get-PropertyOrDefault -Object $record -Name "managedPackageUrl" -DefaultValue "")) -Severity "action-required" -Detail "managedPackageUrl must point to public managed package evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-package-url" -Passed (Test-PublicUrl -Value (Get-PropertyOrDefault -Object $record -Name "runtimePackageUrl" -DefaultValue "")) -Severity "action-required" -Detail "runtimePackageUrl must point to public runtime package evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-direct-nupkg-reference" -Passed (-not $consumerProjectFlags.usesDirectNupkg) -Severity "action-required" -Detail "Direct .nupkg references cannot be used as public proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-nupkg-not-dry-run-artifact" -Passed (Test-ManagedNupkgPathIsNotDryRunArtifact -ManagedPath (Get-PropertyOrDefault -Object $record -Name "managedNupkgPath" -DefaultValue "") -DryRunPath $packageDryRunArtifactPath) -Severity "action-required" -Detail "managedNupkgPath must be the public package download evidence path, not the GitHub Actions package-managed dry-run artifact path.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-nupkg-sha256-format" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $record -Name "managedNupkgSha256" -DefaultValue "")) -Severity "action-required" -Detail "managedNupkgSha256 must be a 64-character SHA256 hash.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-nupkg-sha256-format" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $record -Name "runtimeNupkgSha256" -DefaultValue "")) -Severity "action-required" -Detail "runtimeNupkgSha256 must be a 64-character SHA256 hash.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-log-sha256-format" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $record -Name "smokeLogSha256" -DefaultValue "")) -Severity "action-required" -Detail "smokeLogSha256 must be a 64-character SHA256 hash.")) | Out-Null
$items.Add((New-ValidationItem -Id "managed-nupkg-hash-match" -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name "managedNupkgPath" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "managedNupkgSha256" -DefaultValue "")) -Severity "action-required" -Detail "managedNupkgPath must exist and match managedNupkgSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-nupkg-hash-match" -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name "runtimeNupkgPath" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "runtimeNupkgSha256" -DefaultValue "")) -Severity "action-required" -Detail "runtimeNupkgPath must exist and match runtimeNupkgSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-log-hash-match" -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name "smokeLogPath" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "smokeLogSha256" -DefaultValue "")) -Severity "action-required" -Detail "smokeLogPath must exist and match smokeLogSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-key-ready" -Passed (-not (Test-IsPlaceholder -Value $runtimePackageKey)) -Severity "action-required" -Detail "runtimePackageKey must be real and match the target runtime package key.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-command-runtime-key" -Passed ($smokeCommand.Contains("--runtime-package-key", [StringComparison]::Ordinal) -and -not (Test-IsPlaceholder -Value $runtimePackageKey) -and $smokeCommand.Contains($runtimePackageKey, [StringComparison]::Ordinal)) -Severity "action-required" -Detail "smokeCommand must include --runtime-package-key and the runtimePackageKey value.")) | Out-Null

foreach ($field in @("hostOs", "hostArchitecture", "cudaDriverVersion", "cudaRuntimeVersion", "tensorRtVersion", "stdoutSummary", "stderrSummary")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be real non-placeholder owner input.")) | Out-Null
}

foreach ($field in @("ownerName", "machineName", "gpuName", "cudaDriverSupportedRuntime", "cudnnVersion", "tensorRtLine", "restoreCommand", "buildCommand", "failureDiagnostic")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be real non-placeholder owner input.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "exit-code-zero" -Passed (Test-IntZero -Value (Get-PropertyOrDefault -Object $record -Name "exitCode" -DefaultValue "")) -Severity "action-required" -Detail "exitCode must be the real package consumer smoke process exit code and must be 0.")) | Out-Null
$items.Add((New-ValidationItem -Id "started-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $record -Name "startedAtUtc" -DefaultValue "")) -Severity "action-required" -Detail "startedAtUtc must be a real parseable DateTimeOffset value.")) | Out-Null
$items.Add((New-ValidationItem -Id "finished-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $record -Name "finishedAtUtc" -DefaultValue "")) -Severity "action-required" -Detail "finishedAtUtc must be a real parseable DateTimeOffset value.")) | Out-Null
$items.Add((New-ValidationItem -Id "dependency-probe-status-passed" -Passed (Test-ValueInSet -Value (Get-PropertyOrDefault -Object $record -Name "dependencyProbeStatus" -DefaultValue "") -AllowedValues @("passed", "compatible-host-passed")) -Severity "action-required" -Detail "dependencyProbeStatus must be passed or compatible-host-passed.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-status-passed" -Passed (Test-ValueInSet -Value (Get-PropertyOrDefault -Object $record -Name "smokeStatus" -DefaultValue "") -AllowedValues @("passed")) -Severity "action-required" -Detail "smokeStatus must be passed.")) | Out-Null
$items.Add((New-ValidationItem -Id "native-assets-copied-true" -Passed (Test-BoolTrue -Value (Get-PropertyOrDefault -Object $record -Name "nativeAssetsCopied" -DefaultValue "")) -Severity "action-required" -Detail "nativeAssetsCopied must be true for package consumer runtime proof owner input.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-runner-not-queued" -Passed (Test-RunnerQueueCompleted -Value (Get-PropertyOrDefault -Object $record -Name "sourceRunnerQueueStatus" -DefaultValue "")) -Severity "action-required" -Detail "sourceRunnerQueueStatus must be completed/not-queued. queued is owner-infra-action only and not proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-runner-infrastructure-ready" -Passed (Test-RunnerInfrastructureReady -Value (Get-PropertyOrDefault -Object $record -Name "sourceRunnerInfrastructureStatus" -DefaultValue "")) -Severity "action-required" -Detail "sourceRunnerInfrastructureStatus must be available/ready/not-required. missing self-hosted runner is owner-infra-action only and not proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-runner-owner-action-boundary" -Passed (([string](Get-PropertyOrDefault -Object $record -Name "sourceRunnerOwnerAction" -DefaultValue "")).Contains("owner-infra-action", [StringComparison]::OrdinalIgnoreCase)) -Severity "action-required" -Detail "sourceRunnerOwnerAction must explicitly preserve queued/missing-runner states as owner-infra-action, not CI proof.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$passedItemIds = @($items | Where-Object { $_.passed } | ForEach-Object { [string]$_.id })
$cleanOwnerInputReady = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$ownerInputForbiddenSubstituteFree = $passedItemIds -contains "clean-root-outside-repository" -and
  $passedItemIds -contains "consumer-project-exists" -and
  $passedItemIds -contains "no-project-reference-to-repository" -and
  $passedItemIds -contains "public-package-source-kind" -and
  $passedItemIds -contains "public-package-source-not-local" -and
  $passedItemIds -contains "public-package-feed-url" -and
  $passedItemIds -contains "managed-package-url" -and
  $passedItemIds -contains "runtime-package-url" -and
  $passedItemIds -contains "no-direct-nupkg-reference" -and
  $passedItemIds -contains "managed-nupkg-not-dry-run-artifact"
$ownerInputHashFieldsReady = $passedItemIds -contains "managed-nupkg-sha256-format" -and
  $passedItemIds -contains "runtime-nupkg-sha256-format" -and
  $passedItemIds -contains "smoke-log-sha256-format"
$ownerInputPackageHashFilesMatch = $passedItemIds -contains "managed-nupkg-hash-match" -and
  $passedItemIds -contains "runtime-nupkg-hash-match"
$ownerInputSmokeLogReady = $passedItemIds -contains "smoke-command-runtime-key" -and
  $passedItemIds -contains "exit-code-zero" -and
  $passedItemIds -contains "dependency-probe-status-passed" -and
  $passedItemIds -contains "smoke-status-passed" -and
  $passedItemIds -contains "native-assets-copied-true" -and
  $passedItemIds -contains "smoke-log-hash-match"
$ownerInputHostMetadataReady = $passedItemIds -contains "field-ownerName" -and
  $passedItemIds -contains "field-machineName" -and
  $passedItemIds -contains "field-hostOs" -and
  $passedItemIds -contains "field-hostArchitecture" -and
  $passedItemIds -contains "field-gpuName" -and
  $passedItemIds -contains "field-cudaDriverVersion" -and
  $passedItemIds -contains "field-cudaDriverSupportedRuntime" -and
  $passedItemIds -contains "field-cudaRuntimeVersion" -and
  $passedItemIds -contains "field-cudnnVersion" -and
  $passedItemIds -contains "field-tensorRtVersion" -and
  $passedItemIds -contains "field-tensorRtLine"
$ownerInputCommandEvidenceReady = $passedItemIds -contains "field-restoreCommand" -and
  $passedItemIds -contains "field-buildCommand" -and
  $passedItemIds -contains "started-at-utc-parseable" -and
  $passedItemIds -contains "finished-at-utc-parseable" -and
  $passedItemIds -contains "field-stdoutSummary" -and
  $passedItemIds -contains "field-stderrSummary"
$ownerInputRunnerInfrastructureReady = $passedItemIds -contains "source-runner-not-queued" -and
  $passedItemIds -contains "source-runner-infrastructure-ready" -and
  $passedItemIds -contains "source-runner-owner-action-boundary"
$ownerInputBlockedReasons = @($items | Where-Object { -not $_.passed } | ForEach-Object { [string]$_.id })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "owner-input-ready-for-candidate-overlay"
}
else {
  "blocked-owner-input-required"
}

$validation = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-owner-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  isValidOwnerInputShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  cleanOwnerInputReady = $cleanOwnerInputReady
  ownerInputForbiddenSubstituteFree = $ownerInputForbiddenSubstituteFree
  ownerInputHashFieldsReady = $ownerInputHashFieldsReady
  ownerInputPackageHashFilesMatch = $ownerInputPackageHashFilesMatch
  ownerInputSmokeLogReady = $ownerInputSmokeLogReady
  ownerInputHostMetadataReady = $ownerInputHostMetadataReady
  ownerInputCommandEvidenceReady = $ownerInputCommandEvidenceReady
  ownerInputRunnerInfrastructureReady = $ownerInputRunnerInfrastructureReady
  sourceGitHubActionsRunEvidenceImportPresent = $sourceEvidenceImportPresent
  sourceGitHubActionsDryRunPackClaimReady = $sourceEvidenceDryRunClaim
  sourceGitHubActionsRunId = $sourceGitHubActionsRunId
  sourceHeadSha = $sourceHeadSha
  packageDryRunArtifactPath = $packageDryRunArtifactPath
  packageDryRunManagedNupkgSha256 = $packageDryRunManagedNupkgSha256
  packageDryRunCanClaimPack = $packageDryRunCanClaimPack
  dryRunOnlyNotProof = $isDryRunOnly
  isPublishedPackageProof = $isPublishedPackageProof
  isPackageConsumerRuntimeProof = $isPackageConsumerRuntimeProof
  ownerInputCanPromoteRuntimeProof = $false
  ownerInputBlockedReason = if ($ownerInputBlockedReasons.Count -eq 0) { "none" } else { $ownerInputBlockedReasons -join "; " }
  consumerProjectExists = $consumerProjectFlags.projectExists
  consumerProjectUsesProjectReference = $consumerProjectFlags.usesProjectReference
  consumerProjectUsesLocalFeed = $consumerProjectFlags.usesLocalFeed
  consumerProjectUsesDirectNupkg = $consumerProjectFlags.usesDirectNupkg
  sourceRunnerQueueStatus = [string](Get-PropertyOrDefault -Object $record -Name "sourceRunnerQueueStatus" -DefaultValue "")
  sourceRunnerInfrastructureStatus = [string](Get-PropertyOrDefault -Object $record -Name "sourceRunnerInfrastructureStatus" -DefaultValue "")
  sourceRunnerOwnerAction = [string](Get-PropertyOrDefault -Object $record -Name "sourceRunnerOwnerAction" -DefaultValue "")
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates owner input readiness only. It cannot publish packages, close the release issue, or promote runtime proof."
}

$jsonPath = Join-Path $OutputRoot "package-consumer-runtime-proof-owner-input-validation.json"
$markdownPath = Join-Path $OutputRoot "package-consumer-runtime-proof-owner-input-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Package Consumer Runtime Proof Owner Input Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| isValidOwnerInputShape | ``$($validation.isValidOwnerInputShape)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| cleanOwnerInputReady | ``$($validation.cleanOwnerInputReady)`` |
| ownerInputForbiddenSubstituteFree | ``$($validation.ownerInputForbiddenSubstituteFree)`` |
| ownerInputHashFieldsReady | ``$($validation.ownerInputHashFieldsReady)`` |
| ownerInputPackageHashFilesMatch | ``$($validation.ownerInputPackageHashFilesMatch)`` |
| ownerInputSmokeLogReady | ``$($validation.ownerInputSmokeLogReady)`` |
| ownerInputHostMetadataReady | ``$($validation.ownerInputHostMetadataReady)`` |
| ownerInputCommandEvidenceReady | ``$($validation.ownerInputCommandEvidenceReady)`` |
| sourceGitHubActionsRunEvidenceImportPresent | ``$($validation.sourceGitHubActionsRunEvidenceImportPresent)`` |
| sourceGitHubActionsDryRunPackClaimReady | ``$($validation.sourceGitHubActionsDryRunPackClaimReady)`` |
| packageDryRunCanClaimPack | ``$($validation.packageDryRunCanClaimPack)`` |
| dryRunOnlyNotProof | ``$($validation.dryRunOnlyNotProof)`` |
| isPublishedPackageProof | ``$($validation.isPublishedPackageProof)`` |
| isPackageConsumerRuntimeProof | ``$($validation.isPackageConsumerRuntimeProof)`` |
| ownerInputCanPromoteRuntimeProof | ``$($validation.ownerInputCanPromoteRuntimeProof)`` |
| canPromoteProof | ``$($validation.canPromoteProof)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Package consumer runtime proof owner input validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Package consumer runtime proof owner input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount) PerformsPublish=$($validation.performsPublish) CanPublishPublicly=$($validation.canPublishPublicly) CanCloseReleaseIssue=$($validation.canCloseReleaseIssue)"
