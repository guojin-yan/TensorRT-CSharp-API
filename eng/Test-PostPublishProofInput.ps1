[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-proof-input.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { $scriptRoot = (Get-Location).Path } else { $scriptRoot = $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release" }
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-PropertyOrDefault { param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue) if ($null -eq $Object) { return $DefaultValue } if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value } return $DefaultValue }
function Get-BoolPropertyOrDefault { param([AllowNull()][object]$Object, [string]$Name, [bool]$DefaultValue) $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue; if ($value -is [bool]) { return [bool]$value }; $parsed = $false; if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) { return $parsed }; return $DefaultValue }
function New-ValidationItem { param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail) [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail } }
function Test-IsPlaceholder { param([AllowNull()][object]$Value) $text = [string]$Value; return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>" }
function Test-Sha256Format { param([AllowNull()][object]$Value) return ([string]$Value) -match "^[0-9a-fA-F]{64}$" }
function Resolve-InputPath { param([string]$Path) if ([IO.Path]::IsPathRooted($Path)) { return $Path }; return Join-Path $RepositoryRoot $Path }
function Test-DateTimeOffsetFormat { param([AllowNull()][object]$Value) if (Test-IsPlaceholder -Value $Value) { return $false }; $parsed = [DateTimeOffset]::MinValue; return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed) }
function Test-IntZero { param([AllowNull()][object]$Value) if (Test-IsPlaceholder -Value $Value) { return $false }; $parsed = 0; return [int]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed -eq 0 }
function Test-BoolTrue { param([AllowNull()][object]$Value) if (Test-IsPlaceholder -Value $Value) { return $false }; $parsed = $false; return [bool]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed }
function Test-ValueInSet { param([AllowNull()][object]$Value, [string[]]$AllowedValues) if (Test-IsPlaceholder -Value $Value) { return $false }; $text = ([string]$Value).Trim(); foreach ($allowed in $AllowedValues) { if ($text.Equals($allowed, [StringComparison]::OrdinalIgnoreCase)) { return $true } }; return $false }

function Test-HttpsNonLocalUrl {
  param([AllowNull()][object]$Value)
  $text = ([string]$Value).Trim()
  if (Test-IsPlaceholder -Value $text) { return $false }
  if (-not $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase)) { return $false }
  return -not $text.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("github-actions-runs", [StringComparison]::OrdinalIgnoreCase) -and
    -not $text.Contains("localhost", [StringComparison]::OrdinalIgnoreCase)
}

function Test-FileHashMatches {
  param([AllowNull()][object]$Path, [AllowNull()][object]$Sha256)
  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $shaText)) { return $false }
  $resolvedPath = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) { return $false }
  $actual = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

function Test-IsOutsideRepository {
  param([AllowNull()][object]$Path)
  $text = [string]$Path
  if (Test-IsPlaceholder -Value $text) { return $false }
  try {
    $repo = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $candidate = [IO.Path]::GetFullPath($text).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    return -not $candidate.StartsWith($repo, [StringComparison]::OrdinalIgnoreCase)
  }
  catch { return $false }
}

function Get-ConsumerProjectFlags {
  param([AllowNull()][object]$ProjectPath)
  $result = [ordered]@{ exists = $false; outsideRepository = $false; usesProjectReference = $true; usesSrcPath = $true; usesRepositoryAbsolutePath = $true; usesLocalRestoreSource = $true; usesDirectNupkg = $true }
  $pathText = [string]$ProjectPath
  if (Test-IsPlaceholder -Value $pathText) { return [pscustomobject]$result }
  $result.outsideRepository = Test-IsOutsideRepository -Path $pathText
  $resolved = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return [pscustomobject]$result }
  $result.exists = $true
  $content = Get-Content -LiteralPath $resolved -Raw -Encoding utf8
  $repo = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
  $repoSlash = $repo.Replace("\", "/")
  $result.usesProjectReference = $content.Contains("<ProjectReference", [StringComparison]::OrdinalIgnoreCase)
  $result.usesSrcPath = $content.Contains("..\src\", [StringComparison]::OrdinalIgnoreCase) -or $content.Contains("../src/", [StringComparison]::OrdinalIgnoreCase)
  $result.usesRepositoryAbsolutePath = $content.Contains($repo, [StringComparison]::OrdinalIgnoreCase) -or $content.Contains($repoSlash, [StringComparison]::OrdinalIgnoreCase)
  $result.usesLocalRestoreSource = $content.Contains("RestoreSources", [StringComparison]::OrdinalIgnoreCase) -and ($content.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or $content.Contains("local", [StringComparison]::OrdinalIgnoreCase) -or $content.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase))
  $result.usesDirectNupkg = $content.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
  return [pscustomobject]$result
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) { throw "Post-publish proof input not found: $resolvedInputPath" }

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$projectFlags = Get-ConsumerProjectFlags -ProjectPath (Get-PropertyOrDefault -Object $record -Name "consumerProjectPath" -DefaultValue "")
$runtimePackageKey = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageKey" -DefaultValue "")
$smokeCommand = [string](Get-PropertyOrDefault -Object $record -Name "smokeCommand" -DefaultValue "")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-proof-input") -Severity "blocker" -Detail "recordKind must be post-publish-proof-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not (Get-BoolPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not (Get-BoolPropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not (Get-BoolPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Post-publish proof validation must not publish or use tokens.")) | Out-Null

foreach ($field in @("publishedManagedPackageUrl", "publishedRuntimePackageUrl", "nugetPackageMetadataUrl", "githubPackagesMetadataUrl")) {
  $items.Add((New-ValidationItem -Id "$field-https-nonlocal" -Passed (Test-HttpsNonLocalUrl -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be HTTPS and not local/artifacts/dry-run.")) | Out-Null
}

foreach ($field in @("publishedManagedPackageSha256", "publishedRuntimePackageSha256", "downloadedManagedNupkgSha256", "downloadedRuntimeNupkgSha256")) {
  $items.Add((New-ValidationItem -Id "$field-format" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be a 64-character SHA256.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "downloaded-managed-hash-match" -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name "downloadedManagedNupkgPath" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "downloadedManagedNupkgSha256" -DefaultValue "")) -Severity "action-required" -Detail "downloadedManagedNupkgPath must exist and match downloadedManagedNupkgSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "downloaded-runtime-hash-match" -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name "downloadedRuntimeNupkgPath" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "downloadedRuntimeNupkgSha256" -DefaultValue "")) -Severity "action-required" -Detail "downloadedRuntimeNupkgPath must exist and match downloadedRuntimeNupkgSha256.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-root-outside-repository" -Passed (Test-IsOutsideRepository -Path (Get-PropertyOrDefault -Object $record -Name "cleanExternalConsumerRoot" -DefaultValue "")) -Severity "action-required" -Detail "cleanExternalConsumerRoot must be outside this repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "consumer-project-exists" -Passed $projectFlags.exists -Severity "action-required" -Detail "consumerProjectPath must exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "consumer-project-outside-repository" -Passed $projectFlags.outsideRepository -Severity "action-required" -Detail "consumerProjectPath must be outside this repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-project-reference" -Passed (-not $projectFlags.usesProjectReference) -Severity "action-required" -Detail "ProjectReference is forbidden.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-src-path-reference" -Passed (-not $projectFlags.usesSrcPath) -Severity "action-required" -Detail "../src or ..\\src references are forbidden.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-repository-absolute-path" -Passed (-not $projectFlags.usesRepositoryAbsolutePath) -Severity "action-required" -Detail "Repository absolute path is forbidden.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-local-restore-source" -Passed (-not $projectFlags.usesLocalRestoreSource) -Severity "action-required" -Detail "local/artifacts RestoreSources are forbidden.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-direct-nupkg-reference" -Passed (-not $projectFlags.usesDirectNupkg) -Severity "action-required" -Detail "direct .nupkg references are forbidden.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-command-runtime-key" -Passed (-not (Test-IsPlaceholder -Value $runtimePackageKey) -and $smokeCommand.Contains("--runtime-package-key", [StringComparison]::Ordinal) -and $smokeCommand.Contains($runtimePackageKey, [StringComparison]::Ordinal)) -Severity "action-required" -Detail "smokeCommand must include --runtime-package-key and runtimePackageKey.")) | Out-Null
$items.Add((New-ValidationItem -Id "exit-code-zero" -Passed (Test-IntZero -Value (Get-PropertyOrDefault -Object $record -Name "exitCode" -DefaultValue "")) -Severity "action-required" -Detail "exitCode must be 0.")) | Out-Null

foreach ($field in @("stdoutLog", "stderrLog", "runtimeProbeReport")) {
  $items.Add((New-ValidationItem -Id "$field-hash-match" -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name "${field}Path" -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name "${field}Sha256" -DefaultValue "")) -Severity "action-required" -Detail "${field}Path must exist and match ${field}Sha256.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "dependency-probe-status-passed" -Passed (Test-ValueInSet -Value (Get-PropertyOrDefault -Object $record -Name "dependencyProbeStatus" -DefaultValue "") -AllowedValues @("passed", "compatible-host-passed")) -Severity "action-required" -Detail "dependencyProbeStatus must be passed or compatible-host-passed.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-status-passed" -Passed (Test-ValueInSet -Value (Get-PropertyOrDefault -Object $record -Name "smokeStatus" -DefaultValue "") -AllowedValues @("passed")) -Severity "action-required" -Detail "smokeStatus must be passed.")) | Out-Null
$items.Add((New-ValidationItem -Id "native-assets-copied-true" -Passed (Test-BoolTrue -Value (Get-PropertyOrDefault -Object $record -Name "nativeAssetsCopied" -DefaultValue "")) -Severity "action-required" -Detail "nativeAssetsCopied must be true.")) | Out-Null

foreach ($field in @("publishedAtUtc", "ownerReviewedAtUtc")) {
  $items.Add((New-ValidationItem -Id "$field-parseable" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue "")) -Severity "action-required" -Detail "$field must be parseable.")) | Out-Null
}

foreach ($field in @("hostOs", "hostArchitecture", "gpuName", "driverVersion", "cudaRuntimeVersion", "tensorRtVersion", "cudnnVersion", "ownerName")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be real post-publish metadata.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$ready = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$validationState = if ($ready) { "post-publish-proof-input-ready" } else { "blocked-post-publish-proof-required" }

$validation = [pscustomobject]@{
  recordKind = "post-publish-proof-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  postPublishProofReady = $ready
  canCloseReleaseIssue = $false
  performsPublish = $false
  usesPublishToken = $false
  isPostPublishProof = $ready
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates post-publish proof input. Even ready proof does not automatically close a release issue; owner close decision remains separate."
}

$jsonPath = Join-Path $OutputRoot "post-publish-proof-input-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-proof-input-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Post-Publish Proof Input Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| postPublishProofReady | ``$($validation.postPublishProofReady)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| usesPublishToken | ``$($validation.usesPublishToken)`` |
| isPostPublishProof | ``$($validation.isPostPublishProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) { throw "Post-publish proof input validation failed with $($failedBlockers.Count) blocker(s)." }
Write-Host "Post-publish proof input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
