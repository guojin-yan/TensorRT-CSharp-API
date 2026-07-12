[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\clean-external-consumer-smoke-input.template.json",
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

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
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

function Test-DateTimeOffsetFormat {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed)
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

function Get-ConsumerProjectFlags {
  param([AllowNull()][object]$ProjectPath)

  $result = [ordered]@{
    projectExists = $false
    outsideRepository = $false
    usesProjectReference = $true
    usesSrcPath = $true
    usesRepositoryAbsolutePath = $true
    usesLocalRestoreSource = $true
    usesDirectNupkg = $true
  }

  $pathText = [string]$ProjectPath
  if (Test-IsPlaceholder -Value $pathText) {
    return [pscustomobject]$result
  }

  $result.outsideRepository = Test-IsOutsideRepository -Path $pathText
  $resolvedPath = Resolve-InputPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return [pscustomobject]$result
  }

  $result.projectExists = $true
  $content = Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8
  $repoPath = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
  $repoSlash = $repoPath.Replace("\", "/")
  $result.usesProjectReference = $content.Contains("<ProjectReference", [StringComparison]::OrdinalIgnoreCase)
  $result.usesSrcPath = $content.Contains("..\src\", [StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains("../src/", [StringComparison]::OrdinalIgnoreCase)
  $result.usesRepositoryAbsolutePath = $content.Contains($repoPath, [StringComparison]::OrdinalIgnoreCase) -or
    $content.Contains($repoSlash, [StringComparison]::OrdinalIgnoreCase)
  $result.usesLocalRestoreSource = $content.Contains("RestoreSources", [StringComparison]::OrdinalIgnoreCase) -and
    ($content.Contains("artifacts", [StringComparison]::OrdinalIgnoreCase) -or
     $content.Contains("local", [StringComparison]::OrdinalIgnoreCase) -or
     $content.Contains("package-managed-dry-run", [StringComparison]::OrdinalIgnoreCase))
  $result.usesDirectNupkg = $content.Contains(".nupkg", [StringComparison]::OrdinalIgnoreCase)
  return [pscustomobject]$result
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Clean external consumer smoke input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$performsPublish = Get-BoolPropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true
$usesPublishToken = Get-BoolPropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true
$canPublishPublicly = Get-BoolPropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true
$canPublishGitHubPackages = Get-BoolPropertyOrDefault -Object $record -Name "canPublishGitHubPackages" -DefaultValue $true
$canCloseReleaseIssue = Get-BoolPropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true
$canClaimRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "canClaimRuntimeProof" -DefaultValue $true
$canClaimPackageConsumerRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "canClaimPackageConsumerRuntimeProof" -DefaultValue $true
$isRuntimeExecutionProof = Get-BoolPropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true
$isPackageConsumerRuntimeProof = Get-BoolPropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true
$isPostPublishProof = Get-BoolPropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true
$runtimePackageKey = [string](Get-PropertyOrDefault -Object $record -Name "runtimePackageKey" -DefaultValue "")
$smokeCommand = [string](Get-PropertyOrDefault -Object $record -Name "smokeCommand" -DefaultValue "")
$projectFlags = Get-ConsumerProjectFlags -ProjectPath (Get-PropertyOrDefault -Object $record -Name "consumerProjectPath" -DefaultValue "")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "clean-external-consumer-smoke-input") -Severity "blocker" -Detail "recordKind must be clean-external-consumer-smoke-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $usesPublishToken -and -not $canPublishPublicly -and -not $canPublishGitHubPackages -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Clean external consumer smoke validation must not publish, use token, approve publication, or close issues.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-claims-false" -Passed (-not $canClaimRuntimeProof -and -not $canClaimPackageConsumerRuntimeProof -and -not $isRuntimeExecutionProof -and -not $isPackageConsumerRuntimeProof -and -not $isPostPublishProof) -Severity "blocker" -Detail "Input template and validation output are not proof claims.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-root-outside-repository" -Passed (Test-IsOutsideRepository -Path (Get-PropertyOrDefault -Object $record -Name "cleanExternalConsumerRoot" -DefaultValue "")) -Severity "action-required" -Detail "cleanExternalConsumerRoot must be outside this repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "consumer-project-exists" -Passed $projectFlags.projectExists -Severity "action-required" -Detail "consumerProjectPath must point to an existing external .csproj.")) | Out-Null
$items.Add((New-ValidationItem -Id "consumer-project-outside-repository" -Passed $projectFlags.outsideRepository -Severity "action-required" -Detail "consumerProjectPath must be outside this repository.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-project-reference" -Passed (-not $projectFlags.usesProjectReference) -Severity "action-required" -Detail "Clean consumer project must not use ProjectReference.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-src-path-reference" -Passed (-not $projectFlags.usesSrcPath) -Severity "action-required" -Detail "Clean consumer project must not reference ../src or ..\\src.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-repository-absolute-path" -Passed (-not $projectFlags.usesRepositoryAbsolutePath) -Severity "action-required" -Detail "Clean consumer project must not contain this repository absolute path.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-local-restore-source" -Passed (-not $projectFlags.usesLocalRestoreSource) -Severity "action-required" -Detail "Clean consumer project RestoreSources must not point to local/artifacts/dry-run feeds.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-direct-nupkg-reference" -Passed (-not $projectFlags.usesDirectNupkg) -Severity "action-required" -Detail "Clean consumer project must not use direct .nupkg references.")) | Out-Null
$items.Add((New-ValidationItem -Id "restore-command-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "restoreCommand" -DefaultValue ""))) -Severity "action-required" -Detail "restoreCommand must be filled.")) | Out-Null
$items.Add((New-ValidationItem -Id "build-command-present" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "buildCommand" -DefaultValue ""))) -Severity "action-required" -Detail "buildCommand must be filled.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-command-runtime-key" -Passed (-not (Test-IsPlaceholder -Value $runtimePackageKey) -and $smokeCommand.Contains("--runtime-package-key", [StringComparison]::Ordinal) -and $smokeCommand.Contains($runtimePackageKey, [StringComparison]::Ordinal)) -Severity "action-required" -Detail "smokeCommand must include --runtime-package-key and the runtimePackageKey value.")) | Out-Null
$items.Add((New-ValidationItem -Id "exit-code-zero" -Passed (Test-IntZero -Value (Get-PropertyOrDefault -Object $record -Name "exitCode" -DefaultValue "")) -Severity "action-required" -Detail "exitCode must be 0.")) | Out-Null
$items.Add((New-ValidationItem -Id "started-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $record -Name "startedAtUtc" -DefaultValue "")) -Severity "action-required" -Detail "startedAtUtc must be parseable.")) | Out-Null
$items.Add((New-ValidationItem -Id "finished-at-utc-parseable" -Passed (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $record -Name "finishedAtUtc" -DefaultValue "")) -Severity "action-required" -Detail "finishedAtUtc must be parseable.")) | Out-Null

foreach ($field in @("stdoutLog", "stderrLog", "runtimeProbeReport")) {
  $pathName = "${field}Path"
  $shaName = "${field}Sha256"
  $items.Add((New-ValidationItem -Id "$field-sha256-format" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $record -Name $shaName -DefaultValue "")) -Severity "action-required" -Detail "$shaName must be a 64-character SHA256.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$field-hash-match" -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name $pathName -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name $shaName -DefaultValue "")) -Severity "action-required" -Detail "$pathName must exist and match $shaName.")) | Out-Null
}

foreach ($field in @("hostOs", "hostArchitecture", "gpuName", "driverVersion", "cudaRuntimeVersion", "tensorRtVersion", "cudnnVersion")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be real host metadata.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "native-assets-copied-true" -Passed (Test-BoolTrue -Value (Get-PropertyOrDefault -Object $record -Name "nativeAssetsCopied" -DefaultValue "")) -Severity "action-required" -Detail "nativeAssetsCopied must be true.")) | Out-Null
$items.Add((New-ValidationItem -Id "dependency-probe-status-passed" -Passed (Test-ValueInSet -Value (Get-PropertyOrDefault -Object $record -Name "dependencyProbeStatus" -DefaultValue "") -AllowedValues @("passed", "compatible-host-passed")) -Severity "action-required" -Detail "dependencyProbeStatus must be passed or compatible-host-passed; build-only and dependency-probe-only are not smoke proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "smoke-status-passed" -Passed (Test-ValueInSet -Value (Get-PropertyOrDefault -Object $record -Name "smokeStatus" -DefaultValue "") -AllowedValues @("passed")) -Severity "action-required" -Detail "smokeStatus must be passed.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "clean-external-consumer-smoke-input-ready"
}
else {
  "blocked-clean-external-consumer-smoke-required"
}

$validation = [pscustomobject]@{
  recordKind = "clean-external-consumer-smoke-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  cleanExternalConsumerSmokeReady = ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0)
  consumerProjectExists = $projectFlags.projectExists
  consumerProjectOutsideRepository = $projectFlags.outsideRepository
  consumerProjectUsesProjectReference = $projectFlags.usesProjectReference
  consumerProjectUsesSrcPath = $projectFlags.usesSrcPath
  consumerProjectUsesRepositoryAbsolutePath = $projectFlags.usesRepositoryAbsolutePath
  consumerProjectUsesLocalRestoreSource = $projectFlags.usesLocalRestoreSource
  consumerProjectUsesDirectNupkg = $projectFlags.usesDirectNupkg
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canPublishGitHubPackages = $false
  canCloseReleaseIssue = $false
  canClaimRuntimeProof = $false
  canClaimPackageConsumerRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates clean external consumer smoke input only. It rejects in-repository samples, ProjectReference, source path leakage, local RestoreSources, direct nupkg shortcuts, build-only results, and dependency-probe-only substitutes."
}

$jsonPath = Join-Path $OutputRoot "clean-external-consumer-smoke-input-validation.json"
$markdownPath = Join-Path $OutputRoot "clean-external-consumer-smoke-input-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Clean External Consumer Smoke Input Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| cleanExternalConsumerSmokeReady | ``$($validation.cleanExternalConsumerSmokeReady)`` |
| consumerProjectOutsideRepository | ``$($validation.consumerProjectOutsideRepository)`` |
| consumerProjectUsesProjectReference | ``$($validation.consumerProjectUsesProjectReference)`` |
| consumerProjectUsesLocalRestoreSource | ``$($validation.consumerProjectUsesLocalRestoreSource)`` |
| consumerProjectUsesDirectNupkg | ``$($validation.consumerProjectUsesDirectNupkg)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| usesPublishToken | ``$($validation.usesPublishToken)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canClaimPackageConsumerRuntimeProof | ``$($validation.canClaimPackageConsumerRuntimeProof)`` |
| isPackageConsumerRuntimeProof | ``$($validation.isPackageConsumerRuntimeProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Safety Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Clean external consumer smoke input validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Clean external consumer smoke input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount) PerformsPublish=False UsesPublishToken=False CanPublishPublicly=False"
