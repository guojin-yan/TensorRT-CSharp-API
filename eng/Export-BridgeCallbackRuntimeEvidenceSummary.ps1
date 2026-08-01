[CmdletBinding()]
param(
  [string[]]$SourceRuntimeKey = @(
    "win-x64-trt10.11-cuda12.9-cudnn9.22",
    "win-x64-trt11.0-cuda12.9-cudnn9.22"),
  [string]$ReportRoot,
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else {
  $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
}

function Resolve-PathFromRepository {
  param(
    [string]$Value,
    [Parameter(Mandatory = $true)]
    [string]$DefaultRelativePath
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $DefaultRelativePath))
  }

  if ([IO.Path]::IsPathRooted($Value)) {
    return [IO.Path]::GetFullPath($Value)
  }

  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Value))
}

function Expand-KeyList {
  param([string[]]$Values)

  return @(
    foreach ($value in @($Values)) {
      foreach ($token in @(([string]$value) -split '[,;]')) {
        if (-not [string]::IsNullOrWhiteSpace($token)) {
          $token.Trim()
        }
      }
    }
  ) | Select-Object -Unique
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [Parameter(Mandatory = $true)][string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object -or $Object.PSObject.Properties.Name -notcontains $Name) {
    return $DefaultValue
  }

  return $Object.$Name
}

function Get-ArtifactPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  $fullPath = [IO.Path]::GetFullPath($Path)
  $repositoryPrefix = $RepositoryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
  if ($fullPath.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    return $fullPath.Substring($repositoryPrefix.Length).Replace('\', '/')
  }

  return $fullPath
}

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)

  $stream = [IO.File]::OpenRead($Path)
  try {
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
      return ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
    }
    finally {
      $sha256.Dispose()
    }
  }
  finally {
    $stream.Dispose()
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return (([string]$Value -replace '\|', '\|') -replace "(`r`n|`n|`r)", "<br>")
}

function Write-Utf8File {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [AllowNull()][object]$Content
  )

  $directory = Split-Path -Parent $Path
  New-Item -ItemType Directory -Path $directory -Force | Out-Null
  $text = @($Content) -join [Environment]::NewLine
  [IO.File]::WriteAllText($Path, $text + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
}

$runtimeKeys = @(Expand-KeyList -Values $SourceRuntimeKey)
if ($runtimeKeys.Count -eq 0) {
  throw "At least one source runtime key is required."
}

$ReportRoot = Resolve-PathFromRepository -Value $ReportRoot -DefaultRelativePath "artifacts\package-consumer\bridge-runtime"
$OutputRoot = Resolve-PathFromRepository -Value $OutputRoot -DefaultRelativePath "artifacts\final-release"
$rows = [Collections.Generic.List[object]]::new()
$findings = [Collections.Generic.List[string]]::new()

foreach ($runtimeKey in $runtimeKeys) {
  $reportPath = Join-Path (Join-Path $ReportRoot $runtimeKey) "bridge-package-runtime-consumer-proof.json"
  $rowFindings = [Collections.Generic.List[string]]::new()
  $report = $null

  if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
    $rowFindings.Add("missing-report")
  }
  else {
    try {
      $report = Get-Content -LiteralPath $reportPath -Raw -Encoding utf8 | ConvertFrom-Json
    }
    catch {
      $rowFindings.Add("invalid-json: $($_.Exception.Message)")
    }
  }

  $callbackState = Get-PropertyOrDefault -Object $report -Name "callbackStateSnapshot" -DefaultValue $null
  $callback = Get-PropertyOrDefault -Object $report -Name "debugListenerCallback" -DefaultValue $null
  $proofScopes = Get-PropertyOrDefault -Object $report -Name "proofScopes" -DefaultValue $null
  $sourceTreeScope = Get-PropertyOrDefault -Object $proofScopes -Name "sourceTree" -DefaultValue $null
  $localPackageScope = Get-PropertyOrDefault -Object $proofScopes -Name "localPackage" -DefaultValue $null
  $publicPackageScope = Get-PropertyOrDefault -Object $proofScopes -Name "publicPackage" -DefaultValue $null
  $postPublishScope = Get-PropertyOrDefault -Object $proofScopes -Name "postPublish" -DefaultValue $null

  $sourceRuntimeKey = [string](Get-PropertyOrDefault -Object $report -Name "sourceRuntimeKey" -DefaultValue "")
  $smokeStatus = [string](Get-PropertyOrDefault -Object $report -Name "smokeStatus" -DefaultValue "missing")
  $runtimeExecutionProof = [bool](Get-PropertyOrDefault -Object $report -Name "isRuntimeExecutionProof" -DefaultValue $false)
  $packageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $report -Name "isPackageConsumerRuntimeProof" -DefaultValue $false)
  $rootLocalCallbackProof = [bool](Get-PropertyOrDefault -Object $report -Name "isLocalPackageDebugListenerCallbackRuntimeProof" -DefaultValue $false)
  $canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $report -Name "canPromoteRuntimeProof" -DefaultValue $false)
  $canPublishPublicly = [bool](Get-PropertyOrDefault -Object $report -Name "canPublishPublicly" -DefaultValue $false)
  $canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $report -Name "canCloseReleaseIssue" -DefaultValue $false)

  $callbackStateObserved = [bool](Get-PropertyOrDefault -Object $callbackState -Name "observed" -DefaultValue $false)
  $callbackStateComplete = [bool](Get-PropertyOrDefault -Object $callbackState -Name "complete" -DefaultValue $false)
  $callbackStateCoherent = [bool](Get-PropertyOrDefault -Object $callbackState -Name "coherent" -DefaultValue $false)
  $callbackStatePointerFree = [bool](Get-PropertyOrDefault -Object $callbackState -Name "pointerFree" -DefaultValue $false)
  $callbackStateLastStatus = [string](Get-PropertyOrDefault -Object $callbackState -Name "lastStatus" -DefaultValue "")
  $callbackStateLastOperation = [string](Get-PropertyOrDefault -Object $callbackState -Name "lastOperation" -DefaultValue "")

  $callbackStatus = [string](Get-PropertyOrDefault -Object $callback -Name "status" -DefaultValue "missing")
  $invocationCount = [long](Get-PropertyOrDefault -Object $callback -Name "invocationCount" -DefaultValue 0)
  $failureCount = [long](Get-PropertyOrDefault -Object $callback -Name "failureCount" -DefaultValue 0)
  $inFlightCallbackCount = [long](Get-PropertyOrDefault -Object $callback -Name "inFlightCallbackCount" -DefaultValue 0)
  $callbackAttached = [bool](Get-PropertyOrDefault -Object $callback -Name "attached" -DefaultValue $false)
  $nativeVTableInstalled = [bool](Get-PropertyOrDefault -Object $callback -Name "nativeVTableInstalled" -DefaultValue $false)
  $callbackDetached = [bool](Get-PropertyOrDefault -Object $callback -Name "detached" -DefaultValue $false)
  $clearReturned = [bool](Get-PropertyOrDefault -Object $callback -Name "clearReturned" -DefaultValue $false)
  $callbackRuntimeProof = [bool](Get-PropertyOrDefault -Object $callback -Name "isRealCallbackRuntimeProof" -DefaultValue $false)
  $callbackLocalPackageProof = [bool](Get-PropertyOrDefault -Object $callback -Name "isLocalPackageCallbackRuntimeProof" -DefaultValue $false)
  $sourceTreeProof = [bool](Get-PropertyOrDefault -Object $sourceTreeScope -Name "isProof" -DefaultValue $false)
  $localPackageProof = [bool](Get-PropertyOrDefault -Object $localPackageScope -Name "isProof" -DefaultValue $false)
  $publicPackageProof = [bool](Get-PropertyOrDefault -Object $publicPackageScope -Name "isProof" -DefaultValue $false)
  $postPublishProof = [bool](Get-PropertyOrDefault -Object $postPublishScope -Name "isProof" -DefaultValue $false)

  if ($null -ne $report) {
    if (-not [string]::Equals($sourceRuntimeKey, $runtimeKey, [StringComparison]::Ordinal)) { $rowFindings.Add("source-runtime-key-mismatch") }
    if ($smokeStatus -ne "passed") { $rowFindings.Add("smoke-not-passed") }
    if (-not $runtimeExecutionProof) { $rowFindings.Add("runtime-execution-proof-not-observed") }
    if ($packageConsumerRuntimeProof) { $rowFindings.Add("local-report-promoted-to-package-consumer-proof") }
    if (-not $rootLocalCallbackProof) { $rowFindings.Add("root-local-callback-proof-missing") }
    if (-not $callbackStateObserved) { $rowFindings.Add("callback-state-not-observed") }
    if (-not $callbackStateCoherent) { $rowFindings.Add("callback-state-incoherent") }
    if (-not $callbackStatePointerFree) { $rowFindings.Add("callback-state-not-pointer-free") }
    if ($callbackStatus -ne "passed") { $rowFindings.Add("callback-status-not-passed") }
    if ($invocationCount -le 0) { $rowFindings.Add("callback-not-invoked") }
    if ($failureCount -ne 0) { $rowFindings.Add("callback-failure-count-nonzero") }
    if ($inFlightCallbackCount -ne 0) { $rowFindings.Add("callback-inflight-count-nonzero") }
    if (-not $callbackAttached) { $rowFindings.Add("callback-not-attached") }
    if (-not $nativeVTableInstalled) { $rowFindings.Add("native-vtable-not-installed") }
    if (-not $callbackDetached) { $rowFindings.Add("callback-not-detached") }
    if (-not $clearReturned) { $rowFindings.Add("callback-clear-did-not-return") }
    if (-not $callbackRuntimeProof) { $rowFindings.Add("callback-runtime-proof-missing") }
    if (-not $callbackLocalPackageProof) { $rowFindings.Add("callback-local-package-proof-missing") }
    if ($sourceTreeProof) { $rowFindings.Add("source-tree-scope-must-remain-false") }
    if (-not $localPackageProof) { $rowFindings.Add("local-package-scope-missing") }
    if ($publicPackageProof) { $rowFindings.Add("public-package-scope-must-remain-false") }
    if ($postPublishProof) { $rowFindings.Add("post-publish-scope-must-remain-false") }
    if ($canPromoteRuntimeProof) { $rowFindings.Add("release-runtime-promotion-must-remain-false") }
    if ($canPublishPublicly) { $rowFindings.Add("public-publish-must-remain-false") }
    if ($canCloseReleaseIssue) { $rowFindings.Add("release-close-must-remain-false") }
  }

  $valid = $null -ne $report -and $rowFindings.Count -eq 0
  foreach ($finding in $rowFindings) {
    $findings.Add("$runtimeKey/$finding")
  }

  $rows.Add([pscustomobject][ordered]@{
      runtimeKey = $runtimeKey
      status = if ($valid) { "local-package-callback-runtime-observed" } elseif ($null -eq $report) { "missing-report" } else { "invalid-local-package-callback-runtime-evidence" }
      valid = $valid
      reportPath = Get-ArtifactPath -Path $reportPath
      reportSha256 = if ($null -ne $report) { Get-Sha256 -Path $reportPath } else { "" }
      sourceRuntimeKey = $sourceRuntimeKey
      smokeStatus = $smokeStatus
      sourceReportRuntimeExecutionProof = $runtimeExecutionProof
      sourceReportPackageConsumerRuntimeProof = $packageConsumerRuntimeProof
      sourceReportLocalCallbackRuntimeProof = $rootLocalCallbackProof
      callbackStateObserved = $callbackStateObserved
      callbackStateComplete = $callbackStateComplete
      callbackStateCoherent = $callbackStateCoherent
      callbackStatePointerFree = $callbackStatePointerFree
      callbackStateLastStatus = $callbackStateLastStatus
      callbackStateLastOperation = $callbackStateLastOperation
      callbackStatus = $callbackStatus
      invocationCount = $invocationCount
      failureCount = $failureCount
      inFlightCallbackCount = $inFlightCallbackCount
      attached = $callbackAttached
      nativeVTableInstalled = $nativeVTableInstalled
      detached = $callbackDetached
      clearReturned = $clearReturned
      callbackRuntimeProof = $callbackRuntimeProof
      callbackLocalPackageProof = $callbackLocalPackageProof
      sourceTreeProof = $sourceTreeProof
      localPackageProof = $localPackageProof
      publicPackageProof = $publicPackageProof
      postPublishProof = $postPublishProof
      findingCount = $rowFindings.Count
      findings = @($rowFindings.ToArray())
    })
}

$validRowCount = @($rows | Where-Object { [bool]$_.valid }).Count
$allRequiredLocalPackageCallbackProofObserved = $rows.Count -eq $runtimeKeys.Count -and $validRowCount -eq $runtimeKeys.Count
$result = [ordered]@{
  schemaVersion = 1
  recordKind = "bridge-callback-runtime-evidence-summary"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  summaryState = if ($allRequiredLocalPackageCallbackProofObserved) { "local-package-callback-runtime-observed" } else { "blocked-local-package-callback-runtime-evidence-incomplete" }
  evidenceScope = "local-package"
  requiredRuntimeKeys = @($runtimeKeys)
  requiredRuntimeKeyCount = $runtimeKeys.Count
  observedRuntimeKeyCount = $validRowCount
  allRequiredLocalPackageCallbackProofObserved = $allRequiredLocalPackageCallbackProofObserved
  sourceReportsContainRuntimeExecutionProof = $allRequiredLocalPackageCallbackProofObserved
  sourceReportsContainLocalPackageCallbackRuntimeProof = $allRequiredLocalPackageCallbackProofObserved
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPublicPackageProof = $false
  isPostPublishProof = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  findingCount = $findings.Count
  findings = @($findings.ToArray())
  rows = @($rows.ToArray())
  proofBoundary = "This read-only summary observes local PackageReference-only callback runtime reports. It does not execute callbacks and cannot promote local package evidence to public-package, post-publish, publication, or release-close proof."
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$jsonPath = Join-Path $OutputRoot "bridge-callback-runtime-evidence-summary.json"
$markdownPath = Join-Path $OutputRoot "bridge-callback-runtime-evidence-summary.md"
Write-Utf8File -Path $jsonPath -Content ($result | ConvertTo-Json -Depth 10)

$lines = [Collections.Generic.List[string]]::new()
$lines.Add("# Bridge Callback Runtime Evidence Summary")
$lines.Add("")
$lines.Add("- state: ``$($result.summaryState)``")
$lines.Add("- evidence scope: ``$($result.evidenceScope)``")
$lines.Add("- observed runtime keys: ``$($result.observedRuntimeKeyCount)/$($result.requiredRuntimeKeyCount)``")
$lines.Add("- source reports contain local callback proof: ``$($result.sourceReportsContainLocalPackageCallbackRuntimeProof)``")
$lines.Add("- this summary is runtime/public/post-publish proof: ``$($result.isRuntimeExecutionProof)/$($result.isPublicPackageProof)/$($result.isPostPublishProof)``")
$lines.Add("- findings: ``$($result.findingCount)``")
$lines.Add("")
$lines.Add("## Runtime Lines")
$lines.Add("")
$lines.Add("| Runtime key | State | Callback state | Invocation/failure/in-flight | Local/public/post-publish | Findings |")
$lines.Add("| --- | --- | --- | --- | --- | --- |")
foreach ($row in $rows) {
  $lines.Add("| ``$($row.runtimeKey)`` | ``$($row.status)`` | observed=$($row.callbackStateObserved); coherent=$($row.callbackStateCoherent); pointerFree=$($row.callbackStatePointerFree); complete=$($row.callbackStateComplete) | $($row.invocationCount)/$($row.failureCount)/$($row.inFlightCallbackCount) | $($row.localPackageProof)/$($row.publicPackageProof)/$($row.postPublishProof) | $(ConvertTo-MarkdownCell -Value ($row.findings -join '; ')) |")
}
$lines.Add("")
$lines.Add("## Proof Boundary")
$lines.Add("")
$lines.Add($result.proofBoundary)
Write-Utf8File -Path $markdownPath -Content $lines

Write-Host "Bridge callback runtime evidence summary written:"
Write-Host "  JSON=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "SummaryState=$($result.summaryState) Observed=$validRowCount/$($runtimeKeys.Count) Findings=$($findings.Count)"
