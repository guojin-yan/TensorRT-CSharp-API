[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [string[]]$RuntimePackageKey = @(
    "win-x64-trt8.6-cuda12.1-cudnn8.9",
    "win-x64-trt10.11-cuda12.9-cudnn9.22",
    "win-x64-trt11.0-cuda12.9-cudnn9.22"
  ),
  [string]$ManagedPackageDirectory,
  [string]$YoloVisionPackageDirectory,
  [string]$ModelPath,
  [string]$LabelsPath,
  [string]$ImagePath,
  [string]$PackageVersion = "4.0.0",
  [switch]$RequireAllRuntimePass,
  [switch]$KeepWorkspace
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$outerRoot = [IO.Path]::GetFullPath((Split-Path -Parent $RepositoryRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-PathValue {
  param([string]$Value, [string]$DefaultValue, [string]$RelativeRoot)

  $candidate = if ([string]::IsNullOrWhiteSpace($Value)) { $DefaultValue } else { $Value }
  if ([IO.Path]::IsPathRooted($candidate)) {
    return [IO.Path]::GetFullPath($candidate)
  }

  return [IO.Path]::GetFullPath((Join-Path $RelativeRoot $candidate))
}

function Assert-PathUnderRoot {
  param([string]$Path, [string]$Root, [string]$Description)

  $fullPath = [IO.Path]::GetFullPath($Path).TrimEnd('\')
  $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
  if (-not $fullPath.StartsWith($fullRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "$Description must remain under '$fullRoot': $fullPath"
  }
}

function Assert-NonCDrivePath {
  param([string]$Path, [string]$Description)

  if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)).TrimEnd('\') -ieq "C:") {
    throw "$Description must not use the C drive: $Path"
  }
}

function Remove-DirectoryWithRetry {
  param([string]$Path, [int]$Attempts = 20, [int]$DelayMilliseconds = 500)

  if (-not (Test-Path -LiteralPath $Path)) {
    return $true
  }

  for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    try {
      Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
      if (-not (Test-Path -LiteralPath $Path)) {
        return $true
      }
    }
    catch {
      if ($attempt -eq $Attempts) {
        Write-Warning "Failed to remove '$Path' after $Attempts attempts: $($_.Exception.Message)"
      }
    }

    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
    Start-Sleep -Milliseconds $DelayMilliseconds
  }

  return -not (Test-Path -LiteralPath $Path)
}

function Invoke-CapturedProcess {
  param([string]$FileName, [string[]]$Arguments, [string]$WorkingDirectory)

  $startInfo = [Diagnostics.ProcessStartInfo]::new()
  $startInfo.FileName = $FileName
  $startInfo.WorkingDirectory = $WorkingDirectory
  $startInfo.UseShellExecute = $false
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $startInfo.CreateNoWindow = $true
  foreach ($argument in $Arguments) {
    $startInfo.ArgumentList.Add($argument)
  }

  $process = [Diagnostics.Process]::new()
  $process.StartInfo = $startInfo
  if (-not $process.Start()) {
    throw "Failed to start process: $FileName"
  }

  $stdoutTask = $process.StandardOutput.ReadToEndAsync()
  $stderrTask = $process.StandardError.ReadToEndAsync()
  $process.WaitForExit()
  $result = [pscustomobject]@{
    ExitCode = $process.ExitCode
    Stdout = $stdoutTask.GetAwaiter().GetResult()
    Stderr = $stderrTask.GetAwaiter().GetResult()
  }
  $process.Dispose()
  return $result
}

function Get-RelativePathValue {
  param([string]$Path)

  return [IO.Path]::GetRelativePath($RepositoryRoot, [IO.Path]::GetFullPath($Path)).Replace('\', '/')
}

function Get-LastDiagnostic {
  param([string]$Stdout, [string]$Stderr)

  $combined = $Stdout + "`n" + $Stderr
  if ($combined -match 'YoloVision=Skipped Reason=(?<reason>[^\r\n]+)') {
    return $Matches.reason.Trim()
  }

  $lines = @($combined -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($lines.Count -eq 0) {
    return "No process diagnostic was captured."
  }

  $diagnostic = $lines[-1].Trim()
  if ($diagnostic.Length -le 1000) {
    return $diagnostic
  }

  return $diagnostic.Substring(0, 1000)
}

function Get-StageReached {
  param([string]$LineReportDirectory, [bool]$RuntimePassed)

  if ($RuntimePassed) { return "runtime-passed" }
  if (Test-Path -LiteralPath (Join-Path $LineReportDirectory "runtime.stdout.log")) { return "runtime-attempted" }
  if (Test-Path -LiteralPath (Join-Path $LineReportDirectory "build.log")) { return "consumer-built" }
  if (Test-Path -LiteralPath (Join-Path $LineReportDirectory "restore.log")) { return "consumer-restored" }
  return "preflight"
}

function Get-PackageEvidence {
  param([string]$Directory, [string]$PackageId, [string]$Version, [string]$Role)

  $packagePath = Join-Path $Directory "$PackageId.$Version.nupkg"
  if (-not (Test-Path -LiteralPath $packagePath -PathType Leaf)) {
    throw "Expected $Role package does not exist: $packagePath"
  }

  $item = Get-Item -LiteralPath $packagePath
  return [pscustomobject][ordered]@{
    role = $Role
    id = $PackageId
    version = $Version
    path = Get-RelativePathValue -Path $item.FullName
    length = $item.Length
    sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}

function Get-MissingRuntimePatterns {
  param([string]$Root, [object[]]$Patterns)

  if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    return @($Patterns | ForEach-Object { [string]$_ })
  }

  return @(
    foreach ($pattern in $Patterns) {
      $candidate = Join-Path $Root ([string]$pattern)
      if (-not (Test-Path -Path $candidate -PathType Leaf)) {
        [string]$pattern
      }
    }
  )
}

function Resolve-TensorRtRuntimeRoot {
  param([object]$RuntimePackage, [string]$DefaultRoot, [string[]]$AdditionalCandidates)

  $patterns = @($RuntimePackage.tensorRtFiles)
  $candidates = @($DefaultRoot) + @($AdditionalCandidates)
  foreach ($candidate in @($candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
    $fullCandidate = [IO.Path]::GetFullPath($candidate)
    $missingPatterns = @(Get-MissingRuntimePatterns -Root $fullCandidate -Patterns $patterns)
    if ($missingPatterns.Count -eq 0) {
      return [pscustomobject]@{
        Root = $fullCandidate
        Source = if ([string]::Equals($fullCandidate, [IO.Path]::GetFullPath($DefaultRoot), [StringComparison]::OrdinalIgnoreCase)) { "resolved-runtime-root" } else { "existing-assembled-runtime" }
        MissingPatterns = @()
      }
    }
  }

  return [pscustomobject]@{
    Root = [IO.Path]::GetFullPath($DefaultRoot)
    Source = "resolved-runtime-root-incomplete"
    MissingPatterns = @(Get-MissingRuntimePatterns -Root $DefaultRoot -Patterns $patterns)
  }
}

$OutputRoot = Resolve-PathValue -Value $OutputRoot -DefaultValue (Join-Path $outerRoot "consumer-workspaces\yolovision-yolox-local-package-matrix") -RelativeRoot $outerRoot
$ReportDirectory = Resolve-PathValue -Value $ReportDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\yolovision\yolox-local-package-consumer-matrix") -RelativeRoot $RepositoryRoot
$ManagedPackageDirectory = Resolve-PathValue -Value $ManagedPackageDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\managed") -RelativeRoot $RepositoryRoot
$YoloVisionPackageDirectory = Resolve-PathValue -Value $YoloVisionPackageDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\yolovision-nupkg") -RelativeRoot $RepositoryRoot

Assert-PathUnderRoot -Path $OutputRoot -Root $outerRoot -Description "Matrix consumer workspace"
Assert-PathUnderRoot -Path $ReportDirectory -Root $outerRoot -Description "Matrix report directory"
foreach ($pathItem in @(
  @{ Path = $OutputRoot; Description = "Matrix consumer workspace" },
  @{ Path = $ReportDirectory; Description = "Matrix report directory" },
  @{ Path = $ManagedPackageDirectory; Description = "Managed package feed" },
  @{ Path = $YoloVisionPackageDirectory; Description = "YoloVision package feed" }
)) {
  Assert-NonCDrivePath -Path $pathItem.Path -Description $pathItem.Description
}

if (Test-Path -LiteralPath $OutputRoot) {
  $resolvedOutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
  Assert-PathUnderRoot -Path $resolvedOutputRoot -Root $outerRoot -Description "Existing matrix workspace"
  if (-not (Remove-DirectoryWithRetry -Path $resolvedOutputRoot)) {
    throw "Existing matrix workspace could not be cleaned: $resolvedOutputRoot"
  }
}
if (Test-Path -LiteralPath $ReportDirectory) {
  $resolvedReportDirectory = (Resolve-Path -LiteralPath $ReportDirectory).Path
  Assert-PathUnderRoot -Path $resolvedReportDirectory -Root $outerRoot -Description "Existing matrix report directory"
  if (-not (Remove-DirectoryWithRetry -Path $resolvedReportDirectory)) {
    throw "Existing matrix report directory could not be cleaned: $resolvedReportDirectory"
  }
}
New-Item -ItemType Directory -Path $OutputRoot, $ReportDirectory -Force | Out-Null

$manifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$downloadsRoot = Join-Path $outerRoot "downloads"
$assembledRuntimeCandidates = if (Test-Path -LiteralPath $downloadsRoot -PathType Container) {
  @(Get-ChildItem -LiteralPath $downloadsRoot -Directory -Recurse -Filter "assembled-runtime" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
} else {
  @()
}
$consumerScript = Join-Path $RepositoryRoot "eng\Test-YoloVisionLocalPackageConsumer.ps1"
$rows = [Collections.Generic.List[object]]::new()
$bridgePackages = [Collections.Generic.List[object]]::new()
$lineIndex = 0

foreach ($key in @($RuntimePackageKey | Select-Object -Unique)) {
  $lineIndex++
  $bridgeEntries = @($manifest.packages | Where-Object {
    [string]::Equals([string]$_.sourceRuntimeKey, $key, [StringComparison]::OrdinalIgnoreCase) -and
    [string]::Equals([string]$_.role, "bridge", [StringComparison]::OrdinalIgnoreCase)
  })
  if ($bridgeEntries.Count -ne 1) {
    throw "Runtime package key '$key' must resolve to one bridge manifest entry. Found $($bridgeEntries.Count)."
  }

  $bridge = $bridgeEntries[0]
  $runtimePackages = @($runtimeManifest.packages | Where-Object {
    [string]::Equals([string]$_.key, $key, [StringComparison]::OrdinalIgnoreCase)
  })
  if ($runtimePackages.Count -ne 1) {
    throw "Runtime package key '$key' must resolve to one full runtime manifest entry. Found $($runtimePackages.Count)."
  }
  $runtimePackage = $runtimePackages[0]
  $lineOutputRoot = Join-Path $OutputRoot ("line-{0:D2}-trt{1}" -f $lineIndex, [string]$bridge.tensorRtLine)
  $lineReportDirectory = Join-Path $ReportDirectory $key
  New-Item -ItemType Directory -Path $lineReportDirectory -Force | Out-Null
  $bridgePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$key"
  $bridgeEvidence = Get-PackageEvidence -Directory $bridgePackageDirectory -PackageId ([string]$bridge.packageId) -Version $PackageVersion -Role "bridge-only"
  $bridgePackages.Add($bridgeEvidence)

  $rootsJson = (& pwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") -RuntimePackageKey $key -RepositoryRoot $RepositoryRoot | Out-String).Trim()
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to resolve vendor roots for runtime package key '$key'."
  }
  $roots = $rootsJson | ConvertFrom-Json
  $runtimeRootResolution = Resolve-TensorRtRuntimeRoot -RuntimePackage $runtimePackage -DefaultRoot ([string]$roots.tensorRtRoot) -AdditionalCandidates $assembledRuntimeCandidates
  $tensorRtRuntimeRoot = [string]$runtimeRootResolution.Root
  $tensorRtRootExists = Test-Path -LiteralPath ([string]$roots.tensorRtRoot) -PathType Container
  $tensorRtRuntimeRootExists = Test-Path -LiteralPath $tensorRtRuntimeRoot -PathType Container
  $cudaRootExists = Test-Path -LiteralPath ([string]$roots.cudaRoot) -PathType Container
  $cudnnRootExists = Test-Path -LiteralPath ([string]$roots.cudnnRoot) -PathType Container
  $tensorRtDllCount = if ($tensorRtRuntimeRootExists) { @(Get-ChildItem -LiteralPath $tensorRtRuntimeRoot -Recurse -File -Filter 'nvinfer*.dll' -ErrorAction SilentlyContinue).Count } else { 0 }
  $cudaRuntimeDllCount = if ($cudaRootExists) { @(Get-ChildItem -LiteralPath ([string]$roots.cudaRoot) -Recurse -File -Filter 'cudart64*.dll' -ErrorAction SilentlyContinue).Count } else { 0 }
  $cudnnDllCount = if ($cudnnRootExists) { @(Get-ChildItem -LiteralPath ([string]$roots.cudnnRoot) -Recurse -File -Filter 'cudnn*.dll' -ErrorAction SilentlyContinue).Count } else { 0 }

  $arguments = [Collections.Generic.List[string]]::new()
  foreach ($value in @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $consumerScript,
    "-RepositoryRoot", $RepositoryRoot,
    "-OutputRoot", $lineOutputRoot,
    "-ReportDirectory", $lineReportDirectory,
    "-ManagedPackageDirectory", $ManagedPackageDirectory,
    "-YoloVisionPackageDirectory", $YoloVisionPackageDirectory,
    "-BridgePackageDirectory", $bridgePackageDirectory,
    "-RuntimePackageKey", $key,
    "-TensorRtRoot", [string]$roots.tensorRtRoot,
    "-TensorRtRuntimeRoot", $tensorRtRuntimeRoot,
    "-CudaRoot", [string]$roots.cudaRoot,
    "-CudnnRoot", [string]$roots.cudnnRoot,
    "-PackageVersion", $PackageVersion
  )) {
    $arguments.Add([string]$value)
  }
  foreach ($optional in @(
    @{ Name = "-ModelPath"; Value = $ModelPath },
    @{ Name = "-LabelsPath"; Value = $LabelsPath },
    @{ Name = "-ImagePath"; Value = $ImagePath }
  )) {
    if (-not [string]::IsNullOrWhiteSpace([string]$optional.Value)) {
      $arguments.Add([string]$optional.Name)
      $arguments.Add([string]$optional.Value)
    }
  }
  if ($KeepWorkspace.IsPresent) {
    $arguments.Add("-KeepWorkspace")
  }

  Write-Host "YoloVisionMatrix RuntimePackageKey=$key TensorRtLine=$($bridge.tensorRtLine) Starting=True"
  $processResult = Invoke-CapturedProcess -FileName "pwsh" -Arguments $arguments.ToArray() -WorkingDirectory $RepositoryRoot
  $invocationStdoutPath = Join-Path $lineReportDirectory "invocation.stdout.log"
  $invocationStderrPath = Join-Path $lineReportDirectory "invocation.stderr.log"
  [IO.File]::WriteAllText($invocationStdoutPath, $processResult.Stdout, $utf8)
  [IO.File]::WriteAllText($invocationStderrPath, $processResult.Stderr, $utf8)
  Write-Host $processResult.Stdout
  if (-not [string]::IsNullOrWhiteSpace($processResult.Stderr)) {
    Write-Warning $processResult.Stderr
  }

  $capturedBridgeTensorRtVersion = ""
  $capturedBridgeCudaToolkitVersion = ""
  if ($processResult.Stdout -match 'YoloVisionPackageConsumer BridgeTensorRt=(?<trt>\S+) BridgeCuda=(?<cuda>\S+)') {
    $capturedBridgeTensorRtVersion = $Matches.trt
    $capturedBridgeCudaToolkitVersion = $Matches.cuda
  }

  $lineReportPath = Join-Path $lineReportDirectory "yolox-local-package-consumer-runtime.json"
  $lineReport = if (Test-Path -LiteralPath $lineReportPath -PathType Leaf) {
    Get-Content -LiteralPath $lineReportPath -Raw -Encoding utf8 | ConvertFrom-Json
  } else {
    $null
  }
  $runtimePassed = $processResult.ExitCode -eq 0 -and $null -ne $lineReport -and
    [string]::Equals([string]$lineReport.validationState, "passed-local-package-consumer-runtime", [StringComparison]::Ordinal)
  $stageReached = Get-StageReached -LineReportDirectory $lineReportDirectory -RuntimePassed $runtimePassed
  $diagnostic = if ($runtimePassed) { "Runtime passed with required package, bridge-version, inference, and output markers." } else { Get-LastDiagnostic -Stdout $processResult.Stdout -Stderr $processResult.Stderr }

  $lineWorkspaceRemoved = -not (Test-Path -LiteralPath $lineOutputRoot)
  if (-not $KeepWorkspace.IsPresent -and -not $lineWorkspaceRemoved) {
    $resolvedLineOutputRoot = (Resolve-Path -LiteralPath $lineOutputRoot).Path
    Assert-PathUnderRoot -Path $resolvedLineOutputRoot -Root $OutputRoot -Description "Matrix line cleanup target"
    $lineWorkspaceRemoved = Remove-DirectoryWithRetry -Path $resolvedLineOutputRoot
  }

  $rows.Add([pscustomobject][ordered]@{
    runtimePackageKey = $key
    tensorRtLine = [string]$bridge.tensorRtLine
    bridgePackageId = [string]$bridge.packageId
    bridgePackageSha256 = $bridgeEvidence.sha256
    invocationExitCode = $processResult.ExitCode
    state = if ($runtimePassed) { "passed-local-package-consumer-runtime" } else { "blocked-runtime-attempt" }
    evidenceClassification = if ($runtimePassed) { "local-package-consumer-runtime" } else { "runtime-attempt-blocked" }
    stageReached = $stageReached
    runtimePassed = $runtimePassed
    predictionCount = if ($runtimePassed) { [int]$lineReport.runtime.predictionCount } else { 0 }
    elapsedMilliseconds = if ($runtimePassed) { [double]$lineReport.runtime.elapsedMilliseconds } else { 0.0 }
    bridgeBuildTensorRtVersion = if ($runtimePassed) { [string]$lineReport.nativeDependency.bridgeBuildTensorRtVersion } else { $capturedBridgeTensorRtVersion }
    bridgeBuildCudaToolkitVersion = if ($runtimePassed) { [string]$lineReport.nativeDependency.bridgeBuildCudaToolkitVersion } else { $capturedBridgeCudaToolkitVersion }
    diagnostic = $diagnostic
    dependencyProbe = [pscustomobject][ordered]@{
      tensorRtRoot = [string]$roots.tensorRtRoot
      tensorRtRootExists = $tensorRtRootExists
      tensorRtRuntimeRoot = $tensorRtRuntimeRoot
      tensorRtRuntimeRootSource = [string]$runtimeRootResolution.Source
      tensorRtRuntimeRootExists = $tensorRtRuntimeRootExists
      tensorRtRuntimeRequiredPatternCount = @($runtimePackage.tensorRtFiles).Count
      tensorRtRuntimeMissingPatterns = @($runtimeRootResolution.MissingPatterns)
      tensorRtDllCount = $tensorRtDllCount
      cudaRoot = [string]$roots.cudaRoot
      cudaRootExists = $cudaRootExists
      cudaRuntimeDllCount = $cudaRuntimeDllCount
      cudnnRoot = [string]$roots.cudnnRoot
      cudnnRootExists = $cudnnRootExists
      cudnnDllCount = $cudnnDllCount
    }
    reportPath = if ($runtimePassed) { Get-RelativePathValue -Path $lineReportPath } else { "" }
    invocationStdoutPath = Get-RelativePathValue -Path $invocationStdoutPath
    invocationStdoutSha256 = (Get-FileHash -LiteralPath $invocationStdoutPath -Algorithm SHA256).Hash.ToLowerInvariant()
    invocationStderrPath = Get-RelativePathValue -Path $invocationStderrPath
    invocationStderrSha256 = (Get-FileHash -LiteralPath $invocationStderrPath -Algorithm SHA256).Hash.ToLowerInvariant()
    workspaceRemovedAfterValidation = $lineWorkspaceRemoved
  })
}

$managedPackage = Get-PackageEvidence -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API" -Version $PackageVersion -Role "managed-api"
$yoloVisionPackage = Get-PackageEvidence -Directory $YoloVisionPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API.YoloVision" -Version $PackageVersion -Role "yolovision"
$passedCount = @($rows | Where-Object { $_.runtimePassed }).Count
$blockedCount = $rows.Count - $passedCount
$workspaceRemoved = if ($KeepWorkspace.IsPresent) { $false } else {
  if (Test-Path -LiteralPath $OutputRoot) {
    [void](Remove-DirectoryWithRetry -Path $OutputRoot)
  }
  -not (Test-Path -LiteralPath $OutputRoot)
}

$matrix = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-yolox-local-package-consumer-runtime-matrix"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  validationState = if ($blockedCount -eq 0) { "passed-all-requested-runtime-lines" } else { "completed-with-runtime-blockers" }
  evidenceClassification = "local-package-consumer-runtime-matrix"
  packageVersion = $PackageVersion
  requestedRuntimeCount = $rows.Count
  passedRuntimeCount = $passedCount
  blockedRuntimeCount = $blockedCount
  workspaceDrive = [IO.Path]::GetPathRoot($OutputRoot).TrimEnd('\')
  workspaceRemovedAfterValidation = $workspaceRemoved
  sharedPackages = @($managedPackage, $yoloVisionPackage)
  bridgePackages = @($bridgePackages)
  rows = @($rows)
  boundary = [pscustomobject][ordered]@{
    isLocalPackageConsumerRuntimeMatrix = $true
    successfulRowsAreLocalPackageConsumerRuntimeEvidence = $true
    blockedRowsAreRuntimeExecutionProof = $false
    isPackageConsumerRuntimeProof = $false
    packagesDownloadedFromPublicFeed = $false
    publicRedistributionOwnerApproval = $false
    canPromotePackageConsumerRuntime = $false
    canPublishPublicly = $false
    isPostPublishProof = $false
    canCloseReleaseIssue = $false
    performsPublish = $false
  }
}

$matrixPath = Join-Path $ReportDirectory "yolox-local-package-consumer-runtime-matrix.json"
$matrix | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $matrixPath -Encoding utf8
$markdownPath = [IO.Path]::ChangeExtension($matrixPath, ".md")
$lines = [Collections.Generic.List[string]]::new()
$lines.Add("# YOLOX Local Package Consumer Runtime Matrix")
$lines.Add("")
$lines.Add("- state: ``$($matrix.validationState)``")
$lines.Add("- requested: ``$($rows.Count)``")
$lines.Add("- runtime passed: ``$passedCount``")
$lines.Add("- runtime blocked: ``$blockedCount``")
$lines.Add("- workspace removed: ``$workspaceRemoved``")
$lines.Add("- public package-consumer proof: ``False``")
$lines.Add("")
$lines.Add("| Runtime key | TRT line | Stage | State | Predictions | Diagnostic |")
$lines.Add("| --- | ---: | --- | --- | ---: | --- |")
foreach ($row in $rows) {
  $safeDiagnostic = ([string]$row.diagnostic).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
  $lines.Add("| ``$($row.runtimePackageKey)`` | ``$($row.tensorRtLine)`` | ``$($row.stageReached)`` | ``$($row.state)`` | $($row.predictionCount) | $safeDiagnostic |")
}
$lines.Add("")
$lines.Add("Successful rows prove only a local-file-feed PackageReference restore/build/real-model runtime on this host. Blocked rows remain non-proof. No row proves public download, redistribution approval, post-publish verification, or release closure.")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "ValidationState=$($matrix.validationState) PassedRuntimeCount=$passedCount BlockedRuntimeCount=$blockedCount"
Write-Host "EvidenceClassification=$($matrix.evidenceClassification) IsPackageConsumerRuntimeProof=False CanPublishPublicly=False"
Write-Host "Report=$matrixPath"

if ($RequireAllRuntimePass.IsPresent -and $blockedCount -ne 0) {
  throw "$blockedCount requested TensorRT runtime line(s) did not pass. See $matrixPath"
}
