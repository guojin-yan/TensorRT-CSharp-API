[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$MatrixPath,
  [string]$SurfaceAuditPath,
  [string]$HandoffPath,
  [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if ([string]::IsNullOrWhiteSpace($MatrixPath)) { $MatrixPath = Join-Path $RepositoryRoot "artifacts\yolovision\yolox-local-package-consumer-matrix\yolox-local-package-consumer-runtime-matrix.json" }
if ([string]::IsNullOrWhiteSpace($SurfaceAuditPath)) { $SurfaceAuditPath = Join-Path $RepositoryRoot "artifacts\yolovision\package-surface-audit\yolovision-package-surface-audit.json" }
if ([string]::IsNullOrWhiteSpace($HandoffPath)) { $HandoffPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\yolovision-public-package-owner-handoff.json" }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $RepositoryRoot "artifacts\interface-coverage" }
foreach ($path in @($MatrixPath, $SurfaceAuditPath, $HandoffPath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Matrix proof input does not exist: $path" }
}
$matrix = Get-Content -LiteralPath $MatrixPath -Raw -Encoding utf8 | ConvertFrom-Json
$surface = Get-Content -LiteralPath $SurfaceAuditPath -Raw -Encoding utf8 | ConvertFrom-Json
$handoff = Get-Content -LiteralPath $HandoffPath -Raw -Encoding utf8 | ConvertFrom-Json

$expectedKeys = @(
  "win-x64-trt8.6-cuda12.1-cudnn8.9",
  "win-x64-trt10.11-cuda12.9-cudnn9.22",
  "win-x64-trt11.0-cuda12.9-cudnn9.22"
)
$rows = @($matrix.rows)
$passedRows = @($rows | Where-Object { [bool]$_.runtimePassed })
$blockedRows = @($rows | Where-Object { -not [bool]$_.runtimePassed })
$errors = [Collections.Generic.List[string]]::new()
if ($rows.Count -ne 3) { $errors.Add("Matrix must contain exactly three requested rows.") }
foreach ($key in $expectedKeys) {
  if (@($rows | Where-Object { $_.runtimePackageKey -eq $key }).Count -ne 1) { $errors.Add("Matrix is missing unique runtime key: $key") }
}
if ($passedRows.Count -ne 2 -or $blockedRows.Count -ne 1) { $errors.Add("Expected two runtime passes and one blocker for the current host/assets.") }
if (-not [bool]$matrix.workspaceRemovedAfterValidation -or @($rows | Where-Object { -not [bool]$_.workspaceRemovedAfterValidation }).Count -ne 0) { $errors.Add("Every matrix workspace must be removed.") }
if (-not [bool]$surface.valid -or [int]$surface.surface.forbiddenPointerOrHandleFindingCount -ne 0 -or [int]$surface.surface.sampleInternalTypeLeakFindingCount -ne 0) { $errors.Add("YoloVision package public surface audit is not clean.") }
if ([bool]$handoff.performsPublish -or [bool]$handoff.canPublishPublicly -or [bool]$handoff.canCloseReleaseIssue) { $errors.Add("Owner handoff must remain read-only and non-publishing.") }
if ([bool]$matrix.boundary.isPackageConsumerRuntimeProof -or [bool]$matrix.boundary.packagesDownloadedFromPublicFeed -or [bool]$matrix.boundary.performsPublish) { $errors.Add("Local matrix proof boundaries were promoted incorrectly.") }

$compactRows = foreach ($row in $rows) {
  $bridgeTensorRtVersion = [string]$row.bridgeBuildTensorRtVersion
  $bridgeCudaVersion = [string]$row.bridgeBuildCudaToolkitVersion
  if ([string]::IsNullOrWhiteSpace($bridgeTensorRtVersion) -and -not [string]::IsNullOrWhiteSpace([string]$row.invocationStdoutPath)) {
    $stdoutPath = Join-Path $RepositoryRoot ([string]$row.invocationStdoutPath).Replace('/', '\')
    if (Test-Path -LiteralPath $stdoutPath -PathType Leaf) {
      $stdout = Get-Content -LiteralPath $stdoutPath -Raw -Encoding utf8
      if ($stdout -match 'YoloVisionPackageConsumer BridgeTensorRt=(?<trt>\S+) BridgeCuda=(?<cuda>\S+)') {
        $bridgeTensorRtVersion = $Matches.trt
        $bridgeCudaVersion = $Matches.cuda
      }
    }
  }
  if ($bridgeTensorRtVersion -notmatch '^(?<major>[0-9]+)' -or $Matches.major -ne [string]$row.tensorRtLine) {
    $errors.Add("Bridge build version does not match requested line for $($row.runtimePackageKey).")
  }
  if ([bool]$row.runtimePassed -and ([int]$row.predictionCount -le 0 -or [string]$row.evidenceClassification -ne "local-package-consumer-runtime")) {
    $errors.Add("Passing row lacks runtime predictions/classification: $($row.runtimePackageKey)")
  }
  if ((-not [bool]$row.runtimePassed) -and ([string]$row.evidenceClassification -ne "runtime-attempt-blocked" -or [string]::IsNullOrWhiteSpace([string]$row.diagnostic))) {
    $errors.Add("Blocked row lacks fail-closed classification/diagnostic: $($row.runtimePackageKey)")
  }
  [pscustomobject][ordered]@{
    runtimePackageKey = [string]$row.runtimePackageKey
    tensorRtLine = [string]$row.tensorRtLine
    bridgePackageId = [string]$row.bridgePackageId
    bridgePackageSha256 = [string]$row.bridgePackageSha256
    state = [string]$row.state
    evidenceClassification = [string]$row.evidenceClassification
    stageReached = [string]$row.stageReached
    runtimePassed = [bool]$row.runtimePassed
    predictionCount = [int]$row.predictionCount
    elapsedMilliseconds = [double]$row.elapsedMilliseconds
    bridgeBuildTensorRtVersion = $bridgeTensorRtVersion
    bridgeBuildCudaToolkitVersion = $bridgeCudaVersion
    bridgeBuildTensorRtLineMatches = $true
    tensorRtRuntimeRootSource = [string]$row.dependencyProbe.tensorRtRuntimeRootSource
    tensorRtRuntimeRequiredPatternCount = [int]$row.dependencyProbe.tensorRtRuntimeRequiredPatternCount
    tensorRtRuntimeMissingPatternCount = @($row.dependencyProbe.tensorRtRuntimeMissingPatterns).Count
    cudnnRuntimeDllCount = [int]$row.dependencyProbe.cudnnDllCount
    diagnostic = [string]$row.diagnostic
    invocationStdoutSha256 = [string]$row.invocationStdoutSha256
    invocationStderrSha256 = [string]$row.invocationStderrSha256
    workspaceRemovedAfterValidation = [bool]$row.workspaceRemovedAfterValidation
  }
}

$successfulReports = foreach ($row in $passedRows) {
  $path = Join-Path $RepositoryRoot ([string]$row.reportPath).Replace('/', '\')
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    $errors.Add("Successful row report is missing: $($row.runtimePackageKey)")
    continue
  }
  Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}
$assetReference = @($successfulReports | Select-Object -First 1)
if ($assetReference.Count -ne 1) { $errors.Add("No successful line report was available for asset hash closure.") }
$assets = if ($assetReference.Count -eq 1) {
  [pscustomobject][ordered]@{
    modelSha256 = [string]$assetReference[0].assets.modelSha256
    labelsSha256 = [string]$assetReference[0].assets.labelsSha256
    imageSha256 = [string]$assetReference[0].assets.imageSha256
    tensorSha256 = [string]$assetReference[0].assets.tensorSha256
    remainOnEDrive = [bool]$assetReference[0].assets.assetsRemainOnEDrive
    committed = $false
  }
} else { $null }

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$testDirectoryPatterns = @("jyppx-yolovision-assets-*", "jyppx-yolovision-input-tests", "jyppx-yolovision-preprocess-tests", "jyppx-yolox-preprocess-tests", "jyppx-yolovision-report-tests", "jyppx-yolovision-preflight", "jyppx-yolovision-output-report")
$testDirectoryMatches = foreach ($pattern in $testDirectoryPatterns) {
  Get-ChildItem -LiteralPath $tempRoot -Force -Directory -Filter $pattern -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
}
$scanRoots = @("$env:USERPROFILE\Downloads", "$env:USERPROFILE\Documents", "$env:USERPROFILE\Desktop") | Where-Object { Test-Path -LiteralPath $_ }
$options = [IO.EnumerationOptions]::new()
$options.RecurseSubdirectories = $true
$options.IgnoreInaccessible = $true
$options.AttributesToSkip = [IO.FileAttributes]::ReparsePoint
$assetMatches = foreach ($root in $scanRoots) {
  [IO.Directory]::EnumerateFiles($root, '*', $options) | Where-Object { $_ -match '(?i)(yolox|yolovision|YoloVision\.PackageConsumer)' }
}
if (@($testDirectoryMatches).Count -ne 0 -or @($assetMatches).Count -ne 0) { $errors.Add("C-drive YoloVision test directories or named assets remain.") }

$packageEvidence = foreach ($package in @($matrix.sharedPackages) + @($matrix.bridgePackages)) {
  [pscustomobject][ordered]@{
    role = [string]$package.role
    id = [string]$package.id
    version = [string]$package.version
    length = [int64]$package.length
    sha256 = [string]$package.sha256
  }
}
$proof = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-yolox-multi-version-local-package-consumer-runtime-proof-closure"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  validationState = if ($errors.Count -eq 0) { "passed-path-free-matrix-proof-closure" } else { "failed-matrix-proof-closure" }
  evidenceClassification = "local-package-consumer-runtime-matrix"
  requestedRuntimeCount = $rows.Count
  passedRuntimeCount = $passedRows.Count
  blockedRuntimeCount = $blockedRows.Count
  packages = @($packageEvidence)
  rows = @($compactRows)
  assets = $assets
  packageSurfaceAudit = [pscustomobject][ordered]@{
    valid = [bool]$surface.valid
    packageSha256 = [string]$surface.package.sha256
    exportedTypeCount = [int]$surface.surface.exportedTypeCount
    publicDeclaredMemberCount = [int]$surface.surface.publicDeclaredMemberCount
    xmlDocumentedMemberCount = [int]$surface.surface.xmlDocumentedMemberCount
    forbiddenPointerOrHandleFindingCount = [int]$surface.surface.forbiddenPointerOrHandleFindingCount
    sampleInternalTypeLeakFindingCount = [int]$surface.surface.sampleInternalTypeLeakFindingCount
  }
  ownerHandoff = [pscustomobject][ordered]@{
    state = [string]$handoff.state
    packageCount = @($handoff.packages).Count
    runtimeKeyCount = @($handoff.runtimePackageKeys).Count
    strictValidatorCommandCount = @($handoff.strictValidatorCommands).Count
    performsPublish = [bool]$handoff.performsPublish
  }
  localEvidence = [pscustomobject][ordered]@{
    root = "artifacts/yolovision/yolox-local-package-consumer-matrix"
    rawEvidenceCommitted = $false
    reason = "Raw reports contain local paths; this compact closure retains path-free hashes, states, and boundaries."
  }
  cDriveAudit = [pscustomobject][ordered]@{
    testDirectoryMatchCount = @($testDirectoryMatches).Count
    yoloXOrConsumerAssetMatchCount = @($assetMatches).Count
    packageCacheRootExists = Test-Path -LiteralPath "C:\jyppx-pkgcache"
    splitPackageTempRootExists = Test-Path -LiteralPath "C:\jyppx-split-packages"
    consumerWorkspaceUsedCDrive = $false
    unrelatedUserOrSystemFilesRemoved = $false
  }
  errors = @($errors)
  boundary = [pscustomobject][ordered]@{
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
if ([bool]$proof.cDriveAudit.packageCacheRootExists -or [bool]$proof.cDriveAudit.splitPackageTempRootExists) {
  $errors.Add("Known C-drive package cache roots still exist.")
  $proof.validationState = "failed-matrix-proof-closure"
  $proof.errors = @($errors)
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$jsonPath = Join-Path $OutputDirectory "yolox-multi-version-local-package-consumer-runtime-proof-closure.json"
$proof | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$mdPath = [IO.Path]::ChangeExtension($jsonPath, ".md")
$lines = [Collections.Generic.List[string]]::new()
$lines.Add("# YOLOX Multi-version Local Package Consumer Runtime Proof Closure")
$lines.Add("")
$lines.Add("- state: ``$($proof.validationState)``")
$lines.Add("- requested / passed / blocked: ``$($rows.Count) / $($passedRows.Count) / $($blockedRows.Count)``")
$lines.Add("- package surface findings: ``0``")
$lines.Add("- C-drive test/assets matches: ``$(@($testDirectoryMatches).Count) / $(@($assetMatches).Count)``")
$lines.Add("- package-consumer runtime proof: ``False``")
$lines.Add("- publish executed: ``False``")
$lines.Add("")
$lines.Add("| Runtime key | TRT | State | Predictions | Elapsed ms | Runtime root | Diagnostic |")
$lines.Add("| --- | ---: | --- | ---: | ---: | --- | --- |")
foreach ($row in $compactRows) {
  $safeDiagnostic = $row.diagnostic.Replace("|", "\|")
  $lines.Add("| ``$($row.runtimePackageKey)`` | $($row.tensorRtLine) | ``$($row.state)`` | $($row.predictionCount) | $($row.elapsedMilliseconds) | ``$($row.tensorRtRuntimeRootSource)`` | $safeDiagnostic |")
}
$lines.Add("")
$lines.Add("TRT10 and TRT11 are real local-file-feed PackageReference YOLOX runtimes on this host. TRT8 is blocked because the available cuDNN 8 developer root has no cudnn64_8 runtime DLL, so its bridge intentionally excludes the ONNX parser. No row proves public download, owner redistribution approval, post-publish verification, or release closure.")
$lines | Set-Content -LiteralPath $mdPath -Encoding utf8
Write-Host "ValidationState=$($proof.validationState) PassedRuntimeCount=$($passedRows.Count) BlockedRuntimeCount=$($blockedRows.Count) ErrorCount=$($errors.Count)"
Write-Host "PackageConsumerRuntimeProof=False PerformsPublish=False"
Write-Host "Proof=$jsonPath"
if ($errors.Count -ne 0) { throw "YOLOX multi-version compact proof export failed. See $jsonPath" }
