[CmdletBinding()]
param(
  [string[]]$SourceRuntimeKey = @(
    "win-x64-trt10.11-cuda12.9-cudnn9.22",
    "win-x64-trt11.0-cuda12.9-cudnn9.22"),
  [string]$ManagedPackageDirectory,
  [string]$BridgePackageRoot,
  [string[]]$AdditionalPackageSource = @(),
  [string]$OutputRoot,
  [string]$ReportRoot,
  [switch]$SkipInstalledVendorAssetHashing,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

function Resolve-RepositoryPath {
  param(
    [string]$Value,
    [Parameter(Mandatory = $true)]
    [string]$DefaultRelativePath
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $DefaultRelativePath))
  }

  if ([System.IO.Path]::IsPathRooted($Value)) {
    return [System.IO.Path]::GetFullPath($Value)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Value))
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
  ) | Sort-Object -Unique
}

function Get-BooleanProperty {
  param(
    [AllowNull()]
    [object]$Object,
    [Parameter(Mandatory = $true)]
    [string]$Name
  )

  if ($null -eq $Object -or $Object.PSObject.Properties.Name -notcontains $Name) {
    return $false
  }

  return [bool]$Object.$Name
}

function ConvertTo-MarkdownCell {
  param([string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return ""
  }

  return (($Value -replace '\|', '\|') -replace "(`r`n|`n|`r)", "<br>")
}

function Get-MissingRuntimePatterns {
  param(
    [string]$Root,
    [object[]]$Patterns
  )

  if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path -LiteralPath $Root -PathType Container)) {
    return @($Patterns | ForEach-Object { [string]$_ })
  }

  return @(
    foreach ($pattern in $Patterns) {
      if (-not (Test-Path -Path (Join-Path $Root ([string]$pattern)) -PathType Leaf)) {
        [string]$pattern
      }
    }
  )
}

function Resolve-TensorRtRuntimeRoot {
  param(
    [object]$RuntimePackage,
    [string]$DefaultRoot,
    [string[]]$AdditionalCandidates
  )

  $patterns = @($RuntimePackage.tensorRtFiles)
  $candidates = @($DefaultRoot) + @($AdditionalCandidates)
  foreach ($candidate in @($candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
    $fullCandidate = [System.IO.Path]::GetFullPath($candidate)
    if (@(Get-MissingRuntimePatterns -Root $fullCandidate -Patterns $patterns).Count -eq 0) {
      return $fullCandidate
    }
  }

  $missing = @(Get-MissingRuntimePatterns -Root $DefaultRoot -Patterns $patterns)
  throw "TensorRT runtime root for '$($RuntimePackage.key)' is incomplete. Missing: $($missing -join ', ')"
}

$runtimeKeys = @(Expand-KeyList -Values $SourceRuntimeKey)
if ($runtimeKeys.Count -eq 0) {
  throw "At least one source runtime key is required."
}

$scenarios = @(
  "callback-return-false",
  "callback-throw",
  "attempted-no-invocation",
  "missing-vendor-dependency")
$ManagedPackageDirectory = Resolve-RepositoryPath -Value $ManagedPackageDirectory -DefaultRelativePath "artifacts\managed"
$BridgePackageRoot = Resolve-RepositoryPath -Value $BridgePackageRoot -DefaultRelativePath "artifacts\runtime-split-nupkg"
$ReportRoot = Resolve-RepositoryPath -Value $ReportRoot -DefaultRelativePath "artifacts\package-consumer\bridge-runtime-negative-controls"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path ([System.IO.Path]::GetTempPath()) "jybnm"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $OutputRoot))
}
else {
  $OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
}

$consumerScript = Join-Path $RepositoryRoot "eng\Test-BridgePackageRuntimeConsumer.ps1"
if (-not (Test-Path -LiteralPath $consumerScript -PathType Leaf)) {
  throw "Bridge package runtime consumer script was not found: $consumerScript"
}

$powerShell = (Get-Command pwsh -ErrorAction Stop).Source
$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$assembledRuntimeCandidates = @(
  foreach ($downloadsRoot in @(
      (Join-Path (Split-Path -Parent $RepositoryRoot) "downloads"),
      (Join-Path $RepositoryRoot "downloads")) | Select-Object -Unique) {
    if (Test-Path -LiteralPath $downloadsRoot -PathType Container) {
      Get-ChildItem -LiteralPath $downloadsRoot -Directory -Recurse -Filter "assembled-runtime" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName
    }
  }
) | Select-Object -Unique
$rows = [System.Collections.Generic.List[object]]::new()
$failures = [System.Collections.Generic.List[string]]::new()

$runtimeIndex = 0
foreach ($runtimeKey in $runtimeKeys) {
  $runtimePackages = @($runtimeManifest.packages | Where-Object {
      [string]::Equals([string]$_.key, $runtimeKey, [System.StringComparison]::OrdinalIgnoreCase)
    })
  if ($runtimePackages.Count -ne 1) {
    throw "Runtime package key '$runtimeKey' must resolve to one runtime manifest entry. Found $($runtimePackages.Count)."
  }

  $rootsJson = (& $powerShell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") `
      -RuntimePackageKey $runtimeKey -RepositoryRoot $RepositoryRoot | Out-String).Trim()
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to resolve vendor roots for runtime package key '$runtimeKey'."
  }
  $resolvedRoots = $rootsJson | ConvertFrom-Json
  $tensorRtRuntimeRoot = Resolve-TensorRtRuntimeRoot `
    -RuntimePackage $runtimePackages[0] `
    -DefaultRoot ([string]$resolvedRoots.tensorRtRoot) `
    -AdditionalCandidates $assembledRuntimeCandidates

  $bridgePackageDirectory = Join-Path $BridgePackageRoot $runtimeKey
  $scenarioIndex = 0
  foreach ($scenario in $scenarios) {
    $scenarioOutputIndex = $scenarioIndex
    $scenarioIndex++
    $scenarioReportDirectory = Join-Path (Join-Path $ReportRoot $runtimeKey) $scenario
    $scenarioOutputRoot = Join-Path $OutputRoot ("r{0:D2}-s{1:D2}" -f $runtimeIndex, $scenarioOutputIndex)
    $arguments = @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      $consumerScript,
      "-SourceRuntimeKey",
      $runtimeKey,
      "-ManagedPackageDirectory",
      $ManagedPackageDirectory,
      "-BridgePackageDirectory",
      $bridgePackageDirectory,
      "-OutputRoot",
      $scenarioOutputRoot,
      "-ReportDirectory",
      $scenarioReportDirectory,
      "-DebugListenerScenario",
      $scenario,
      "-SkipBaselineValidation")

    $arguments += @("-TensorRtRoot", $tensorRtRuntimeRoot)
    if (-not [string]::IsNullOrWhiteSpace([string]$resolvedRoots.cudaRoot)) {
      $arguments += @("-CudaRoot", [string]$resolvedRoots.cudaRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$resolvedRoots.cudnnRoot)) {
      $arguments += @("-CudnnRoot", [string]$resolvedRoots.cudnnRoot)
    }

    $additionalSources = @(Expand-KeyList -Values $AdditionalPackageSource)
    if ($additionalSources.Count -gt 0) {
      $arguments += @("-AdditionalPackageSource", ($additionalSources -join ","))
    }
    if ($SkipInstalledVendorAssetHashing.IsPresent) {
      $arguments += "-SkipInstalledVendorAssetHashing"
    }

    Write-Host "Running bridge runtime negative control: key=$runtimeKey scenario=$scenario"
    & $powerShell @arguments
    $runnerExitCode = $LASTEXITCODE
    $reportPath = Join-Path $scenarioReportDirectory "bridge-package-runtime-consumer-proof.json"

    if ($runnerExitCode -ne 0 -or -not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
      $failures.Add("$runtimeKey/$scenario did not produce a valid report; runner exit code=$runnerExitCode.")
      $rows.Add([pscustomobject]@{
        runtimeKey = $runtimeKey
        scenario = $scenario
        status = "runner-failed"
        passed = $false
        runnerExitCode = $runnerExitCode
        reportPath = $reportPath
        tensorRtRuntimeRoot = $tensorRtRuntimeRoot
      })
      continue
    }

    try {
      $report = Get-Content -LiteralPath $reportPath -Raw -Encoding utf8 | ConvertFrom-Json
      $negative = $report.negativeControl
      $callbackState = $report.callbackStateSnapshot
      $callbackStateExpected = $scenario -ne "missing-vendor-dependency"
      $callbackStateObserved = Get-BooleanProperty -Object $callbackState -Name "observed"
      $callbackStateCoherent = Get-BooleanProperty -Object $callbackState -Name "coherent"
      $callbackStatePointerFree = Get-BooleanProperty -Object $callbackState -Name "pointerFree"
      $missingVendorObserved = Get-BooleanProperty -Object $negative -Name "missingVendorDependencyObserved"
      $callbackStateRequirementSatisfied = if ($callbackStateExpected) {
        $callbackStateObserved -and $callbackStateCoherent -and $callbackStatePointerFree
      }
      else {
        -not $callbackStateObserved -and -not $callbackStateCoherent -and -not $callbackStatePointerFree -and $missingVendorObserved
      }

      $proofFlags = [ordered]@{
        isRuntimeExecutionProof = Get-BooleanProperty -Object $report -Name "isRuntimeExecutionProof"
        isPackageConsumerRuntimeProof = Get-BooleanProperty -Object $report -Name "isPackageConsumerRuntimeProof"
        isLocalPackageDebugListenerCallbackRuntimeProof = Get-BooleanProperty -Object $report -Name "isLocalPackageDebugListenerCallbackRuntimeProof"
        canPromoteCompatibleHostRuntimeProof = Get-BooleanProperty -Object $report -Name "canPromoteCompatibleHostRuntimeProof"
        canPromoteRuntimeProof = Get-BooleanProperty -Object $report -Name "canPromoteRuntimeProof"
        canPublishPublicly = Get-BooleanProperty -Object $report -Name "canPublishPublicly"
        canCloseReleaseIssue = Get-BooleanProperty -Object $report -Name "canCloseReleaseIssue"
        debugListenerIsRealCallbackRuntimeProof = Get-BooleanProperty -Object $report.debugListenerCallback -Name "isRealCallbackRuntimeProof"
        debugListenerIsLocalPackageCallbackRuntimeProof = Get-BooleanProperty -Object $report.debugListenerCallback -Name "isLocalPackageCallbackRuntimeProof"
        sourceTreeProof = Get-BooleanProperty -Object $report.proofScopes.sourceTree -Name "isProof"
        localPackageProof = Get-BooleanProperty -Object $report.proofScopes.localPackage -Name "isProof"
        publicPackageProof = Get-BooleanProperty -Object $report.proofScopes.publicPackage -Name "isProof"
        postPublishProof = Get-BooleanProperty -Object $report.proofScopes.postPublish -Name "isProof"
      }
      $trueProofFlags = @($proofFlags.GetEnumerator() | Where-Object { [bool]$_.Value } | ForEach-Object { [string]$_.Key })
      $negativeContractSatisfied =
        [string]::Equals([string]$report.sourceRuntimeKey, $runtimeKey, [System.StringComparison]::Ordinal) -and
        [string]::Equals([string]$negative.scenario, $scenario, [System.StringComparison]::Ordinal) -and
        (Get-BooleanProperty -Object $negative -Name "requested") -and
        (Get-BooleanProperty -Object $negative -Name "passed") -and
        [string]::Equals([string]$report.smokeStatus, "expected-failure-verified", [System.StringComparison]::Ordinal) -and
        [string]::Equals([string]$report.proofClassification, "local-package-debug-listener-negative-control", [System.StringComparison]::Ordinal)
      $proofBoundarySatisfied = $trueProofFlags.Count -eq 0
      $passed = $runnerExitCode -eq 0 -and $negativeContractSatisfied -and $callbackStateRequirementSatisfied -and $proofBoundarySatisfied
      if (-not $passed) {
        $failures.Add("$runtimeKey/$scenario failed aggregate validation. negative=$negativeContractSatisfied callbackState=$callbackStateRequirementSatisfied proofFalse=$proofBoundarySatisfied")
      }

      $rows.Add([pscustomobject]@{
        runtimeKey = $runtimeKey
        scenario = $scenario
        status = if ($passed) { "expected-failure-verified" } else { "invalid-evidence" }
        passed = $passed
        runnerExitCode = $runnerExitCode
        reportPath = $reportPath
        tensorRtRuntimeRoot = $tensorRtRuntimeRoot
        reportSha256 = (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash
        negativeControlRequested = Get-BooleanProperty -Object $negative -Name "requested"
        negativeControlPassed = Get-BooleanProperty -Object $negative -Name "passed"
        callbackStateExpected = $callbackStateExpected
        callbackStateObserved = $callbackStateObserved
        callbackStateCoherent = $callbackStateCoherent
        callbackStatePointerFree = $callbackStatePointerFree
        callbackStateRequirementSatisfied = $callbackStateRequirementSatisfied
        callbackStateLastStatus = [string]$callbackState.lastStatus
        callbackStateLastOperation = [string]$callbackState.lastOperation
        missingVendorDependencyObserved = $missingVendorObserved
        proofBoundarySatisfied = $proofBoundarySatisfied
        trueProofFlags = @($trueProofFlags)
      })
    }
    catch {
      $failures.Add("$runtimeKey/$scenario report parsing failed: $($_.Exception.Message)")
      $rows.Add([pscustomobject]@{
        runtimeKey = $runtimeKey
        scenario = $scenario
        status = "invalid-report"
        passed = $false
        runnerExitCode = $runnerExitCode
        reportPath = $reportPath
        tensorRtRuntimeRoot = $tensorRtRuntimeRoot
      })
    }

  }

  $runtimeIndex++
}

$expectedRowCount = $runtimeKeys.Count * $scenarios.Count
$allRowsPassed = $rows.Count -eq $expectedRowCount -and @($rows | Where-Object { -not [bool]$_.passed }).Count -eq 0
$result = [ordered]@{
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  evidenceKind = "bridge-package-runtime-negative-control-matrix"
  runtimePackageKeys = @($runtimeKeys)
  scenarios = @($scenarios)
  expectedRowCount = $expectedRowCount
  actualRowCount = $rows.Count
  allRowsPassed = $allRowsPassed
  isRuntimeExecutionProof = $false
  isLocalPackageCallbackRuntimeProof = $false
  isPublicPackageProof = $false
  isPostPublishProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  proofBoundary = "Expected negative failures verify fail-closed behavior only. They are not runtime success, callback runtime, public-package, post-publish, publication, or release-close proof."
  failureCount = $failures.Count
  failures = @($failures.ToArray())
  rows = @($rows.ToArray())
}

New-Item -ItemType Directory -Path $ReportRoot -Force | Out-Null
$jsonPath = Join-Path $ReportRoot "bridge-package-runtime-negative-control-matrix.json"
$markdownPath = Join-Path $ReportRoot "bridge-package-runtime-negative-control-matrix.md"
$result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Bridge Package Runtime Negative-Control Matrix")
$lines.Add("")
$lines.Add("- evidence kind: ``$($result.evidenceKind)``")
$lines.Add("- rows: $($result.actualRowCount)/$($result.expectedRowCount)")
$lines.Add("- all rows passed: $($result.allRowsPassed)")
$lines.Add("- runtime/local callback/public/post-publish proof: False/False/False/False")
$lines.Add("- boundary: $($result.proofBoundary)")
$lines.Add("")
$lines.Add("| Runtime key | Scenario | Status | Callback expected | Observed | Coherent | Pointer-free | Missing vendor observed | Proof flags false | Report |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
foreach ($row in $rows) {
  $lines.Add("| $($row.runtimeKey) | $($row.scenario) | $($row.status) | $($row.callbackStateExpected) | $($row.callbackStateObserved) | $($row.callbackStateCoherent) | $($row.callbackStatePointerFree) | $($row.missingVendorDependencyObserved) | $($row.proofBoundarySatisfied) | ``$(ConvertTo-MarkdownCell -Value ([string]$row.reportPath))`` |")
}
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "BridgeRuntimeNegativeControlMatrixPassed=$allRowsPassed Rows=$($rows.Count)/$expectedRowCount Failures=$($failures.Count)"
Write-Host "Json=$jsonPath"
Write-Host "Markdown=$markdownPath"
if (-not $allRowsPassed -or $failures.Count -gt 0) {
  throw "Bridge package runtime negative-control matrix failed: $($failures -join ' | ')"
}
