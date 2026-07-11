[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [int]$MaxItems = 15,
  [switch]$IncludeMediumRisk
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

function Write-TextFileWithRetry {
  param(
    [string]$Path,
    [string]$Value,
    [System.Text.Encoding]$Encoding,
    [int]$RetryCount = 40,
    [int]$DelayMilliseconds = 250
  )

  $directory = Split-Path -Parent $Path
  New-Item -ItemType Directory -Force -Path $directory | Out-Null

  $tempFileName = ".{0}.{1}.tmp" -f ([System.IO.Path]::GetFileName($Path)), ([System.Guid]::NewGuid().ToString("N"))
  $tempPath = Join-Path $directory $tempFileName

  try {
    [System.IO.File]::WriteAllText($tempPath, $Value, $Encoding)

    for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
      try {
        Move-Item -LiteralPath $tempPath -Destination $Path -Force
        return
      }
      catch {
        if ($attempt -eq $RetryCount) {
          throw
        }

        Start-Sleep -Milliseconds $DelayMilliseconds
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
  }
}

$matrixPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
if (-not (Test-Path -LiteralPath $matrixPath -PathType Leaf)) {
  throw "TensorRT interface comparison CSV was not found: $matrixPath"
}

function Test-UnsafeBoundary {
  param(
    [string]$Class,
    [string]$Method,
    [string]$Interface
  )

  $classText = $Class.ToLowerInvariant()
  $methodText = $Method.ToLowerInvariant()
  $interfaceText = $Interface.ToLowerInvariant()
  $combined = "$classText::$methodText $interfaceText"

  if ($classText -match "callback|listener|allocator|progressmonitor|loggerfinder|streamreader|streamwriter|pluginv2|pluginv3|iversionedinterface|ipluginresource|ipluginresourcecontext|ialgorithm|ialgorithmcontext|ialgorithmioinfo|ialgorithmvariant|ialgorithmselector") {
    return $true
  }

  if ($methodText -match "^(add|remove|enable|disable|mark|unmark|reduce|create|destroy|clone|enqueue|register|deregister|acquire|release|allocate|deallocate|free|set|clear|reset|report|select|build|parse|read|write)") {
    return $true
  }

  if ($combined -match "void\*|intptr|callback|trampoline|borrowed pointer|owner-owned|pluginv2|pluginv3") {
    return $true
  }

  if ($methodText -match "tensor|buffer|allocator|weight|state|plugin|resource|library|shapevalues|shapebinding|profile.*values") {
    return $true
  }

  return $false
}

function Test-ObjectOrBorrowedReturnCandidate {
  param(
    [string]$Class,
    [string]$Method,
    [string]$Interface
  )

  $methodText = $Method.ToLowerInvariant()
  $interfaceText = $Interface.ToLowerInvariant()
  $classText = $Class.ToLowerInvariant()
  $combined = "$Class::$Method $Interface".ToLowerInvariant()

  if ($classText -match "idimensionexpr|iexprbuilder|ionnxconfig|iversionedinterface|irnnv2layer") {
    return $true
  }

  if ($methodText -match "^(getlogger|geterrorrecorder|getalgorithmselector|getint8calibrator|getbuilder|getruntime|getpluginregistry|getplugincreator|getplugincreatorlist|getplugincreatorinterface|getalgorithmvariant|getalgorithmioinfo|get.+byindex)") {
    return $true
  }

  if ($combined -match "list|registry|creator|recorder|selector|calibrator|logger|tensor|buffer|allocator|weights|state|pluginresource|shapevalues|shapebinding|profile.*values") {
    return $true
  }

  return $false
}

function Test-ReadonlyShape {
  param([string]$Method)

  $methodText = $Method.ToLowerInvariant()
  return $methodText -match "^(get|has|is|can|count|num)" -or
    $methodText -match "(count|name|version|namespace|shape|dtype|format|metadata|error|profil|timing|workspace|dimension|strides|tactic|implementation|flag|level|verbosity|platform|capability|datatype|componentsperelement|vectorizeddim|nberrors|nbinputs|nboutputs|nbplugins)"
}

function Get-Priority {
  param(
    [string]$Class,
    [string]$Method,
    [string]$Interface
  )

  if (Test-UnsafeBoundary -Class $Class -Method $Method -Interface $Interface) {
    return 90
  }

  if (Test-ObjectOrBorrowedReturnCandidate -Class $Class -Method $Method -Interface $Interface) {
    return 50
  }

  $methodText = $Method.ToLowerInvariant()
  if ($methodText -match "^(getnb|count|num)|count|nberrors|nbinputs|nboutputs|nbplugins") {
    return 0
  }

  if ($methodText -match "^(has|is|can)") {
    return 1
  }

  if ($methodText -match "timing|workspace|size|level|verbosity|platform|capability|datatype|format|flag|tactic|implementation|dimension|strides|shape|dtype") {
    return 2
  }

  if ($methodText -match "name|version|namespace|metadata|error|profil|diagnostic") {
    return 3
  }

  return 20
}

function Get-Risk {
  param(
    [string]$Class,
    [string]$Method,
    [string]$Interface
  )

  if (Test-UnsafeBoundary -Class $Class -Method $Method -Interface $Interface) {
    return "high"
  }

  if (-not (Test-ReadonlyShape -Method $Method)) {
    return "medium"
  }

  if (Test-ObjectOrBorrowedReturnCandidate -Class $Class -Method $Method -Interface $Interface) {
    return "medium"
  }

  return "low"
}

function Get-RiskRank {
  param([string]$Risk)

  if ($Risk -eq "low") {
    return 0
  }

  if ($Risk -eq "medium") {
    return 1
  }

  return 2
}

function Get-ManualDesignGroup {
  param(
    [string]$Class,
    [string]$Method,
    [string]$Interface
  )

  $classText = $Class.ToLowerInvariant()
  $methodText = $Method.ToLowerInvariant()
  $interfaceText = $Interface.ToLowerInvariant()

  if ($classText -eq "iexecutioncontext" -and $methodText -match "^execute") {
    return "runtime-execution-boundary"
  }

  if ($classText -eq "iruntime" -and ($methodText -match "^deserializecudaengine" -or $methodText -eq "loadruntime")) {
    return "runtime-deserialization-boundary"
  }

  if ($classText -eq "idimensionexpr") {
    return "dimension-expression-snapshot-design"
  }

  if ($classText -eq "iexprbuilder") {
    return "expression-builder-design-gate"
  }

  if ($classText -match "^ialgorithm" -or $interfaceText -match "ialgorithm") {
    return "algorithm-selector-ownership-boundary"
  }

  if ($classText -eq "ierrorrecorder") {
    return "error-recorder-diagnostics-design"
  }

  if ($classText -match "iint8.*calibrator" -or $interfaceText -match "iint8.*calibrator") {
    return "calibrator-callback-metadata-design"
  }

  if ($interfaceText -match "plugin|resource|creator|registry") {
    return "plugin-ownership-boundary"
  }

  if ($interfaceText -match "callback|listener|allocator") {
    return "callback-allocator-boundary"
  }

  return "manual-review-other"
}

function Get-ManualDesignRecommendation {
  param([string]$DesignGroup)

  switch ($DesignGroup) {
    "runtime-execution-boundary" { return "Do not promote as a deferred cleanup item. Design binding-buffer ownership, stream/profile state, exception-to-status mapping, and real runtime smoke first." }
    "runtime-deserialization-boundary" { return "Do not promote directly. Design bridge-owned engine lifetime, serialized buffer copy policy, plugin/library dependency diagnostics, and package-consumer runtime proof first." }
    "dimension-expression-snapshot-design" { return "Promote only through a pointer-free snapshot tied to a known owner object. Do not expose borrowed IDimensionExpr pointers." }
    "expression-builder-design-gate" { return "Keep as a design gate until expression ownership and lifetime are modeled. Do not create expression nodes through public API yet." }
    "algorithm-selector-ownership-boundary" { return "Keep algorithm selector callbacks and borrowed IAlgorithm/IAlgorithmContext/IAlgorithmIOInfo/IAlgorithmVariant snapshots deferred until selector ownership and callback result lifetime are modeled. Do not expose borrowed algorithm pointers." }
    "error-recorder-diagnostics-design" { return "Prefer copied diagnostics and presence/snapshot APIs. Do not expose recorder pointers or make ref-count calls public ownership controls." }
    "calibrator-callback-metadata-design" { return "Limit to presence and safe metadata. Do not invoke calibration callbacks or take ownership of batch buffers." }
    "plugin-ownership-boundary" { return "Use count/copy or copied metadata only. Do not expose borrowed plugin creators, resources, or registries." }
    "callback-allocator-boundary" { return "Treat as callback/allocator design work. Require owner ledger, no-throw callback boundary, and runtime proof before promotion." }
    default { return "Manual review is required before promotion. Define pointer-free lifetime, native/source/wrapper/docs/tests, and version guards first." }
  }
}

function Test-SafeAlternativeOrAliasCandidate {
  param(
    [string]$Interface,
    [string]$ImplementationStatus,
    [string]$NativeManifestStatus,
    [string]$NativeSourceStatus,
    [string]$ManagedInteropStatus,
    [string]$MatchedManifestIds
  )

  $combined = "$Interface $ImplementationStatus $NativeManifestStatus $NativeSourceStatus $ManagedInteropStatus $MatchedManifestIds".ToLowerInvariant()

  if ($ImplementationStatus -eq "implemented-with-deferred-history") {
    return $true
  }

  return $combined -match "safe|snapshot|has-|has_|count|copy|alias|inventory|probe|diagnostic|metadata|presence"
}

function Test-KeepDeferredBoundary {
  param(
    [string]$Class,
    [string]$Method,
    [string]$Interface
  )

  $combined = "$Class::$Method $Interface".ToLowerInvariant()

  if ($combined -match "loadlibrary|deregisterlibrary|registercreator|deregistercreator|pluginresource|acquirepluginresource|releasepluginresource") {
    return $true
  }

  if ($combined -match "createplugin|clone|enqueue|pluginv2|pluginv3|callback|trampoline|listener|allocator|borrowed pointer") {
    return $true
  }

  return $false
}

function Get-SafetyTier {
  param(
    [string]$Class,
    [string]$Method,
    [string]$Interface,
    [string]$OwnershipRisk,
    [string]$DesignGroup,
    [string]$ImplementationStatus,
    [string]$NativeManifestStatus,
    [string]$NativeSourceStatus,
    [string]$ManagedInteropStatus,
    [string]$MatchedManifestIds
  )

  if (Test-KeepDeferredBoundary -Class $Class -Method $Method -Interface $Interface) {
    return "D - keep-deferred"
  }

  if (Test-SafeAlternativeOrAliasCandidate -Interface $Interface -ImplementationStatus $ImplementationStatus -NativeManifestStatus $NativeManifestStatus -NativeSourceStatus $NativeSourceStatus -ManagedInteropStatus $ManagedInteropStatus -MatchedManifestIds $MatchedManifestIds) {
    return "B - safe-alternative-or-alias"
  }

  if ($OwnershipRisk -eq "low" -and $DesignGroup -eq "manual-review-other") {
    return "A - immediate-safe"
  }

  return "C - design-gate-required"
}

function Get-SafetyTierAction {
  param([string]$SafetyTier)

  switch ($SafetyTier) {
    "A - immediate-safe" { return "promote-real-api" }
    "B - safe-alternative-or-alias" { return "alias-or-proof-safe-alternative" }
    "C - design-gate-required" { return "design-gate-required" }
    default { return "keep-deferred" }
  }
}

function Get-SafetyTierReason {
  param(
    [string]$SafetyTier,
    [string]$DesignGroup,
    [string]$PromotionBoundary
  )

  switch ($SafetyTier) {
    "A - immediate-safe" { return "Low-risk readonly scalar/count/copy-style row; still requires manifest, native/source, wrapper, docs, smoke or quality tests, and version guards." }
    "B - safe-alternative-or-alias" { return "Already has deferred-history implementation or a copied snapshot/presence/count/copy/probe alternative; close by alias/proof without deleting deferred history." }
    "C - design-gate-required" { return "Manual design gate required for $DesignGroup. $PromotionBoundary" }
    default { return "Keep deferred until ownership, registry/library lifecycle, plugin resource, plugin instance, callback trampoline, or borrowed pointer boundary has a safe public object model and runtime proof." }
  }
}

$rows = Import-Csv -LiteralPath $matrixPath
$deferredRows = @($rows | Where-Object {
    $_.ImplementationStatus -eq "deferred-only" -or
    $_.NativeManifestStatus -match "deferred" -or
    $_.NativeSourceStatus -match "deferred" -or
    $_.ManagedInteropStatus -match "deferred"
  })

$annotatedCandidates = @(
  $deferredRows |
    ForEach-Object {
      $risk = Get-Risk -Class $_.Class -Method $_.Method -Interface $_.Interface
      $designGroup = Get-ManualDesignGroup -Class $_.Class -Method $_.Method -Interface $_.Interface
      [pscustomobject]@{
        interface = $_.Interface
        class = $_.Class
        method = $_.Method
        tensorRtLine = $_.TensorRtLine
        header = $_.Header
        implementationStatus = $_.ImplementationStatus
        nativeManifestStatus = $_.NativeManifestStatus
        nativeSourceStatus = $_.NativeSourceStatus
        managedInteropStatus = $_.ManagedInteropStatus
        matchedManifestIds = $_.MatchedManifestIds
        priority = Get-Priority -Class $_.Class -Method $_.Method -Interface $_.Interface
        ownershipRisk = $risk
        recommendedAction = if ($risk -eq "low") { "candidate-for-readonly-diagnostic-promotion" } elseif ($risk -eq "medium") { "manual-review-before-promotion" } else { "defer-ownership-or-callback-boundary" }
        designGroup = $designGroup
        canPromoteWithoutDesignGate = $risk -eq "low"
        promotionBoundary = Get-ManualDesignRecommendation -DesignGroup $designGroup
      }
    }
)

$triageRows = @(
  $rows |
    Where-Object {
      $_.ImplementationStatus -eq "deferred-only" -or
      $_.ImplementationStatus -eq "implemented-with-deferred-history" -or
      $_.NativeManifestStatus -match "deferred" -or
      $_.NativeSourceStatus -match "deferred" -or
      $_.ManagedInteropStatus -match "deferred"
    } |
    ForEach-Object {
      $risk = Get-Risk -Class $_.Class -Method $_.Method -Interface $_.Interface
      $designGroup = Get-ManualDesignGroup -Class $_.Class -Method $_.Method -Interface $_.Interface
      $boundary = Get-ManualDesignRecommendation -DesignGroup $designGroup
      $tier = Get-SafetyTier `
        -Class $_.Class `
        -Method $_.Method `
        -Interface $_.Interface `
        -OwnershipRisk $risk `
        -DesignGroup $designGroup `
        -ImplementationStatus $_.ImplementationStatus `
        -NativeManifestStatus $_.NativeManifestStatus `
        -NativeSourceStatus $_.NativeSourceStatus `
        -ManagedInteropStatus $_.ManagedInteropStatus `
        -MatchedManifestIds $_.MatchedManifestIds

      [pscustomobject]@{
        interface = $_.Interface
        class = $_.Class
        method = $_.Method
        tensorRtLine = $_.TensorRtLine
        header = $_.Header
        implementationStatus = $_.ImplementationStatus
        nativeManifestStatus = $_.NativeManifestStatus
        nativeSourceStatus = $_.NativeSourceStatus
        managedInteropStatus = $_.ManagedInteropStatus
        matchedManifestIds = $_.MatchedManifestIds
        ownershipRisk = $risk
        designGroup = $designGroup
        safetyTier = $tier
        recommendedAction = Get-SafetyTierAction -SafetyTier $tier
        reason = Get-SafetyTierReason -SafetyTier $tier -DesignGroup $designGroup -PromotionBoundary $boundary
      }
    }
)

$tierOrder = @{
  "A - immediate-safe" = 0
  "B - safe-alternative-or-alias" = 1
  "C - design-gate-required" = 2
  "D - keep-deferred" = 3
}

$tierSummaries = @(
  @("A - immediate-safe", "B - safe-alternative-or-alias", "C - design-gate-required", "D - keep-deferred") |
    ForEach-Object {
      $tierName = $_
      $tierRows = @($triageRows | Where-Object { $_.safetyTier -eq $tierName })
      $designGroups = @($tierRows | Select-Object -ExpandProperty designGroup -Unique | Sort-Object)
      [pscustomobject]@{
        safetyTier = $tierName
        recommendedAction = Get-SafetyTierAction -SafetyTier $tierName
        candidateCount = $tierRows.Count
        designGroups = $designGroups
        representativeInterfaces = @(
          $tierRows |
            Sort-Object @{ Expression = { Get-RiskRank -Risk $_.ownershipRisk } }, class, method, tensorRtLine |
            Select-Object -Property interface, tensorRtLine -Unique |
            Select-Object -First 10 |
            ForEach-Object { "$($_.interface) [TRT$($_.tensorRtLine)]" }
        )
      }
    }
)

$eligibleCandidates = @(
  $annotatedCandidates |
    Where-Object {
      $_.ownershipRisk -eq "low" -or
      ($IncludeMediumRisk.IsPresent -and $_.ownershipRisk -eq "medium")
    }
)

$deduplicatedCandidates = @(
  $eligibleCandidates |
    Group-Object -Property interface, tensorRtLine |
    ForEach-Object {
      $_.Group |
        Sort-Object priority, @{ Expression = { Get-RiskRank -Risk $_.ownershipRisk } }, class, method, matchedManifestIds |
        Select-Object -First 1
    }
)

$candidates = @(
  $deduplicatedCandidates |
    Sort-Object priority, @{ Expression = { Get-RiskRank -Risk $_.ownershipRisk } }, class, method, tensorRtLine |
    Select-Object -First $MaxItems
)

$manualReviewDesignGroups = @(
  $annotatedCandidates |
    Where-Object { $_.ownershipRisk -eq "medium" -or $_.ownershipRisk -eq "high" } |
    Group-Object -Property designGroup |
    Sort-Object Count, Name -Descending |
    ForEach-Object {
      $representatives = @(
        $_.Group |
          Sort-Object priority, class, method, tensorRtLine |
          Select-Object -Property interface, tensorRtLine -Unique |
          Select-Object -First 8 |
          ForEach-Object { "$($_.interface) [TRT$($_.tensorRtLine)]" }
      )

      [pscustomobject]@{
        designGroup = $_.Name
        candidateCount = $_.Count
        ownershipRiskLevels = @($_.Group | Select-Object -ExpandProperty ownershipRisk -Unique | Sort-Object)
        canPromoteWithoutDesignGate = $false
        recommendedDesignAction = Get-ManualDesignRecommendation -DesignGroup $_.Name
        representativeInterfaces = $representatives
      }
    }
)

$selectedDesignGroups = @(
  $candidates |
    Group-Object -Property designGroup |
    Sort-Object Count, Name -Descending |
    ForEach-Object {
      [pscustomobject]@{
        designGroup = $_.Name
        selectedCandidateCount = $_.Count
        recommendedDesignAction = Get-ManualDesignRecommendation -DesignGroup $_.Name
      }
    }
)

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  planKind = "deferred-readonly-api-candidate-plan"
  sourceMatrix = "artifacts/interface-coverage/tensorrt-interface-comparison.csv"
  totalDeferredRowCount = $deferredRows.Count
  annotatedCandidateCount = $annotatedCandidates.Count
  eligibleCandidateCount = $eligibleCandidates.Count
  highRiskDeferredRowCount = @($annotatedCandidates | Where-Object { $_.ownershipRisk -eq "high" }).Count
  mediumRiskDeferredRowCount = @($annotatedCandidates | Where-Object { $_.ownershipRisk -eq "medium" }).Count
  lowRiskDeferredRowCount = @($annotatedCandidates | Where-Object { $_.ownershipRisk -eq "low" }).Count
  deduplicatedEligibleCandidateCount = $deduplicatedCandidates.Count
  selectedCandidateCount = $candidates.Count
  maxItems = $MaxItems
  includeMediumRisk = [bool]$IncludeMediumRisk.IsPresent
  manualReviewDesignGroupCount = $manualReviewDesignGroups.Count
  selectionPolicy = @(
    "Default selection includes active deferred-only low-risk rows only; pass -IncludeMediumRisk when a human intentionally wants manual-review rows.",
    "Prefer scalar get/has/is/can/count/name/version/namespace/shape/dtype/format/metadata/error/profiling/diagnostic interfaces.",
    "Exclude callback, allocator, listener, algorithm selector callback, IAlgorithm result snapshots, add/remove/enable/disable/reduce, register/deregister, enqueue, clone, create/destroy, acquire/release, set/clear/reset, plugin V2/V3 trampoline, tensor/buffer/allocator/weights/state/resource/plugin path rows, shape-values/profile-values/shape-binding array rows, and borrowed pointer ownership rows.",
    "Keep IAlgorithm/IAlgorithmContext/IAlgorithmIOInfo/IAlgorithmVariant, IDimensionExpr, IExprBuilder, IOnnxConfig, IVersionedInterface base-pointer rows, IRNNv2Layer borrowed-state rows, IPluginResource, and IPluginResourceContext in manual-review or ownership-boundary buckets until bridge-owned lifetime and pointer-free object models are available.",
    "Deduplicate by interface and TensorRT major line before selecting the top candidates.",
    "Each promoted batch must update manifest, native source, generated interop, high-level C# wrapper, docs, smoke or quality tests, and cross-version guards."
  )
  manualReviewDesignGroups = $manualReviewDesignGroups
  selectedDesignGroups = $selectedDesignGroups
  candidates = $candidates
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\interface-coverage"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "deferred-readonly-api-candidate-plan.json"
$markdownPath = Join-Path $outputRoot "deferred-readonly-api-candidate-plan.md"
$triageJsonPath = Join-Path $outputRoot "deferred-candidate-safety-triage.json"
$triageMarkdownPath = Join-Path $outputRoot "deferred-candidate-safety-triage.md"

Write-TextFileWithRetry -Path $jsonPath -Value ($summary | ConvertTo-Json -Depth 8) -Encoding $utf8

$triageSummary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  triageKind = "deferred-candidate-safety-triage"
  sourceMatrix = "artifacts/interface-coverage/tensorrt-interface-comparison.csv"
  totalTriageRowCount = $triageRows.Count
  tierSummaries = $tierSummaries
  policy = @(
    "A rows are the only immediate-safe promotion candidates, and still require real native/source/wrapper/docs/tests changes.",
    "B rows are safe alternatives, aliases, or proof closures for implemented-with-deferred-history rows; do not delete deferred history.",
    "C rows require a design gate before public API promotion.",
    "D rows remain deferred because ownership, plugin resource, plugin instance, callback trampoline, registry library lifecycle, or borrowed pointer boundaries are not safe.",
    "Do not treat this triage as permission to delete deferred records."
  )
  rows = @($triageRows | Sort-Object @{ Expression = { $tierOrder[$_.safetyTier] } }, @{ Expression = { Get-RiskRank -Risk $_.ownershipRisk } }, designGroup, class, method, tensorRtLine)
}

Write-TextFileWithRetry -Path $triageJsonPath -Value ($triageSummary | ConvertTo-Json -Depth 8) -Encoding $utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Deferred Readonly API Candidate Plan")
$lines.Add("")
$lines.Add("- generated at UTC: ``$($summary.generatedAtUtc)``")
$lines.Add("- source matrix: ``$($summary.sourceMatrix)``")
$lines.Add("- total deferred rows: $($summary.totalDeferredRowCount)")
$lines.Add("- low-risk deferred rows: $($summary.lowRiskDeferredRowCount)")
$lines.Add("- medium-risk deferred rows: $($summary.mediumRiskDeferredRowCount)")
$lines.Add("- high-risk deferred rows: $($summary.highRiskDeferredRowCount)")
$lines.Add("- eligible candidates: $($summary.eligibleCandidateCount)")
$lines.Add("- deduplicated eligible candidates: $($summary.deduplicatedEligibleCandidateCount)")
$lines.Add("- selected candidates: $($summary.selectedCandidateCount)")
$lines.Add("- include medium risk: $($summary.includeMediumRisk)")
$lines.Add("- manual review design groups: $($summary.manualReviewDesignGroupCount)")
$lines.Add("")
$lines.Add("## Selection Policy")
$lines.Add("")
foreach ($policy in $summary.selectionPolicy) {
  $lines.Add("- $policy")
}
$lines.Add("")
$lines.Add("## Manual Review Design Groups")
$lines.Add("")
$lines.Add("| Design group | Count | Risk levels | Promote without design gate | Recommended action | Representatives |")
$lines.Add("| --- | ---: | --- | --- | --- | --- |")
foreach ($group in $manualReviewDesignGroups) {
  $representatives = (($group.representativeInterfaces | ForEach-Object { "``$_``" }) -join "<br>")
  $riskLevels = (($group.ownershipRiskLevels | ForEach-Object { "``$_``" }) -join ", ")
  $lines.Add("| ``$($group.designGroup)`` | $($group.candidateCount) | $riskLevels | $($group.canPromoteWithoutDesignGate) | $($group.recommendedDesignAction) | $representatives |")
}
$lines.Add("")
$lines.Add("High-risk design groups are boundary planning input only. They are not promotion lists, even when -IncludeMediumRisk is used.")
$lines.Add("")
$lines.Add("## Selected Design Groups")
$lines.Add("")
$lines.Add("| Design group | Selected candidates | Recommended action |")
$lines.Add("| --- | ---: | --- |")
foreach ($group in $selectedDesignGroups) {
  $lines.Add("| ``$($group.designGroup)`` | $($group.selectedCandidateCount) | $($group.recommendedDesignAction) |")
}
$lines.Add("")
$lines.Add("## Candidates")
$lines.Add("")
$lines.Add("| Interface | TRT | Risk | Design group | Action | Boundary | Manifest | Status |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- | --- |")
foreach ($candidate in $candidates) {
  $lines.Add("| ``$($candidate.interface)`` | $($candidate.tensorRtLine) | $($candidate.ownershipRisk) | ``$($candidate.designGroup)`` | ``$($candidate.recommendedAction)`` | $($candidate.promotionBoundary) | ``$($candidate.matchedManifestIds)`` | ``$($candidate.implementationStatus)`` |")
}
if ($candidates.Count -eq 0) {
  $lines.Add("")
  $lines.Add("No default low-risk deferred rows remain after applying borrowed-pointer, base-interface, callback, plugin, and caller-buffer array guards. Treat the remaining deferred rows as manual design work, not as automatically promotable inventory.")
}
$lines.Add("")
$lines.Add("These candidates are planning input only. Do not delete deferred records to manufacture completion; promote rows only when native implementation, C# wrapper, docs, and tests are all updated.")

Write-TextFileWithRetry -Path $markdownPath -Value ($lines -join "`r`n") -Encoding $utf8

$triageLines = New-Object System.Collections.Generic.List[string]
$triageLines.Add("# Deferred Candidate Safety Triage")
$triageLines.Add("")
$triageLines.Add("- generated at UTC: ``$($triageSummary.generatedAtUtc)``")
$triageLines.Add("- source matrix: ``$($triageSummary.sourceMatrix)``")
$triageLines.Add("- total triage rows: $($triageSummary.totalTriageRowCount)")
$triageLines.Add("")
$triageLines.Add("## Safety Tier Summary")
$triageLines.Add("")
$triageLines.Add("| Safety tier | Count | Action | Design groups | Representatives |")
$triageLines.Add("| --- | ---: | --- | --- | --- |")
foreach ($tier in $tierSummaries) {
  $designGroups = if ($tier.designGroups.Count -gt 0) { (($tier.designGroups | ForEach-Object { "``$_``" }) -join ", ") } else { "" }
  $representatives = (($tier.representativeInterfaces | ForEach-Object { "``$_``" }) -join "<br>")
  $triageLines.Add("| $($tier.safetyTier) | $($tier.candidateCount) | ``$($tier.recommendedAction)`` | $designGroups | $representatives |")
}
$triageLines.Add("")
$triageLines.Add("A - immediate-safe rows are candidates for real API work only after native/source/wrapper/docs/tests/version guards are updated.")
$triageLines.Add("B - safe-alternative-or-alias rows are proof or alias closures for already-safe alternatives and implemented-with-deferred-history records.")
$triageLines.Add("C - design-gate-required rows need explicit owner/lifetime/object-model design before promotion.")
$triageLines.Add("D - keep-deferred rows include registry load/unload, plugin resource acquire/release, plugin instance create/clone/enqueue, callback trampolines, and borrowed pointer boundaries.")
$triageLines.Add("")
$triageLines.Add("Do not treat this triage as permission to delete deferred records.")
$triageLines.Add("")
$triageLines.Add("## Rows")
$triageLines.Add("")
$triageLines.Add("| Tier | Interface | TRT | Risk | Design group | Action | Reason |")
$triageLines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($row in $triageSummary.rows) {
  $triageLines.Add("| $($row.safetyTier) | ``$($row.interface)`` | $($row.tensorRtLine) | $($row.ownershipRisk) | ``$($row.designGroup)`` | ``$($row.recommendedAction)`` | $($row.reason) |")
}

Write-TextFileWithRetry -Path $triageMarkdownPath -Value ($triageLines -join "`r`n") -Encoding $utf8

Write-Host "Deferred readonly API candidate plan written to $jsonPath"
Write-Host "Deferred readonly API candidate plan written to $markdownPath"
Write-Host "Deferred candidate safety triage written to $triageJsonPath"
Write-Host "Deferred candidate safety triage written to $triageMarkdownPath"
