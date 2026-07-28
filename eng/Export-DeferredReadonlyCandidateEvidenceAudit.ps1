[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$CandidatePath = "artifacts/interface-coverage/deferred-readonly-candidate-list.json",
  [string]$EvidenceMapPath = "eng/deferred-readonly-candidate-evidence-map.json",
  [string]$OutputRoot = "artifacts/interface-coverage"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$RelativePath)

  return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot ($RelativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar))))
}

function Convert-ToRepositoryPath {
  param([string]$Path)

  return $Path.Replace('\', '/')
}

function Read-JsonArtifact {
  param([string]$RelativePath)

  $path = Resolve-RepositoryPath $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required JSON artifact was not found: $path"
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object -or -not ($Object.PSObject.Properties.Name -contains $Name)) {
    return $DefaultValue
  }

  return $Object.PSObject.Properties[$Name].Value
}

function Get-ArrayOrEmpty {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue @()
  if ($null -eq $value) {
    return @()
  }

  return @($value)
}

function New-List {
  return ,([System.Collections.Generic.List[object]]::new())
}

function Add-Finding {
  param(
    [System.Collections.Generic.List[object]]$Findings,
    [string]$CandidateId,
    [string]$Kind,
    [string]$Detail
  )

  $Findings.Add([pscustomobject][ordered]@{
      candidateId = $CandidateId
      kind = $Kind
      detail = $Detail
    })
}

function Write-TextFile {
  param(
    [string]$Path,
    [string]$Value
  )

  $directory = Split-Path -Parent $Path
  New-Item -ItemType Directory -Force -Path $directory | Out-Null
  $temporaryPath = Join-Path $directory (".{0}.{1}.tmp" -f [System.IO.Path]::GetFileName($Path), [Guid]::NewGuid().ToString("N"))
  try {
    [System.IO.File]::WriteAllText($temporaryPath, $Value, $utf8)
    Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
  }
  finally {
    if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
      Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    }
  }
}

$candidateList = Read-JsonArtifact $CandidatePath
$evidenceMap = Read-JsonArtifact $EvidenceMapPath
$evidenceMapByCandidateId = @{}
foreach ($mapRow in @(Get-ArrayOrEmpty -Object $evidenceMap -Name "candidates")) {
  $mapCandidateId = [string](Get-PropertyOrDefault -Object $mapRow -Name "candidateId" -DefaultValue "")
  if ([string]::IsNullOrWhiteSpace($mapCandidateId)) {
    throw "Evidence map contains a row without candidateId."
  }
  if ($evidenceMapByCandidateId.ContainsKey($mapCandidateId)) {
    throw "Evidence map contains duplicate candidateId: $mapCandidateId"
  }
  $evidenceMapByCandidateId[$mapCandidateId] = $mapRow
}
$candidateRows = New-List
$allFindings = New-List
$manifestIndex = @{}
$manifestRoot = Resolve-RepositoryPath "native/manifests/tensorrt"

foreach ($manifestFile in @(Get-ChildItem -LiteralPath $manifestRoot -Filter "*.manifest.json" -File -Recurse | Sort-Object FullName)) {
  $relativePath = Convert-ToRepositoryPath ([System.IO.Path]::GetRelativePath($RepositoryRoot, $manifestFile.FullName))
  $raw = Get-Content -LiteralPath $manifestFile.FullName -Raw -Encoding utf8
  $parsed = $null
  $parseError = $null
  try {
    $parsed = $raw | ConvertFrom-Json
  }
  catch {
    $parseError = $_.Exception.Message
  }

  $manifestIndex[$relativePath] = [pscustomobject][ordered]@{
    path = $relativePath
    raw = $raw
    parsed = $parsed
    parseError = $parseError
    versionLineFromPath = if ($relativePath -match '/v(\d+)/') { $Matches[1] } else { "" }
  }
}

$groups = $candidateList.groups.PSObject.Properties | Sort-Object Name
foreach ($groupProperty in $groups) {
  foreach ($candidate in @(Get-ArrayOrEmpty -Object $candidateList.groups -Name $groupProperty.Name)) {
    $candidateId = [string](Get-PropertyOrDefault -Object $candidate -Name "candidateId" -DefaultValue "")
    $status = [string](Get-PropertyOrDefault -Object $candidate -Name "implementationStatus" -DefaultValue "")
    $nativeRequired = [bool](Get-PropertyOrDefault -Object $candidate -Name "nativeLayerRequired" -DefaultValue $false)
    $implementationEvidence = Get-PropertyOrDefault -Object $candidate -Name "implementationEvidence" -DefaultValue $null
    $supplement = if ($evidenceMapByCandidateId.ContainsKey($candidateId)) { $evidenceMapByCandidateId[$candidateId] } else { $null }
    $candidateFindings = New-List

    $nativeSources = @(
      @(Get-ArrayOrEmpty -Object $implementationEvidence -Name "nativeSources") +
      @(Get-ArrayOrEmpty -Object $supplement -Name "additionalNativeSources") |
        Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
        Sort-Object -Unique
    )
    $manifestSources = @(
      @(Get-ArrayOrEmpty -Object $implementationEvidence -Name "manifestSources") +
      @(Get-ArrayOrEmpty -Object $supplement -Name "manifestSources") |
        Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
        Sort-Object -Unique
    )
    $managedSources = @(
      @(Get-ArrayOrEmpty -Object $implementationEvidence -Name "managedSources") +
      @(Get-ArrayOrEmpty -Object $supplement -Name "additionalManagedSources") |
        Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
        Sort-Object -Unique
    )
    $smokeSources = @(Get-ArrayOrEmpty -Object $implementationEvidence -Name "smokeSources")
    $qualityTests = @(Get-ArrayOrEmpty -Object $implementationEvidence -Name "qualityTests")
    $publicSurface = @(Get-ArrayOrEmpty -Object $implementationEvidence -Name "publicSurface")
    $allEvidencePaths = @($nativeSources + $manifestSources + $managedSources + $smokeSources + $qualityTests)
    $missingPaths = New-List
    foreach ($relativePath in $allEvidencePaths) {
      $resolvedPath = Resolve-RepositoryPath ([string]$relativePath)
      if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
        $missingPaths.Add([string]$relativePath)
        Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "missing-evidence-path" -Detail ([string]$relativePath)
      }
    }

    $nativeTexts = New-List
    foreach ($relativePath in $nativeSources) {
      $resolvedPath = Resolve-RepositoryPath ([string]$relativePath)
      if (Test-Path -LiteralPath $resolvedPath -PathType Leaf) {
        $nativeTexts.Add((Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8))
      }
    }

    $nativeText = [string]::Join("`n", @($nativeTexts))
    $nativeEntryPoints = @([regex]::Matches($nativeText, '\bjyppx_trt(?:8|10|11)_[A-Za-z0-9_]+\b') | ForEach-Object { $_.Value } | Sort-Object -Unique)
    $hasPluginMacro = $nativeText -match 'JYPPX_TRT_PLUGIN_FN|JYPPX_TRT_PLUGIN_PREFIX'
    $manifestRows = New-List
    foreach ($relativePath in $manifestSources) {
      $normalizedManifestPath = Convert-ToRepositoryPath ([string]$relativePath)
      if (-not $manifestIndex.ContainsKey($normalizedManifestPath)) {
        Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "manifest-index-miss" -Detail $normalizedManifestPath
        continue
      }

      $manifest = $manifestIndex[$normalizedManifestPath]
      $apis = @()
      if ($null -ne $manifest.parsed) {
        $apis = @(Get-ArrayOrEmpty -Object $manifest.parsed -Name "apis")
      }

      $linkedApis = New-List
      foreach ($api in $apis) {
        $entryPoint = [string](Get-PropertyOrDefault -Object $api -Name "entryPoint" -DefaultValue "")
        if ([string]::IsNullOrWhiteSpace($entryPoint)) {
          continue
        }

        $directLink = $nativeText.IndexOf($entryPoint, [System.StringComparison]::Ordinal) -ge 0
        $macroLink = $false
        if (-not $directLink -and $hasPluginMacro -and $entryPoint -match '^jyppx_trt(8|10|11)_') {
          $macroLink = $nativeText -match ("jyppx_trt{0}_" -f $Matches[1])
        }

        if ($directLink -or $macroLink) {
          $linkedApis.Add([pscustomobject][ordered]@{
              id = [string](Get-PropertyOrDefault -Object $api -Name "id" -DefaultValue "")
              entryPoint = $entryPoint
              linkKind = if ($directLink) { "native-text" } else { "prefix-macro" }
            })
        }
      }

      if ($null -ne $manifest.parseError) {
        Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "invalid-manifest-json" -Detail "${normalizedManifestPath}: $($manifest.parseError)"
      }
      if ($manifest.versionLineFromPath -and $null -ne $manifest.parsed) {
        $manifestVersion = [string](Get-PropertyOrDefault -Object $manifest.parsed -Name "versionLine" -DefaultValue "")
        if ($manifestVersion -and $manifestVersion -ne $manifest.versionLineFromPath) {
          Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "manifest-version-path-mismatch" -Detail "${normalizedManifestPath}: path=$($manifest.versionLineFromPath), json=$manifestVersion"
        }
      }
      if ($linkedApis.Count -eq 0 -and $status -eq "implemented-with-pointer-free-wrapper") {
        Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "manifest-native-link-miss" -Detail $normalizedManifestPath
      }

      $manifestRows.Add([pscustomobject][ordered]@{
          path = $normalizedManifestPath
          versionLine = $manifest.versionLineFromPath
          apiCount = $apis.Count
          linkedApiCount = $linkedApis.Count
          linkedApis = @($linkedApis | Sort-Object entryPoint)
        })
    }

    $manifestVersionLines = @($manifestRows | Where-Object { $_.apiCount -gt 0 } | ForEach-Object { $_.versionLine } | Sort-Object -Unique)
    $managedTexts = New-List
    foreach ($relativePath in $managedSources) {
      $resolvedPath = Resolve-RepositoryPath ([string]$relativePath)
      if (Test-Path -LiteralPath $resolvedPath -PathType Leaf) {
        $managedTexts.Add((Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8))
      }
    }
    $managedText = [string]::Join("`n", @($managedTexts))
    $missingPublicSurface = New-List
    foreach ($surface in $publicSurface) {
      $terminalName = ([string]$surface -split '\.')[-1] -replace '\(\)$', ''
      if ([string]::IsNullOrWhiteSpace($terminalName) -or $managedText -notmatch ("\b{0}\b" -f [regex]::Escape($terminalName))) {
        $missingPublicSurface.Add([string]$surface)
        Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "missing-managed-public-surface" -Detail ([string]$surface)
      }
    }

    $forbiddenPublicSurface = @()
    foreach ($relativePath in $managedSources | Where-Object { ([string]$_) -notmatch '/Internal/' }) {
      $resolvedPath = Resolve-RepositoryPath ([string]$relativePath)
      if (Test-Path -LiteralPath $resolvedPath -PathType Leaf) {
        $fileText = Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8
        $forbiddenPublicSurface += @([regex]::Matches($fileText, '(?im)^\s*public\b[^\r\n;{}]*(?:\bIntPtr\b|\bnint\b|\bUIntPtr\b|\bSafeHandle\b)[^\r\n;{}]*') | ForEach-Object { "${relativePath}: $($_.Value.Trim())" })
      }
    }
    foreach ($finding in @($forbiddenPublicSurface | Sort-Object -Unique)) {
      Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "forbidden-public-handle" -Detail ([string]$finding)
    }

    if ($status -like "implemented*") {
      if ($managedSources.Count -eq 0) { Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "managed-evidence-required" -Detail "implemented candidate has no managed sources" }
      if ($smokeSources.Count -eq 0) { Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "smoke-evidence-required" -Detail "implemented candidate has no smoke sources" }
      if ($qualityTests.Count -eq 0) { Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "quality-evidence-required" -Detail "implemented candidate has no quality tests" }
      if ($nativeRequired -and $nativeSources.Count -eq 0) { Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "native-evidence-required" -Detail "nativeLayerRequired candidate has no native sources" }
      if ($nativeRequired -and $manifestSources.Count -eq 0) { Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "manifest-evidence-required" -Detail "nativeLayerRequired candidate has no manifest sources" }
    }

    $ownershipBoundary = [string](Get-PropertyOrDefault -Object $implementationEvidence -Name "ownershipBoundary" -DefaultValue "")
    if ($status -like "implemented*" -and $ownershipBoundary -notmatch '(?i)not exposed|not expose|not returned|never returned|never exposed|remain deferred|not retained') {
      Add-Finding -Findings $candidateFindings -CandidateId $candidateId -Kind "ownership-boundary-marker-missing" -Detail "ownershipBoundary must document copied output and deferred raw ownership"
    }

    $allFindings.AddRange(@($candidateFindings))
    $candidateRows.Add([pscustomobject][ordered]@{
        candidateId = $candidateId
        group = $groupProperty.Name
        apiArea = [string](Get-PropertyOrDefault -Object $candidate -Name "apiArea" -DefaultValue "")
        riskLevel = [string](Get-PropertyOrDefault -Object $candidate -Name "riskLevel" -DefaultValue "")
        implementationStatus = $status
        nativeLayerRequired = $nativeRequired
        evidencePathCount = $allEvidencePaths.Count
        missingPathCount = $missingPaths.Count
        nativeEntryPointCount = $nativeEntryPoints.Count
        nativeEntryPoints = @($nativeEntryPoints)
        manifestCount = $manifestRows.Count
        manifestVersionLines = @($manifestVersionLines)
        manifestRows = @($manifestRows | Sort-Object path)
        publicSurfaceCount = $publicSurface.Count
        missingPublicSurfaceCount = $missingPublicSurface.Count
        missingPublicSurface = @($missingPublicSurface)
        forbiddenPublicHandleCount = @($forbiddenPublicSurface).Count
        smokeSourceCount = $smokeSources.Count
        qualityTestCount = $qualityTests.Count
        ownershipBoundaryMarker = $ownershipBoundary -match '(?i)not exposed|not expose|not returned|never returned|never exposed|remain deferred|not retained'
        candidateFindingCount = $candidateFindings.Count
        candidateFindings = @($candidateFindings)
        isRuntimeExecutionProof = $false
        isPackageConsumerRuntimeProof = $false
        canPromoteRuntimeProof = $false
        canPromoteReleaseProof = $false
        canDeleteDeferredRecord = $false
      })
  }
}

$implementationRows = @($candidateRows | Where-Object { $_.implementationStatus -like "implemented*" })
$manifestRows = @($candidateRows | ForEach-Object { @($_.manifestRows) })
$record = [ordered]@{
  schemaVersion = "deferred-readonly-candidate-evidence-audit.v1"
  auditKind = "deferred-readonly-candidate-evidence-audit"
  sourceCandidateList = (Convert-ToRepositoryPath $CandidatePath)
  sourceEvidenceMap = (Convert-ToRepositoryPath $EvidenceMapPath)
  candidateCount = $candidateRows.Count
  implementationCandidateCount = $implementationRows.Count
  evidencePathCheckCount = [int](($candidateRows | Measure-Object evidencePathCount -Sum).Sum)
  missingEvidencePathCount = [int](($candidateRows | Measure-Object missingPathCount -Sum).Sum)
  manifestCheckCount = $manifestRows.Count
  manifestFindingCount = [int](@($allFindings | Where-Object { $_.kind -like "manifest-*" }).Count)
  publicSurfaceCheckCount = [int](($candidateRows | Measure-Object publicSurfaceCount -Sum).Sum)
  missingPublicSurfaceCount = [int](($candidateRows | Measure-Object missingPublicSurfaceCount -Sum).Sum)
  forbiddenPublicHandleCount = [int](($candidateRows | Measure-Object forbiddenPublicHandleCount -Sum).Sum)
  findingCount = $allFindings.Count
  allEvidencePathsExist = $allFindings.Where({ $_.kind -eq "missing-evidence-path" }).Count -eq 0
  allManifestRecordsValid = $allFindings.Where({ $_.kind -like "manifest-*" }).Count -eq 0
  allManagedPublicSurfacesPresent = $allFindings.Where({ $_.kind -eq "missing-managed-public-surface" }).Count -eq 0
  noForbiddenPublicHandles = $allFindings.Where({ $_.kind -eq "forbidden-public-handle" }).Count -eq 0
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  canPromoteRuntimeProof = $false
  canPromoteReleaseProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
  candidates = @($candidateRows | Sort-Object group,candidateId)
  findings = @($allFindings | Sort-Object candidateId,kind,detail)
  boundary = "This audit verifies repository evidence linkage only. It does not prove vendor runtime execution, package-consumer execution, post-publish verification, owner authorization, or permission to delete deferred history."
  forbiddenSubstitutes = @("build-only", "ProjectReference", "local feed", "direct .nupkg", "dependency probe", "synthetic runtime", "Skipped=True", "sidecar")
  nextActions = @(
    "Keep deferred manifests and coverage history unchanged.",
    "Use a compatible host and clean package consumer for real runtime proof.",
    "Re-run this audit after any native, manifest, wrapper, smoke, or quality evidence change."
  )
}

$artifactRoot = Resolve-RepositoryPath $OutputRoot
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "deferred-readonly-candidate-evidence-audit.json"
$markdownPath = Join-Path $artifactRoot "deferred-readonly-candidate-evidence-audit.md"
Write-TextFile -Path $jsonPath -Value ($record | ConvertTo-Json -Depth 20)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Deferred Readonly Candidate Evidence Audit")
$lines.Add("")
$lines.Add("- schema: ``$($record.schemaVersion)``")
$lines.Add("- candidates: $($record.candidateCount) (implemented/design-gate status: $($record.implementationCandidateCount))")
$lines.Add("- evidence paths checked: $($record.evidencePathCheckCount); missing: $($record.missingEvidencePathCount)")
$lines.Add("- manifest records checked: $($record.manifestCheckCount); manifest findings: $($record.manifestFindingCount)")
$lines.Add("- public surface checks: $($record.publicSurfaceCheckCount); missing: $($record.missingPublicSurfaceCount); forbidden public handles: $($record.forbiddenPublicHandleCount)")
$lines.Add("- total findings: $($record.findingCount)")
$lines.Add("- promotion boundary: ``canPromoteRuntimeProof=false``; ``canPromoteReleaseProof=false``; ``canPublishPublicly=false``; ``canCloseReleaseIssue=false``")
$lines.Add("")
$lines.Add("## Candidate Matrix")
$lines.Add("")
$lines.Add("| Candidate | Status | Evidence paths | Manifests | Version lines | Public surface missing | Findings | Runtime proof |")
$lines.Add("| --- | --- | ---: | ---: | --- | ---: | ---: | --- |")
foreach ($candidate in @($record.candidates)) {
  $versionLines = [string]::Join(",", @($candidate.manifestVersionLines))
  $lines.Add("| ``$($candidate.candidateId)`` | ``$($candidate.implementationStatus)`` | $($candidate.evidencePathCount) | $($candidate.manifestCount) | $versionLines | $($candidate.missingPublicSurfaceCount) | $($candidate.candidateFindingCount) | false |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)
$lines.Add("")
$lines.Add("All promotion and publication flags are fixed to ``false``. A clean audit means only that the repository's declared evidence paths and linkage are internally consistent; it is not a native runtime, clean consumer, post-publish, or release-close proof.")
$lines.Add("")
$lines.Add("## Findings")
$lines.Add("")
if ($record.findingCount -eq 0) {
  $lines.Add("No findings.")
}
else {
  $lines.Add("| Candidate | Kind | Detail |")
  $lines.Add("| --- | --- | --- |")
  foreach ($finding in @($record.findings)) {
    $detail = ([string]$finding.detail).Replace('|', '\|').Replace("`r", ' ').Replace("`n", ' ')
    $lines.Add("| ``$($finding.candidateId)`` | ``$($finding.kind)`` | $detail |")
  }
}
Write-TextFile -Path $markdownPath -Value ([string]::Join("`n", $lines) + "`n")

Write-Host "Deferred readonly candidate evidence audit written to $jsonPath"
Write-Host "Deferred readonly candidate evidence audit written to $markdownPath"
Write-Host "Candidates: $($record.candidateCount); findings: $($record.findingCount)"
