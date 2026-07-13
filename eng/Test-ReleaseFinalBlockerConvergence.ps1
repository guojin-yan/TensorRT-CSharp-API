[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$SummaryPath = "artifacts\final-release\release-proof-owner-backfill-summary-validation.json",
  [string]$OutputPath = "artifacts\final-release\release-final-blocker-convergence.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepoPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function ConvertTo-OwnerAction {
  param(
    [string]$LaneId,
    [object[]]$MissingInputs
  )

  $baseAction = switch ($LaneId) {
    "real-model-runtime" {
      "Owner must run real detection/classification/segmentation/OBB/pose/semantic runtime cases on a compatible CUDA/TensorRT host, preserve stdout/stderr logs, compute hashes, and run the strict real-case validator."
    }
    "package-consumer-runtime" {
      "Owner must consume the public package from an external clean project, preserve restore/build/runtime smoke logs, compute package and smoke-log hashes, and run the strict package-consumer validator."
    }
    "post-publish-verification" {
      "Owner must verify the package after public publish from a clean consumer, preserve restore/runtime smoke logs, compute hashes, and run the strict post-publish validator."
    }
    "release-issue-close" {
      "Owner must close only after real-model-runtime, package-consumer-runtime, and post-publish-verification strict validators have all passed, then provide the release close record and validator output."
    }
    default {
      "Owner must provide the missing real evidence inputs and rerun the strict validator."
    }
  }

  [pscustomobject]@{
    required = [bool]($MissingInputs.Count -gt 0)
    summary = $baseAction
    missingInputCount = [int]$MissingInputs.Count
  }
}

function ConvertTo-PromotionCriteria {
  param([string]$LaneId)

  $criteria = @(
    "expected record exists",
    "strict validator output exists",
    "all required logs exist",
    "required hash fields are populated from real files",
    "required host metadata is populated from the execution host",
    "strict validator command exits successfully"
  )

  if ($LaneId -eq "release-issue-close") {
    $criteria += "all upstream proof lanes are already strict-validator passed"
    $criteria += "release issue close record is produced from real owner decision input"
  }

  return $criteria
}

$summaryFullPath = Resolve-RepoPath -Path $SummaryPath
if (-not (Test-Path -LiteralPath $summaryFullPath -PathType Leaf)) {
  throw "Summary file not found: $SummaryPath"
}

$summary = Get-Content -LiteralPath $summaryFullPath -Raw | ConvertFrom-Json
$lanes = @($summary.lanes)

$convergedLanes = foreach ($lane in $lanes) {
  $missingInputs = @($lane.missingInputs)
  $canPromote = [bool](
    $lane.recordExists -and
    $lane.validatorOutputExists -and
    $missingInputs.Count -eq 0 -and
    $lane.canPromote
  )

  [pscustomobject]@{
    id = [string]$lane.id
    currentValidationState = [string]$lane.currentValidationState
    recordExists = [bool]$lane.recordExists
    validatorOutputExists = [bool]$lane.validatorOutputExists
    missingInputs = $missingInputs
    missingInputCount = [int]$missingInputs.Count
    strictValidatorCommand = [string]$lane.strictValidatorCommand
    ownerActionRequired = ConvertTo-OwnerAction -LaneId ([string]$lane.id) -MissingInputs $missingInputs
    promotionCriteria = ConvertTo-PromotionCriteria -LaneId ([string]$lane.id)
    canPromote = $canPromote
  }
}

$remainingBlockerCount = @($convergedLanes | Where-Object { -not $_.canPromote }).Count
$missingInputCount = ($convergedLanes | ForEach-Object { $_.missingInputCount } | Measure-Object -Sum).Sum
if ($null -eq $missingInputCount) {
  $missingInputCount = 0
}

$result = [pscustomobject]@{
  recordKind = "release-final-blocker-convergence"
  generatedAtLocal = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
  sourceSummaryPath = "artifacts/final-release/release-proof-owner-backfill-summary-validation.json"
  sourceSummaryRecordKind = [string]$summary.recordKind
  convergenceState = "blocked-owner-action-required"
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
  approvesPublicRelease = $false
  proofBoundary = "This final convergence runner is read-only. It summarizes strict validator blockers and owner actions; it does not create proof, run dotnet nuget publish, promote lanes, or close release issues."
  forbiddenProofSubstitutes = @(
    "template",
    "report",
    "matrix",
    "article",
    "command pack",
    "summary pack",
    "dry-run",
    "build-only",
    "sidecar-only",
    "screenshot-only",
    "Skipped=True",
    "blocked-by-cuda-driver",
    "local feed",
    "ProjectReference",
    "direct .nupkg"
  )
  remainingBlockerCount = [int]$remainingBlockerCount
  missingInputCount = [int]$missingInputCount
  lanes = @($convergedLanes)
  finalOwnerAction = "Provide real owner evidence for all blocked lanes, archive the strict validator outputs, rerun this read-only convergence runner, and only then reassess public release readiness."
}

$outputFullPath = Resolve-RepoPath -Path $OutputPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputFullPath) | Out-Null
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $outputFullPath -Encoding utf8
$result | ConvertTo-Json -Depth 12
