$ErrorActionPreference = "Stop"

$script:OwnerRealEvidenceRequiredLaneIds = @(
  "real-model-runtime-owner-proof-required",
  "package-consumer-runtime-owner-proof-required",
  "post-publish-verification-owner-proof-required",
  "final-owner-real-input-template-pack-owner-input-required",
  "owner-external-proof-result-import-owner-proof-required",
  "owner-result-candidate-bridge-real-proof-required"
)

$script:OwnerRealEvidenceForbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "dashboard",
  "template",
  "draft",
  "candidate",
  "build report",
  "article",
  "screenshot",
  "OnnxToEngine report",
  "TensorRtExec report",
  "YoloVision matrix"
)

function Resolve-OwnerRepositoryPath {
  param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$Path
  )

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-OwnerJsonOrNull {
  param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$Path
  )

  $resolved = Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-OwnerFileSha256 {
  param([Parameter(Mandatory = $true)][string]$Path)

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return ""
  }

  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function ConvertTo-OwnerMarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-OwnerValidationItem {
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

function Test-OwnerTextHasForbiddenSubstitute {
  param([AllowNull()][string]$Text)

  if ([string]::IsNullOrWhiteSpace($Text)) {
    return $false
  }

  foreach ($forbidden in $script:OwnerRealEvidenceForbiddenSubstitutes) {
    if ($Text.IndexOf($forbidden, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $true
    }
  }

  foreach ($marker in @("placeholder", "TODO", "sample-only", "sample only", "substitute proof", "not proof")) {
    if ($Text.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $true
    }
  }

  return $false
}

function Test-OwnerStringFilled {
  param([AllowNull()][string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $false
  }

  if ($Value.IndexOf("FILL_", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
    return $false
  }

  return $true
}
