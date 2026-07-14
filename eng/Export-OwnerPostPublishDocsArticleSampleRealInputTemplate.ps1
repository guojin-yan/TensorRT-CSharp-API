[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

$record = New-OwnerPostPublishTemplateRecord
$jsonPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-real-input.template.json"
$mdPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-real-input.template.md"

Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 18)

$rows = foreach ($lane in @($record.lanes)) {
  "| ``$($lane.id)`` | ``$($lane.requiredFieldCount)`` | ``$($lane.laneState)`` |"
}

Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Post-Publish Docs Article Sample Real Input Template",
  "",
  "- templateState: ``$($record.templateState)``",
  "- laneCount: ``$($record.laneCount)``",
  "- requiredFieldCount: ``$($record.requiredFieldCount)``",
  "- placeholderFieldCount: ``$($record.placeholderFieldCount)``",
  "- forbiddenSubstituteCount: ``$($record.forbiddenSubstituteCount)``",
  "- performsPublish: ``False``",
  "- usesPublishToken: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Lane | Required Fields | State |",
  "| --- | ---: | --- |",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "OwnerPostPublishDocsArticleSampleRealInputTemplateState=$($record.templateState) Lanes=$($record.laneCount) Fields=$($record.requiredFieldCount)"
