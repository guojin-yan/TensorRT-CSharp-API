using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerRealProofImportMasterPackTests
{
    [Fact]
    public void MasterPackKeepsOwnerProofImportBlockedUntilRealRecordsArrive()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-real-proof-import-master-pack.json");
        JsonElement root = document.RootElement;

        Assert.Equal("owner-real-proof-import-master-pack", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("packState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());

        string boundary = root.GetProperty("proofBoundary").GetString()!;
        foreach (string marker in new[]
        {
            "does not create evidence",
            "run models",
            "publish packages",
            "ProjectReference",
            "direct .nupkg",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, boundary, StringComparison.OrdinalIgnoreCase);
        }

        JsonElement[] records = root.GetProperty("records").EnumerateArray().ToArray();
        Assert.Equal(4, records.Length);

        foreach (JsonElement record in records)
        {
            Assert.False(record.GetProperty("canPromote").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(record.GetProperty("expectedRecordPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(record.GetProperty("templateOrSchemaPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(record.GetProperty("validatorCommand").GetString()));
            Assert.True(record.GetProperty("ownerFieldsStillMissing").GetArrayLength() >= 5);
            Assert.True(record.GetProperty("forbiddenSubstitutes").GetArrayLength() >= 5);
        }
    }

    [Fact]
    public void MasterPackMapsAllRealProofRecordsToExistingTemplatesAndValidators()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-real-proof-import-master-pack.json");
        JsonElement[] records = document.RootElement.GetProperty("records").EnumerateArray().ToArray();

        AssertRecord(
            records,
            "real-case-evidence-record",
            "real-model-runtime",
            "artifacts/final-release/real-case-evidence-record.json",
            "artifacts/final-release/real-case-evidence-record-template.json",
            "eng/Test-RealCaseEvidenceRecord.ps1",
            "-RecordPath",
            "-FailOnNotProof");

        AssertRecord(
            records,
            "package-consumer-runtime-proof-record",
            "package-consumer-runtime",
            "artifacts/final-release/package-consumer-runtime-proof-record.json",
            "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
            "eng/Test-PackageConsumerRuntimeProofRecord.ps1",
            "-InputPath",
            "-RequireExistingLog");

        AssertRecord(
            records,
            "post-publish-verification-record",
            "post-publish-verification",
            "artifacts/final-release/post-publish-verification-record.json",
            "artifacts/final-release/post-publish-verification-record-template.json",
            "eng/Test-PostPublishVerificationRecord.ps1",
            "-InputPath",
            "-FailOnNotProof");

        AssertRecord(
            records,
            "release-issue-close-record",
            "release-issue-close",
            "artifacts/final-release/release-issue-close-record.json",
            "artifacts/final-release/release-issue-close-record-template.json",
            "eng/Test-ReleaseIssueCloseRecord.ps1",
            "-InputPath",
            "-FailOnNotCloseReady");
    }

    [Fact]
    public void MasterPackOwnerFieldsAlignWithRealCasePackageAndFinalDashboards()
    {
        using JsonDocument masterDocument = ReadFinalReleaseJson("owner-real-proof-import-master-pack.json");
        using JsonDocument realCaseTemplateDocument = ReadFinalReleaseJson("real-case-evidence-record-template.json");
        using JsonDocument packageSchemaDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.schema.json");
        using JsonDocument packageTemplateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.template.json");
        using JsonDocument dashboardDocument = ReadFinalReleaseJson("final-prepublish-quality-gate-dashboard.json");
        using JsonDocument closureMapDocument = ReadFinalReleaseJson("release-candidate-package-consumer-closure-map.json");
        using JsonDocument closureIndexDocument = ReadFinalReleaseJson("release-evidence-closure-index.json");

        JsonElement[] records = masterDocument.RootElement.GetProperty("records").EnumerateArray().ToArray();
        JsonElement realCaseRecord = Assert.Single(records, item => item.GetProperty("id").GetString() == "real-case-evidence-record");
        JsonElement packageRecord = Assert.Single(records, item => item.GetProperty("id").GetString() == "package-consumer-runtime-proof-record");

        string[] realCaseFields = realCaseRecord.GetProperty("ownerFieldsStillMissing").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] templateFields = realCaseTemplateDocument.RootElement.GetProperty("requiredOwnerFields").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string field in realCaseFields)
        {
            Assert.Contains(field, templateFields);
        }

        string[] requiredTasks = realCaseRecord.GetProperty("requiredTaskCoverage").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains(task, requiredTasks);
        }

        string[] packageFields = packageRecord.GetProperty("ownerFieldsStillMissing").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] schemaFields = packageSchemaDocument.RootElement.GetProperty("fields").EnumerateArray()
            .Select(static item => item.GetProperty("name").GetString()!)
            .ToArray();
        JsonElement packageTemplate = packageTemplateDocument.RootElement;

        foreach (string field in packageFields)
        {
            Assert.Contains(field, schemaFields);
            Assert.True(packageTemplate.TryGetProperty(field, out _), $"Package consumer owner input template is missing {field}.");
        }

        string[] sourceArtifacts = masterDocument.RootElement.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string required in new[]
        {
            "artifacts/final-release/final-prepublish-quality-gate-dashboard.json",
            "artifacts/final-release/release-candidate-package-consumer-closure-map.json",
            "artifacts/final-release/release-evidence-closure-index.json"
        })
        {
            Assert.Contains(required, sourceArtifacts);
        }

        string dashboardText = dashboardDocument.RootElement.GetRawText();
        string closureMapText = closureMapDocument.RootElement.GetRawText();
        string closureIndexText = closureIndexDocument.RootElement.GetRawText();
        Assert.Contains("owner-real-proof-import-master-pack.json", dashboardText, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-import-master-pack.json", closureMapText, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-import-master-pack.json", closureIndexText, StringComparison.Ordinal);
    }

    [Fact]
    public void MasterPackDocumentsAuxiliaryYoloVisionChainAsSixTaskNonReleaseProof()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-real-proof-import-master-pack.json");
        JsonElement chain = Assert.Single(document.RootElement.GetProperty("auxiliaryOwnerInputChains").EnumerateArray());

        Assert.Equal("yolovision-real-asset-owner-proof-input", chain.GetProperty("id").GetString());
        Assert.Equal("six-task-yolov8n-owner-input-candidate", chain.GetProperty("currentScope").GetString());
        Assert.Equal("eng/Import-YoloVisionRealAssetOwnerProofInput.ps1", chain.GetProperty("script").GetString());
        Assert.Equal("eng/Test-YoloVisionRealAssetOwnerProofInput.ps1", chain.GetProperty("validator").GetString());
        Assert.True(chain.GetProperty("mustNotPromoteRelease").GetBoolean());
        Assert.Contains("covers det/seg/pose/obb/cls/sem", chain.GetProperty("knownGap").GetString()!, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release proof", chain.GetProperty("knownGap").GetString()!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void MasterPackIsDocumentedAndCrossLinked()
    {
        string artifactMarkdown = ReadFinalReleaseText("owner-real-proof-import-master-pack.md");
        string article = ReadText("docs", "articles", "zh-cn", "owner-real-proof-import-master-pack.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Owner Real Proof Import Master Pack",
            "real-case-evidence-record.json",
            "package-consumer-runtime-proof-record.json",
            "post-publish-verification-record.json",
            "release-issue-close-record.json",
            "covers `det/seg/pose/obb/cls/sem`",
            "not release proof",
            "ProjectReference",
            "direct `.nupkg`"
        })
        {
            Assert.Contains(marker, artifactMarkdown, StringComparison.Ordinal);
        }

        foreach (string marker in new[]
        {
            "真实 Owner Proof 导入总控包",
            "real-case-evidence-record.json",
            "package-consumer-runtime-proof-record.json",
            "post-publish-verification-record.json",
            "release-issue-close-record.json",
            "覆盖 `det/seg/pose/obb/cls/sem`",
            "不能替代完整 release proof",
            "不允许 local feed",
            "不允许 direct `.nupkg`",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/owner-real-proof-import-master-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/owner-real-proof-import-master-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-real-proof-import-master-pack.md", docsToc, StringComparison.Ordinal);
    }

    private static void AssertRecord(
        JsonElement[] records,
        string id,
        string lane,
        string expectedRecordPath,
        string templateOrSchemaPath,
        string validator,
        string validatorParameter,
        string validatorFlag)
    {
        JsonElement record = Assert.Single(records, item => item.GetProperty("id").GetString() == id);
        Assert.Equal(lane, record.GetProperty("lane").GetString());
        Assert.Equal(expectedRecordPath, record.GetProperty("expectedRecordPath").GetString());
        Assert.Equal(templateOrSchemaPath, record.GetProperty("templateOrSchemaPath").GetString());
        Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, templateOrSchemaPath.Replace('/', Path.DirectorySeparatorChar))));

        string validatorCommand = record.GetProperty("validatorCommand").GetString()!;
        string normalizedValidatorCommand = validatorCommand.Replace('\\', '/');
        Assert.Contains(validator, normalizedValidatorCommand, StringComparison.Ordinal);
        Assert.Contains(validatorParameter, validatorCommand, StringComparison.Ordinal);
        Assert.Contains(validatorFlag, validatorCommand, StringComparison.Ordinal);
        Assert.False(record.GetProperty("canPromote").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(ReadFinalReleaseText(fileName));
    }

    private static string ReadFinalReleaseText(string fileName)
    {
        return ReadText("artifacts", "final-release", fileName);
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}
