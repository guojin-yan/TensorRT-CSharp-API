using System;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RnnV2BorrowedStateDesignGateTests
{
    [Fact]
    public void KnownSurfacePromotesOwnerBoundTensorsAndCopiedWeightsWithoutRuntimeProof()
    {
        TensorRtRnnV2BorrowedStateDesignGateResult gate =
            TensorRtRnnV2BorrowedStateDesignGate.EvaluateKnownSurface();

        Assert.Equal("rnnv2-borrowed-state-design-gate", gate.EvidenceKind);
        Assert.Equal("rnnv2-borrowed-state", gate.DiagnosticsKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.True(gate.LineSupportsRnnV2);
        Assert.True(gate.DataLengthScalarPromoted);
        Assert.True(gate.NetworkOwnedTensorReferencePolicyReady);
        Assert.True(gate.GateWeightSnapshotCopyReady);
        Assert.True(gate.OwnerLifetimeKnown);
        Assert.False(gate.BorrowedTensorPointerExposed);
        Assert.False(gate.BorrowedWeightsPointerExposed);
        Assert.False(gate.BorrowedStateEscapesCall);
        Assert.True(gate.PointerFreeSurfaceReady);
        Assert.True(gate.BorrowedSnapshotPromotionReady);
        Assert.True(gate.DesignGateReady);
        Assert.False(gate.IsRuntimeExecutionEvidence);
        Assert.False(gate.IsRuntimeExecutionProof);
        Assert.False(gate.CanPromoteRuntimeProof);
        Assert.False(gate.DeferredBorrowedRowsStillRequired);
        Assert.Equal(12, gate.SelectedTriageRowCount);
        Assert.Equal(12, gate.PromotedScalarTriageRowCount);
        Assert.Equal(0, gate.RemainingDeferredTriageRowCount);
        Assert.Equal(6, gate.SelectedCandidateMethods.Count);
        Assert.Empty(gate.DeferredBorrowedMethods);
        Assert.Contains("IRNNv2Layer::getDataLength", gate.SelectedCandidateMethods);
        Assert.DoesNotContain("IRNNv2Layer::getDataLength", gate.DeferredBorrowedMethods);
        Assert.DoesNotContain("IRNNv2Layer::getWeightsForGate", gate.DeferredBorrowedMethods);
        Assert.Equal("safe-wrapper-surface-ready-runtime-proof-pending", gate.Status);
        Assert.Contains("RemainingDeferredTriageRowCount=0", gate.Diagnostic, StringComparison.Ordinal);
        Assert.Contains("DeferredBorrowedRowsStillRequired=False", gate.Diagnostic, StringComparison.Ordinal);
    }

    [Fact]
    public void PublicGateSurfaceDoesNotExposeRawPointerTypes()
    {
        Type[] types =
        {
            typeof(TensorRtRnnV2BorrowedStateDesignGate),
            typeof(TensorRtRnnV2BorrowedStateDesignGateResult),
            typeof(TensorRtRnnV2GateWeightsSnapshot),
            typeof(TensorRtLayer),
            typeof(TensorRtTensor)
        };

        foreach (Type type in types)
        {
            foreach (PropertyInfo property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static))
            {
                Assert.NotEqual(typeof(IntPtr), property.PropertyType);
                Assert.NotEqual(typeof(UIntPtr), property.PropertyType);
            }

            foreach (MethodInfo method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly))
            {
                Assert.NotEqual(typeof(IntPtr), method.ReturnType);
                Assert.NotEqual(typeof(UIntPtr), method.ReturnType);
                Assert.DoesNotContain(method.GetParameters(), static parameter =>
                    parameter.ParameterType == typeof(IntPtr) ||
                    parameter.ParameterType == typeof(UIntPtr));
            }
        }
    }

    [Fact]
    public void CopiedGateWeightSnapshotOwnsIndependentByteStorage()
    {
        ConstructorInfo constructor = typeof(TensorRtRnnV2GateWeightsSnapshot)
            .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic)
            .Single();
        byte[] source = { 1, 2, 3, 4 };
        TensorRtRnnV2GateWeightsSnapshot snapshot =
            (TensorRtRnnV2GateWeightsSnapshot)constructor.Invoke(new object[]
            {
                0,
                TensorRtRnnGateType.Input,
                true,
                false,
                TensorRtDataType.Float,
                1L,
                source
            });

        source[0] = 99;
        byte[] first = snapshot.ToArray();
        first[1] = 88;
        byte[] second = snapshot.ToArray();

        Assert.Equal(new byte[] { 1, 2, 3, 4 }, second);
        Assert.Equal(4, snapshot.ByteCount);
        Assert.Equal(1L, snapshot.ElementCount);
        Assert.False(snapshot.IsBias);
    }

    [Fact]
    public void OwnerLeaseAndBorrowedTensorWrappersKeepNetworkLifetimeBound()
    {
        string lease = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Handles", "SafeTensorRtObjectHandleLease.cs");
        string network = ReadSource("src", "JYPPX.TensorRtSharp", "Network", "TensorRtNetworkDefinition.cs");
        string layer = ReadSource("src", "JYPPX.TensorRtSharp", "Layers", "TensorRtLayer.cs");
        string tensor = ReadSource("src", "JYPPX.TensorRtSharp", "Network", "TensorRtTensor.cs");
        string rnn = ReadSource("src", "JYPPX.TensorRtSharp", "Layers", "TensorRtLayer.Trt8RnnV2Diagnostics.cs");

        Assert.Contains("owner.DangerousAddRef(ref addedRef)", lease, StringComparison.Ordinal);
        Assert.Contains("_owner.DangerousRelease()", lease, StringComparison.Ordinal);
        Assert.Contains("SafeTensorRtObjectHandleLease.Create(_handle)", network, StringComparison.Ordinal);
        Assert.Contains("CloneRequiredOwnerLease", layer, StringComparison.Ordinal);
        Assert.Contains("public bool IsOwnerLifetimeBound => _ownerLease != null", tensor, StringComparison.Ordinal);
        Assert.Contains("new TensorRtTensor(Line, tensor, ownerLease)", rnn, StringComparison.Ordinal);
        Assert.Contains("This layer is not bound to a network owner", layer, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageClosesAllTwelveRnnV2RowsAndKeepsDeferredHistory()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        foreach (string method in new[]
        {
            "getDataLength",
            "getBiasForGate",
            "getCellState",
            "getHiddenState",
            "getSequenceLengths",
            "getWeightsForGate"
        })
        {
            Assert.Contains(
                $"\"IRNNv2Layer\",\"{method}\",\"IRNNv2Layer::{method}\",\"network-layer\",\"implemented-with-deferred-history\"",
                comparison,
                StringComparison.Ordinal);
        }

        using JsonDocument triage = JsonDocument.Parse(ReadSource(
            "artifacts",
            "interface-coverage",
            "deferred-candidate-safety-triage.json"));

        string[] deferredMethods =
        {
            "getBiasForGate",
            "getCellState",
            "getHiddenState",
            "getSequenceLengths",
            "getWeightsForGate"
        };
        JsonElement[] remainingRows = triage.RootElement
            .GetProperty("rows")
            .EnumerateArray()
            .Where(row =>
                row.GetProperty("safetyTier").GetString() == "C - design-gate-required" &&
                row.GetProperty("class").GetString() == "IRNNv2Layer" &&
                deferredMethods.Contains(row.GetProperty("method").GetString(), StringComparer.Ordinal))
            .ToArray();

        Assert.Empty(remainingRows);
        Assert.DoesNotContain(
            triage.RootElement.GetProperty("rows").EnumerateArray(),
            static row =>
                row.GetProperty("safetyTier").GetString() == "C - design-gate-required" &&
                row.GetProperty("class").GetString() == "IRNNv2Layer" &&
                row.GetProperty("method").GetString() == "getDataLength");
    }

    [Fact]
    public void DocumentationRecordsThePromotionAndNonProofBoundary()
    {
        string article = ReadSource("docs", "articles", "zh-cn", "rnnv2-borrowed-state-design-gate.md");
        string groups = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");

        Assert.Contains("12 条 C-tier triage 行", article, StringComparison.Ordinal);
        Assert.Contains("2 条 `getDataLength`", article, StringComparison.Ordinal);
        Assert.Contains("10 条 borrowed tensor/weights", article, StringComparison.Ordinal);
        Assert.Contains("全部 12 条", article, StringComparison.Ordinal);
        Assert.Contains("GetRnnV2DataLength", article, StringComparison.Ordinal);
        Assert.Contains("GetRnnV2CellState", article, StringComparison.Ordinal);
        Assert.Contains("GetRnnV2WeightsForGate", article, StringComparison.Ordinal);
        Assert.Contains("RemainingDeferredTriageRowCount=0", article, StringComparison.Ordinal);
        Assert.Contains("design gate，not runtime proof", article, StringComparison.Ordinal);
        Assert.Contains("rnnv2-borrowed-state-design-gate.md", groups, StringComparison.Ordinal);
        Assert.Contains("10 条 borrowed tensor/weights 行已通过 owner-bound/copy-out", groups, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}
