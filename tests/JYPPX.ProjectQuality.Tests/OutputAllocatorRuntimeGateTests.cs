using System.Globalization;
using System.Reflection;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OutputAllocatorRuntimeGateTests
{
    [Fact]
    public void InternalRuntimeGateCopiesShapeAndNeverExposesOutputBufferPointer()
    {
        object gate = CreateGate();
        try
        {
            object notifyRequest = CreateRequest("quality_output", 0UL, 1UL, new long[] { 1, 3, 224, 224 }, "quality-notify", false);
            object notify = InvokeGate(gate, "RunInternalNotifyShapeRuntimeGate", notifyRequest);
            Assert.Equal("output-allocator-internal-runtime-gate", ReadProperty(notify, "EvidenceKind"));
            Assert.Equal("False", ReadProperty(notify, "RealCallbackRuntime"));
            Assert.Equal("output-allocator-prototype", ReadProperty(notify, "CallbackKind"));
            Assert.Equal("notify-shape", ReadProperty(notify, "Operation"));
            Assert.Equal("quality_output", ReadProperty(notify, "TensorName"));
            Assert.Equal("4", ReadProperty(notify, "ShapeRank"));
            Assert.Equal("[1x3x224x224]", ReadProperty(notify, "ShapeSummary"));
            Assert.Equal("1", ReadProperty(notify, "NotifyShapeCount"));
            Assert.Equal("0", ReadProperty(notify, "ReallocateOutputCount"));
            Assert.Equal("False", ReadProperty(notify, "OutputBufferPointerExposed"));
            Assert.Equal("False", ReadProperty(notify, "OutputBufferPointerProduced"));
            Assert.Equal("Ok", ReadProperty(notify, "LastStatus"));

            object reallocateRequest = CreateRequest("quality_output", 4096UL, 256UL, new long[] { 1, 1000 }, "quality-reallocate", true);
            object reallocate = InvokeGate(gate, "RunInternalReallocateOutputRuntimeGate", reallocateRequest);
            Assert.Equal("reallocate-output", ReadProperty(reallocate, "Operation"));
            Assert.Equal("4096", ReadProperty(reallocate, "RequestedSize"));
            Assert.Equal("256", ReadProperty(reallocate, "Alignment"));
            Assert.Equal("2", ReadProperty(reallocate, "InvocationCount"));
            Assert.Equal("1", ReadProperty(reallocate, "NotifyShapeCount"));
            Assert.Equal("1", ReadProperty(reallocate, "ReallocateOutputCount"));
            Assert.Equal("0", ReadProperty(reallocate, "FailureCount"));
            Assert.Equal("0", ReadProperty(reallocate, "InFlightCallbackCount"));
            Assert.Equal("True", ReadProperty(reallocate, "CallbackStatePinned"));
            Assert.Equal("True", ReadProperty(reallocate, "DelegatePinned"));
            Assert.Equal("False", ReadProperty(reallocate, "IsAttached"));
            Assert.Contains("output-buffer-pointer-exposed=false", ReadProperty(reallocate, "LastDiagnostic"));

            ((IDisposable)gate).Dispose();
            object postDispose = GetSnapshot(gate, "post-dispose");
            Assert.Equal("True", ReadProperty(postDispose, "DisposeRequested"));
            Assert.Equal("False", ReadProperty(postDispose, "CallbackStatePinned"));
            Assert.Equal("False", ReadProperty(postDispose, "DelegatePinned"));
            Assert.Equal("1", ReadProperty(postDispose, "ReleaseHookCount"));
            Assert.Contains("release hook", ReadProperty(postDispose, "ReleaseDiagnostic"));
        }
        finally
        {
            ((IDisposable)gate).Dispose();
        }
    }

    [Fact]
    public void InternalRuntimeGateMapsSyntheticExceptionToStatus()
    {
        object gate = CreateGate();
        try
        {
            object request = CreateRequest("quality_output", 8192UL, 512UL, new long[] { 1, 64 }, "throw", false);
            object failure = InvokeGate(gate, "RunInternalReallocateOutputRuntimeGate", request);
            Assert.Equal("output-allocator-internal-runtime-gate", ReadProperty(failure, "EvidenceKind"));
            Assert.Equal("False", ReadProperty(failure, "RealCallbackRuntime"));
            Assert.Equal("InvalidState", ReadProperty(failure, "LastStatus"));
            Assert.Equal("1", ReadProperty(failure, "InvocationCount"));
            Assert.Equal("1", ReadProperty(failure, "ReallocateOutputCount"));
            Assert.Equal("1", ReadProperty(failure, "FailureCount"));
            Assert.Equal("0", ReadProperty(failure, "InFlightCallbackCount"));
            Assert.Equal("False", ReadProperty(failure, "Succeeded"));
            Assert.Contains("InvalidOperationException", ReadProperty(failure, "LastDiagnostic"));
        }
        finally
        {
            ((IDisposable)gate).Dispose();
        }
    }

    private static object CreateGate()
    {
        Type gateType = typeof(TensorRtAllocatorCallbackOwner).Assembly.GetType("JYPPX.TensorRtSharp.TensorRtOutputAllocatorRuntimeGate")!;
        Assert.NotNull(gateType);
        return Activator.CreateInstance(gateType, nonPublic: true)!;
    }

    private static object CreateRequest(
        string tensorName,
        ulong requestedSize,
        ulong alignment,
        long[] shapeDimensions,
        string reason,
        bool hasCurrentMemory)
    {
        Type requestType = typeof(TensorRtAllocatorCallbackOwner).Assembly.GetType("JYPPX.TensorRtSharp.TensorRtOutputAllocatorRuntimeGateRequest")!;
        Assert.NotNull(requestType);
        return Activator.CreateInstance(
            requestType,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            args: new object[] { tensorName, requestedSize, alignment, shapeDimensions, reason, hasCurrentMemory },
            culture: CultureInfo.InvariantCulture)!;
    }

    private static object InvokeGate(object gate, string methodName, object request)
    {
        MethodInfo? method = gate.GetType().GetMethod(methodName, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        return method.Invoke(gate, new[] { request })!;
    }

    private static object GetSnapshot(object gate, string operation)
    {
        MethodInfo? method = gate.GetType().GetMethod("GetInternalRuntimeGateSnapshot", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        return method.Invoke(gate, new object[] { operation })!;
    }

    private static string ReadProperty(object result, string propertyName)
    {
        PropertyInfo? property = result.GetType().GetProperty(
            propertyName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        Assert.NotNull(property);
        return Convert.ToString(property.GetValue(result), CultureInfo.InvariantCulture) ?? string.Empty;
    }
}
