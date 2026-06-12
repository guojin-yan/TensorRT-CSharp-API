using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a TensorRT 11 loop object to the network.
    /// 向 network 添加 TensorRT 11 loop 对象。
    /// </summary>
    /// <returns>A network-owned loop wrapper. / 由 network 持有生命周期的 loop 包装对象。</returns>
    /// <remarks>
    /// This API creates the loop container. Add trip-limit, iterator, recurrence, and output boundary layers through the returned object.
    /// 此 API 创建 loop 容器；请通过返回对象继续添加 trip-limit、iterator、recurrence 和 output 边界层。
    /// </remarks>
    public TensorRtLoop AddLoop()
    {
        return new TensorRtLoop(Line, NativeBridgeApi.AddLoop(Line, _handle));
    }

    /// <summary>
    /// Adds a TensorRT 11 if-conditional object to the network.
    /// 向 network 添加 TensorRT 11 if-conditional 对象。
    /// </summary>
    /// <returns>A network-owned conditional wrapper. / 由 network 持有生命周期的 conditional 包装对象。</returns>
    /// <remarks>
    /// This API creates the conditional container. Configure condition, branch inputs, and branch outputs through the returned object.
    /// 此 API 创建 conditional 容器；请通过返回对象继续配置 condition、分支输入和分支输出。
    /// </remarks>
    public TensorRtIfConditional AddIfConditional()
    {
        return new TensorRtIfConditional(Line, NativeBridgeApi.AddIfConditional(Line, _handle));
    }
}
