#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerNativeAttachBridgeShapeGate final
{
    DebugListenerNativeAttachBridgeShapeGate() noexcept = default;
    DebugListenerNativeAttachBridgeShapeGate(const DebugListenerNativeAttachBridgeShapeGate&) = delete;
    DebugListenerNativeAttachBridgeShapeGate& operator=(const DebugListenerNativeAttachBridgeShapeGate&) = delete;
    DebugListenerNativeAttachBridgeShapeGate(DebugListenerNativeAttachBridgeShapeGate&&) = delete;
    DebugListenerNativeAttachBridgeShapeGate& operator=(DebugListenerNativeAttachBridgeShapeGate&&) = delete;
    ~DebugListenerNativeAttachBridgeShapeGate() noexcept = default;

    bool configure_shape(int32_t api_line, bool lifecycle_gate_ready) noexcept
    {
        line = api_line;
        version_guard_ready = api_line == 10 || api_line == 11;
        attach_bridge_shape_ready = lifecycle_gate_ready && version_guard_ready;
        no_throw_boundary_ready = attach_bridge_shape_ready;
        ownership_diagnostics_ready = attach_bridge_shape_ready;
        pointer_free = true;
        non_null_attach_enabled = false;
        return attach_bridge_shape_ready;
    }

    int32_t line{};
    bool attach_bridge_shape_ready{};
    bool no_throw_boundary_ready{};
    bool version_guard_ready{};
    bool ownership_diagnostics_ready{};
    bool pointer_free{true};
    bool non_null_attach_enabled{};
};

static_assert(
    !std::is_copy_constructible<DebugListenerNativeAttachBridgeShapeGate>::value,
    "debug-listener-native-attach-bridge-shape-gate copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerNativeAttachBridgeShapeGate>::value,
    "debug-listener-native-attach-bridge-shape-gate copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerNativeAttachBridgeShapeGate>::value,
    "debug-listener-native-attach-bridge-shape-gate move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerNativeAttachBridgeShapeGate>::value,
    "debug-listener-native-attach-bridge-shape-gate move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerNativeAttachBridgeShapeGate>::value,
    "debug-listener-native-attach-bridge-shape-gate destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerNativeAttachBridgeShapeGate&>().configure_shape(11, true)),
    "debug-listener-native-attach-bridge-shape-gate configure_shape must stay noexcept.");
} // namespace
