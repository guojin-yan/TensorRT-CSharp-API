#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace
{
struct DebugListenerBorrowedDebugTensorMetadataRuntimeGate final
{
    static constexpr int32_t max_rank = 8;
    static constexpr int32_t max_tensor_name = 128;

    DebugListenerBorrowedDebugTensorMetadataRuntimeGate() noexcept = default;
    DebugListenerBorrowedDebugTensorMetadataRuntimeGate(const DebugListenerBorrowedDebugTensorMetadataRuntimeGate&) = delete;
    DebugListenerBorrowedDebugTensorMetadataRuntimeGate& operator=(const DebugListenerBorrowedDebugTensorMetadataRuntimeGate&) = delete;
    DebugListenerBorrowedDebugTensorMetadataRuntimeGate(DebugListenerBorrowedDebugTensorMetadataRuntimeGate&&) = delete;
    DebugListenerBorrowedDebugTensorMetadataRuntimeGate& operator=(DebugListenerBorrowedDebugTensorMetadataRuntimeGate&&) = delete;
    ~DebugListenerBorrowedDebugTensorMetadataRuntimeGate() noexcept = default;

    bool configure_api_line(int32_t api_line) noexcept
    {
        line_supports_debug_listener = api_line == 10 || api_line == 11;
        return line_supports_debug_listener;
    }

    bool copy_metadata(
        const char* tensor_name,
        int32_t data_type,
        int32_t tensor_location,
        int32_t rank,
        const int64_t* dims,
        bool is_input,
        bool is_output,
        bool is_shape_tensor,
        bool is_execution_tensor) noexcept
    {
        copy_tensor_name(tensor_name);
        copied_data_type = data_type;
        copied_tensor_location = tensor_location;
        copied_rank = clamp_rank(rank);
        for (int32_t index = 0; index < copied_rank; ++index)
        {
            copied_dims[index] = dims == nullptr ? 0 : dims[index];
        }

        for (int32_t index = copied_rank; index < max_rank; ++index)
        {
            copied_dims[index] = 0;
        }

        copied_is_input = is_input;
        copied_is_output = is_output;
        copied_is_shape_tensor = is_shape_tensor;
        copied_is_execution_tensor = is_execution_tensor;
        metadata_copied = copied_name_length > 0 && copied_rank >= 0;
        return metadata_copied;
    }

    bool metadata_copy_ready() const noexcept
    {
        return line_supports_debug_listener && metadata_copied;
    }

    bool borrowed_tensor_pointer_escape_blocked() const noexcept
    {
        return true;
    }

    bool borrowed_tensor_data_pointer_escape_blocked() const noexcept
    {
        return true;
    }

    bool borrowed_tensor_lifetime_runtime_ready() const noexcept
    {
        return false;
    }

    bool borrowed_tensor_data_lifetime_runtime_ready() const noexcept
    {
        return false;
    }

    bool process_debug_tensor_runtime_ready() const noexcept
    {
        return false;
    }

    bool line_supports_debug_listener{};
    bool metadata_copied{};
    char copied_tensor_name[max_tensor_name]{};
    int32_t copied_name_length{};
    int32_t copied_data_type{};
    int32_t copied_tensor_location{};
    int32_t copied_rank{};
    int64_t copied_dims[max_rank]{};
    bool copied_is_input{};
    bool copied_is_output{};
    bool copied_is_shape_tensor{};
    bool copied_is_execution_tensor{};

private:
    static int32_t clamp_rank(int32_t rank) noexcept
    {
        if (rank < 0)
        {
            return 0;
        }

        return rank > max_rank ? max_rank : rank;
    }

    void copy_tensor_name(const char* tensor_name) noexcept
    {
        copied_name_length = 0;
        std::memset(copied_tensor_name, 0, sizeof(copied_tensor_name));
        if (tensor_name == nullptr)
        {
            return;
        }

        while (copied_name_length < max_tensor_name - 1 && tensor_name[copied_name_length] != '\0')
        {
            copied_tensor_name[copied_name_length] = tensor_name[copied_name_length];
            ++copied_name_length;
        }
    }
};

static_assert(
    !std::is_copy_constructible<DebugListenerBorrowedDebugTensorMetadataRuntimeGate>::value,
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate copy construction must stay blocked.");
static_assert(
    !std::is_copy_assignable<DebugListenerBorrowedDebugTensorMetadataRuntimeGate>::value,
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate copy assignment must stay blocked.");
static_assert(
    !std::is_move_constructible<DebugListenerBorrowedDebugTensorMetadataRuntimeGate>::value,
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate move construction must stay blocked.");
static_assert(
    !std::is_move_assignable<DebugListenerBorrowedDebugTensorMetadataRuntimeGate>::value,
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate move assignment must stay blocked.");
static_assert(
    std::is_nothrow_destructible<DebugListenerBorrowedDebugTensorMetadataRuntimeGate>::value,
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate destructor must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerBorrowedDebugTensorMetadataRuntimeGate&>().configure_api_line(11)),
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate configure_api_line must stay noexcept.");
static_assert(
    noexcept(std::declval<DebugListenerBorrowedDebugTensorMetadataRuntimeGate&>().copy_metadata(nullptr, 0, 0, 0, nullptr, false, false, false, true)),
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate copy_metadata must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerBorrowedDebugTensorMetadataRuntimeGate&>().metadata_copy_ready()),
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate metadata predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerBorrowedDebugTensorMetadataRuntimeGate&>().borrowed_tensor_pointer_escape_blocked()),
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate pointer escape predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerBorrowedDebugTensorMetadataRuntimeGate&>().borrowed_tensor_data_pointer_escape_blocked()),
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate data pointer escape predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerBorrowedDebugTensorMetadataRuntimeGate&>().borrowed_tensor_lifetime_runtime_ready()),
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate tensor lifetime predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerBorrowedDebugTensorMetadataRuntimeGate&>().borrowed_tensor_data_lifetime_runtime_ready()),
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate data lifetime predicate must stay noexcept.");
static_assert(
    noexcept(std::declval<const DebugListenerBorrowedDebugTensorMetadataRuntimeGate&>().process_debug_tensor_runtime_ready()),
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate runtime predicate must stay noexcept.");
} // namespace
