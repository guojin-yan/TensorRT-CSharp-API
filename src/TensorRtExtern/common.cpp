#include "common.h"
#include <mutex>

static std::string last_err_msg;
static std::mutex last_msg_mutex;


char* str_to_char_array(const std::string& str) {
    std::unique_ptr<char> _char_array(new char[str.length() + 1]);
    char* char_array = _char_array.release();
    std::copy_n(str.c_str(), str.length() + 1, char_array);
    return char_array;
}

void dup_last_err_msg(const char* msg) {
    std::lock_guard<std::mutex> lock(last_msg_mutex);
    last_err_msg = std::string(msg);
}

const char* trt_get_last_err_msg() {
    std::lock_guard<std::mutex> lock(last_msg_mutex);
    char* res = nullptr;
    if (!last_err_msg.empty()) {
        res = str_to_char_array(last_err_msg);
    }
    return res;
}
