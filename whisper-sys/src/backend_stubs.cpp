// Minimal stub implementations for missing backend registry functions
// These are only used by optional features in whisper.cpp

extern "C" {

// Registry functions - minimal stubs for default CPU backend
size_t ggml_backend_reg_count(void) {
    return 1; // Only CPU backend
}

void * ggml_backend_reg_get(size_t index) {
    if (index == 0) {
        return nullptr; // CPU backend doesn't need special registration
    }
    return nullptr;
}

// Device functions - minimal stubs
size_t ggml_backend_dev_count(void * reg) {
    return 1; // Only default device
}

void * ggml_backend_dev_get(void * reg, size_t index) {
    if (index == 0) {
        return nullptr; // Default device
    }
    return nullptr;
}

void * ggml_backend_dev_by_type(int type) {
    return nullptr; // Default device for any type
}

void * ggml_backend_init_by_type(int type, const char * params) {
    return nullptr; // Will fall back to default backend
}

} // extern "C"