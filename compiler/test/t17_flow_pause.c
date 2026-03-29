/* Emits `("flow", ("pause",))` — aligns with reference_kernel yields / Machine.enable_pause. */
typedef unsigned long uint32_t;

void __builtin_flow_pause(void);

void kernel() {
    __builtin_flow_pause();
}
