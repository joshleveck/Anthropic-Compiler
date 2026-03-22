/* Emits `("flow", ("pause",))` — aligns with reference_kernel yields / Machine.enable_pause. */
typedef unsigned long uint32_t;

void flow_pause(void);

void kernel() {
    flow_pause();
}
