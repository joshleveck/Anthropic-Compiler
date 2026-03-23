/* Outer batch loop with two 8-lane chunks (batch 0 and batch 8) — like sample batch stride. */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t;

uint32_t __builtin_load(uint32_t addr);
vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vbroadcast(uint32_t s);

uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

const uint32_t BATCH_SIZE = 16;

void kernel() {
    uint32_t inp_indices_p = __builtin_load(p_inp_indices_p);
    uint32_t inp_values_p = __builtin_load(p_inp_values_p);
    for (uint32_t batch = 0; batch < BATCH_SIZE; batch += 8) {
        uint32_t idx_addr = inp_indices_p + batch;
        uint32_t val_addr = inp_values_p + batch;
        vec8_t idx = __builtin_vload(idx_addr);
        vec8_t val = __builtin_vload(val_addr);
        val = val + __builtin_vbroadcast(1);
        __builtin_vstore(idx_addr, idx);
        __builtin_vstore(val_addr, val);
    }
}
