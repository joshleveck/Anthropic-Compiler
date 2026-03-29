typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];
const uint32_t VLEN = 8;

uint32_t __builtin_load(uint32_t addr);
void __builtin_store(uint32_t addr, uint32_t v);
vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);

uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t idx_p = __builtin_load(p_inp_indices_p);
    uint32_t val_p = __builtin_load(p_inp_values_p);
    vec8_t idx = __builtin_vload(idx_p);
    vec8_t node;

    for (uint32_t i = 0; i < 8; i++) {
        node[i] = __builtin_load(val_p + i);
    }

    for (uint32_t i = 0; i < 8; i++) {
        idx[i] = idx[i] + node[i];
    }

    __builtin_vstore(idx_p, idx);
    __builtin_store(val_p, idx[0]);  // scalar indexed read path
}
