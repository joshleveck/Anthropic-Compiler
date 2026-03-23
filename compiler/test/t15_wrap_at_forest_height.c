/* idx = __builtin_vbroadcast(0) when r == FOREST_HEIGHT (sample wrap-to-root). */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t;

uint32_t __builtin_load(uint32_t addr);
vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vbroadcast(uint32_t s);

uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

const uint32_t ROUNDS = 4;
const uint32_t FOREST_HEIGHT = 3;

void kernel() {
    uint32_t idx_p = __builtin_load(p_inp_indices_p);
    uint32_t val_p = __builtin_load(p_inp_values_p);
    vec8_t idx = __builtin_vload(idx_p);
    vec8_t val = __builtin_vload(val_p);
    for (uint32_t r = 0; r < ROUNDS; r++) {
        idx = idx + __builtin_vbroadcast(1);
        if (r == FOREST_HEIGHT) {
            idx = __builtin_vbroadcast(0);
        }
    }
    __builtin_vstore(idx_p, idx);
    __builtin_vstore(val_p, val);
}
