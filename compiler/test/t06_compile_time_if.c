typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];
const uint32_t ROUNDS = 4;

uint32_t __builtin_load(uint32_t addr);
vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vbroadcast(uint32_t s);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = __builtin_load(p_inp_values_p);
    vec8_t v = __builtin_vload(vals);
    for (uint32_t r = 0; r < ROUNDS; r++) {
        if (r == 0) {
            v = v + __builtin_vbroadcast(10);
        } else if (r == 1) {
            v = v + __builtin_vbroadcast(20);
        } else {
            v = v + __builtin_vbroadcast(30);
        }
    }
    __builtin_vstore(vals, v);
}
