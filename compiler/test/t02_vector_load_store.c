typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
uint32_t __builtin_load(uint32_t addr);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = __builtin_load(p_inp_values_p);
    vec8_t v = __builtin_vload(vals);
    __builtin_vstore(vals, v);
}
