typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

uint32_t __builtin_load(uint32_t addr);
vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vbroadcast(uint32_t s);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = __builtin_load(p_inp_values_p);
    vec8_t v = __builtin_vload(vals);
    vec8_t c = __builtin_vbroadcast(123);
    v = v + c;
    v = v ^ c;
    __builtin_vstore(vals, v);
}
