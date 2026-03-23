/* Vector lane-wise % with scalar broadcast divisor (matches idx += val % 2 in sample.c). */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t;

vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vbroadcast(uint32_t s);
uint32_t __builtin_load(uint32_t addr);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t val_p = __builtin_load(p_inp_values_p);
    vec8_t val = __builtin_vload(val_p);
    vec8_t two = __builtin_vbroadcast(2);
    vec8_t m = val % two;
    __builtin_vstore(val_p, m);
}
