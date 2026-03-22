/* Vector lane-wise % with scalar broadcast divisor (matches idx += val % 2 in sample.c). */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t;

vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);
uint32_t load(uint32_t addr);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t val_p = load(p_inp_values_p);
    vec8_t val = vload(val_p);
    vec8_t two = vbroadcast(2);
    vec8_t m = val % two;
    vstore(val_p, m);
}
