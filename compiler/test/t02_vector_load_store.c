typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
uint32_t load(uint32_t addr);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = load(p_inp_values_p);
    vec8_t v = vload(vals);
    vstore(vals, v);
}
