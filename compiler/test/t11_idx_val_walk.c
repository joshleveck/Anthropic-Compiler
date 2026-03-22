/* One step of sample index walk: idx = idx*2+1; idx = idx + (val % 2); */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t;

uint32_t load(uint32_t addr);
vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);

uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t idx_p = load(p_inp_indices_p);
    uint32_t val_p = load(p_inp_values_p);
    vec8_t idx = vload(idx_p);
    vec8_t val = vload(val_p);
    idx = idx * 2 + 1;
    idx = idx + (val % 2);
    vstore(idx_p, idx);
    vstore(val_p, val);
}
