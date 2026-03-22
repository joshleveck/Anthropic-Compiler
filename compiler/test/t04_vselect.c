typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

uint32_t load(uint32_t addr);
vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);
vec8_t vselect(vec8_t cond, vec8_t a, vec8_t b);

uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t idx_p = load(p_inp_indices_p);
    uint32_t val_p = load(p_inp_values_p);
    vec8_t idx = vload(idx_p);
    vec8_t val = vload(val_p);

    vec8_t one = vbroadcast(1);
    vec8_t two = vbroadcast(2);
    vec8_t cond = idx % two;  // lane != 0 means choose "a"
    vec8_t out = vselect(cond, one, val);

    vstore(val_p, out);
}
