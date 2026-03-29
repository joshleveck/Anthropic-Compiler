typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

uint32_t __builtin_load(uint32_t addr);
vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vbroadcast(uint32_t s);
vec8_t __builtin_vselect(vec8_t cond, vec8_t a, vec8_t b);

uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t idx_p = __builtin_load(p_inp_indices_p);
    uint32_t val_p = __builtin_load(p_inp_values_p);
    vec8_t idx = __builtin_vload(idx_p);
    vec8_t val = __builtin_vload(val_p);

    vec8_t one = __builtin_vbroadcast(1);
    vec8_t two = __builtin_vbroadcast(2);
    vec8_t cond = idx % two;  // lane != 0 means choose "a"
    vec8_t out = __builtin_vselect(cond, one, val);

    __builtin_vstore(val_p, out);
}
