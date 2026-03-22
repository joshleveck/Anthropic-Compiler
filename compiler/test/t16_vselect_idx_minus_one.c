/* sample.c r==1 branch: t = idx - 1; node = vselect(t, vtwo, vone); */
typedef unsigned long uint32_t;
typedef uint32_t vec8_t;

uint32_t load(uint32_t addr);
vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);
vec8_t vselect(vec8_t cond, vec8_t a, vec8_t b);

uint32_t *p_forest_values_p = (uint32_t *)4;
uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t forest_values_p = load(p_forest_values_p);
    uint32_t idx_p = load(p_inp_indices_p);
    uint32_t val_p = load(p_inp_values_p);
    vec8_t idx = vload(idx_p);
    vec8_t one = vbroadcast(1);
    vec8_t t = idx - one;
    vec8_t vone = vbroadcast(load(forest_values_p + 1));
    vec8_t vtwo = vbroadcast(load(forest_values_p + 2));
    vec8_t node = vselect(t, vtwo, vone);
    vstore(val_p, node);
}
