typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

uint32_t load(uint32_t addr);
vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t p = load(p_inp_values_p);
    vec8_t addrs = vload(p);
    vec8_t node_val;
    for (int vi = 0; vi < 8; vi++) {
        node_val[vi] = load(addrs[vi]);
    }
    vstore(p, node_val);
}
