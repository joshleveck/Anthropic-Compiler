typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];

uint32_t __builtin_load(uint32_t addr);
vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t p = __builtin_load(p_inp_values_p);
    vec8_t addrs = __builtin_vload(p);
    vec8_t node_val;
    for (int vi = 0; vi < 8; vi++) {
        node_val[vi] = __builtin_load(addrs[vi]);
    }
    __builtin_vstore(p, node_val);
}
