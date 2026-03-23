typedef unsigned long uint32_t;

uint32_t __builtin_load(uint32_t addr);
void __builtin_store(uint32_t addr, uint32_t v);
void __builtin_debug(uint32_t value, uint32_t round, uint32_t batch, uint32_t tag);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = __builtin_load(p_inp_values_p);
    uint32_t x = __builtin_load(vals);
    x = x + 1;
    __builtin_debug(x, 0, 0, 1);
    __builtin_store(vals, x);
}
