typedef unsigned long uint32_t;

uint32_t __builtin_load(uint32_t addr);
void __builtin_store(uint32_t addr, uint32_t v);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = __builtin_load(p_inp_values_p);
    uint32_t x = __builtin_load(vals);
    x = x + 7;
    __builtin_store(vals, x);
}
