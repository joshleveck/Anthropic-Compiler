typedef unsigned long uint32_t;

uint32_t __builtin_load(uint32_t addr);
void __builtin_store(uint32_t addr, uint32_t v);
void __builtin_sync(void);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = __builtin_load(p_inp_values_p);
    uint32_t x = __builtin_load(vals);
    uint32_t y = __builtin_load(vals + 1);
    x = x + 1;
    __builtin_sync();
    y = y + 2;
    __builtin_store(vals, x);
    __builtin_store(vals + 1, y);
}
