typedef unsigned long uint32_t;

uint32_t load(uint32_t addr);
void store(uint32_t addr, uint32_t v);
void debug(uint32_t value, uint32_t round, uint32_t batch, uint32_t tag);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = load(p_inp_values_p);
    uint32_t x = load(vals);
    x = x + 1;
    debug(x, 0, 0, 1);
    store(vals, x);
}
