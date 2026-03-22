typedef unsigned long uint32_t;

uint32_t load(uint32_t addr);
void store(uint32_t addr, uint32_t v);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = load(p_inp_values_p);
    uint32_t x = load(vals);
    uint32_t y = load(vals + 1);
    x = x + 10;
    y = y + 20;
    store(vals, x);
    store(vals + 1, y);
}
