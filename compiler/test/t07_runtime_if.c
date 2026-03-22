typedef unsigned long uint32_t;

uint32_t load(uint32_t addr);
void store(uint32_t addr, uint32_t v);

uint32_t *p_rounds = (uint32_t *)0;
uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t rounds = load(p_rounds);
    uint32_t vals = load(p_inp_values_p);
    uint32_t x = load(vals);
    if (rounds == 0) {
        x = x + 111;
    } else {
        x = x + 222;
    }
    store(vals, x);
}
