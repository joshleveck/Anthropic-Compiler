typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];
const uint32_t ROUNDS = 4;

uint32_t load(uint32_t addr);
vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);

uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel() {
    uint32_t vals = load(p_inp_values_p);
    vec8_t v = vload(vals);
    for (uint32_t r = 0; r < ROUNDS; r++) {
        if (r == 0) {
            v = v + vbroadcast(10);
        } else if (r == 1) {
            v = v + vbroadcast(20);
        } else {
            v = v + vbroadcast(30);
        }
    }
    vstore(vals, v);
}
