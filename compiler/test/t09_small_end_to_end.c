typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];
const uint32_t VLEN = 8;
const uint32_t ROUNDS = 2;
const uint32_t FOREST_HEIGHT = 3;

uint32_t load(uint32_t addr);
void store(uint32_t addr, uint32_t v);
vec8_t vload(uint32_t addr);
void vstore(uint32_t addr, vec8_t v);
vec8_t vbroadcast(uint32_t s);
vec8_t vselect(vec8_t cond, vec8_t a, vec8_t b);

vec8_t vhash(vec8_t a)
{
    a = (a + 0x7ED55D16) + (a << 12);
    a = (a ^ 0xC761C23C) ^ (a >> 19);
    a = (a + 0x165667B1) + (a << 5);
    a = (a + 0xD3A2646C) ^ (a << 9);
    a = (a + 0xFD7046C5) + (a << 3);
    a = (a ^ 0xB55A4F09) ^ (a >> 16);
    return a;
}

uint32_t *p_forest_values_p = (uint32_t *)4;
uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel()
{
    uint32_t forest_values_p = load(p_forest_values_p);
    uint32_t idx_p = load(p_inp_indices_p);
    uint32_t val_p = load(p_inp_values_p);

    vec8_t idx = vload(idx_p);
    vec8_t val = vload(val_p);
    uint32_t root = load(forest_values_p + 0);
    vec8_t vroot = vbroadcast(root);

    for (uint32_t r = 0; r < ROUNDS; r++)
    {
        vec8_t node;
        if (r == 0)
        {
            node = vroot;
        }
        else
        {
            for (uint32_t i = 0; i < 8; i++)
            {
                node[i] = load(forest_values_p + idx[i]);
            }
        }
        val = vhash(val ^ node);
        idx = idx * 2 + 1;
        idx = idx + (val % 2);
        if (r == FOREST_HEIGHT)
        {
            idx = vbroadcast(0);
        }
    }

    vstore(idx_p, idx);
    vstore(val_p, val);
}
