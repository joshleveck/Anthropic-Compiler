typedef unsigned long uint32_t;
typedef uint32_t vec8_t[8];
const uint32_t VLEN = 8;
const uint32_t ROUNDS = 2;
const uint32_t FOREST_HEIGHT = 3;

uint32_t __builtin_load(uint32_t addr);
void __builtin_store(uint32_t addr, uint32_t v);
vec8_t __builtin_vload(uint32_t addr);
void __builtin_vstore(uint32_t addr, vec8_t v);
vec8_t __builtin_vbroadcast(uint32_t s);
vec8_t __builtin_vselect(vec8_t cond, vec8_t a, vec8_t b);
vec8_t __builtin_vhash(vec8_t a);

uint32_t *p_forest_values_p = (uint32_t *)4;
uint32_t *p_inp_indices_p = (uint32_t *)5;
uint32_t *p_inp_values_p = (uint32_t *)6;

void kernel()
{
    uint32_t forest_values_p = __builtin_load(p_forest_values_p);
    uint32_t idx_p = __builtin_load(p_inp_indices_p);
    uint32_t val_p = __builtin_load(p_inp_values_p);

    vec8_t idx = __builtin_vload(idx_p);
    vec8_t val = __builtin_vload(val_p);
    uint32_t root = __builtin_load(forest_values_p + 0);
    vec8_t vroot = __builtin_vbroadcast(root);

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
                node[i] = __builtin_load(forest_values_p + idx[i]);
            }
        }
        val = __builtin_vhash(val ^ node);
        idx = idx * 2 + 1;
        idx = idx + (val % 2);
        if (r == FOREST_HEIGHT)
        {
            idx = __builtin_vbroadcast(0);
        }
    }

    __builtin_vstore(idx_p, idx);
    __builtin_vstore(val_p, val);
}
