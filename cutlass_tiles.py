import numpy as np
import cutlass
import random
import numpy as np
import rmm 

rmm.reinitialize(pool_allocator=True,initial_pool_size=15e+9,managed_memory=True,maximum_pool_size=30e+9)

import cutlass

# This controls whether ther C++ GEMM declaration will be printed at each step. Set to `false` to
# omit this information.
print_module = True

m = 128
n = m
k = m

dtype = np.float16
type_A = np.float16
type_B = np.float16
type_C = np.float16
type_D = np.float16

np.random.seed(1234)
random.seed(1234)
scope_min = -4
scope_max = 4

alpha = np.float16(1.)
beta = np.float16(0.)
h = 20608 
m = 4
k = (h*4)
n = int(h)

tensor_A = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(m, k)).astype(type_A))
tensor_B = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(k, n)).astype(type_B))
tensor_C = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(m, n)).astype(type_C))
tensor_D = np.zeros(tensor_C.shape).astype(type_D)

print(f"GEMM Size: ({m}x{n}x{k})")
plan = cutlass.Gemm(element=dtype, layout=cutlass.LayoutType.RowMajor, element_accumulator=np.float32)
#plan.run(tensor_A, tensor_B, tensor_C, tensor_D, print_module=print_module)
print("done")
plan.opclass = cutlass.OpcodeClass.TensorOp
#plan.run(tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta, print_module=print_module)

tiles = plan.tile_descriptions()
#print([td for td in tiles])
#print( [tile for tile in tiles if tile.ThreadblockShape == [128,256,64]])
print('{} tile descriptions returned'.format(len(tiles)))
num_print = 10
print('First {} tile descriptions are:'.format(num_print))
for td in tiles[:num_print]:
    print(td)



idx = 1
td = tiles[idx]
print('Tile description {} is: {}'.format(idx, td))
plan.compile(td)
plan.run(tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta, print_module=print_module)

