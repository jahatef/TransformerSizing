import cutlass
import torch
import random
import time

dtype = torch.float16
plan = cutlass.op.Gemm(element=dtype, layout=cutlass.LayoutType.RowMajor)

h = 2048 #20608 
m = 4
k = (h*4)
n = int(h)
b=2048
file = "results/cutlassBmmBenchMultipleTiles.txt"
def test_torch_against_cutlass(M,K,N,b):
    cutlass_sizes = [(b, M, K), (b, K, N), (b, M, N), (b, M, N)]
    A, B, C, D = [torch.randint(-3,3,size,device='cuda').to(dtype) for size in cutlass_sizes]
    tA = A.reshape((b, M, K))
    tC = C.reshape((b, M, N))
    plan.run(A, B, C, D, print_module=True)

def initialize(dtype, M, N, K, b):
    sizes = [(b, M, K), (b, K, N), (b, M, N), (b, M, N)]
    return [torch.randint(-3, 3, size, device='cuda').to(dtype) for size in sizes]


for h in [2048]: #range(20608-64,20608+64,64): #range(20608-128, 22272+128+64, 64):
    tiles = plan.tile_descriptions()
    
    if h >= 20608 and h <= 22272:
        td = tiles[1]
    else:
        td = tiles[0]
    plan.compile(td)
    m = 2048# int(32)
    k = h #int(h*4)
    n = int(h)
    b=512 #2048
    As, Bs, Cs, Ds, = initialize(dtype, m, n, k, b)
    num_warm = 0
    for i in range( num_warm + 1):
        if i == num_warm:
            start_time = time.time()
        plan.run(As, Bs, Cs, Ds, print_module=True, batch_count=b, sync=True)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / 20
    with open(file, 'a') as f:
        f.write(f"Elapsed time for {m}x{n}x{k}, b={b}: {elapsed_time:.4f}\n")
        f.write(f"Throughput (in TFLOP/s) for {m}x{n}x{k}, b={b}: "
            f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}\n") 
        f.write("-" * 80)
        f.write("\n")

'''
for i in range(100):
    td = tiles[i]
    plan.compile(td)
    As, Bs, Cs, Ds, = generate_problems(b, m, n, k)
    plan.run(As, Bs, Cs, Ds, print_module=True)
    num_warm = 10
    for i in range( num_warm + 20):
        if i == num_warm:
            start_time = time.time()
        plan.run(As, Bs, Cs, Ds, print_module=True)
    elapsed_time = (time.time() - start_time) / 20
    with open(file, 'a') as f:
        f.write(f"Tile description: \n {td} \n")
        f.write(f"Elapsed time for {m}x{n}x{k}, b={b}: {elapsed_time:.4f}\n")
        f.write(f"Throughput (in TFLOP/s) for {m}x{n}x{k}, b={b}: "
            f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}\n") 
        f.write("-" * 80)
        f.write("\n")
'''
