import torch
from triton.testing import do_bench

def get_flops_mm(M,N,K,b=None, get_kernels=False, file= None):
    A = torch.randn((b, M, N), device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C = torch.randn((b, M, K), device='cuda', dtype=torch.float16)
    def f():
        return torch.nn.functional.linear(A, B, out=C)

    if get_kernels:
        with torch.profiler.profile() as prof:
            f()

        for e in prof.events():
            #print(e.name)
            if "gemm" in e.name or "triton" in e.name or "gemv" in e.name:
                if file:
                    print(f"{N}: {e.name}", file = open(file, 'a'))
                else:
                    print(f"{N}: {e.name}")
                timer = e.cuda_time/1e3
    timer = do_bench(f)[0]
    iters_per_second = 1e3/timer
    flops = A.shape[0] * A.shape[1] * A.shape[2] * B.shape[0] * 2
    flops_achieved = iters_per_second * flops/1e12
    if file:
        print(f"{M}x{N}x{K}, B={b}: {flops_achieved:.2f}TF/s", file = open(file, 'a'))
    else: 
        print(f"{M}x{N}x{K}, B={b}: {flops_achieved:.2f}TF/s")

def get_flops_bmm(M,N,K,b=None, get_kernels=False):
    A = torch.randn(b, M, N, device='cuda', dtype=torch.float16)
    B = torch.randn(b, N, K, device='cuda', dtype=torch.float16)
    C = torch.randn(b, M, K, device='cuda', dtype=torch.float16)
    def f():
        return torch.bmm(A, B, out=C)

    if get_kernels:
        with torch.profiler.profile() as prof:
            f()

        #for e in prof.events():
        #    print(e.name)
            #if "gemm" in e.name or "triton" in e.name or "gemv" in e.name:
            #    print(f"{N}: {e.name}")
            #    timer = e.cuda_time/1e3
    #timer = do_bench(f)[0]
    #iters_per_second = 1e3/timer
    #flops = 2* M * N * K * b 
    #flops_achieved = iters_per_second * flops/1e12
    #print(f"{M}x{N}x{K}, B={b}: {flops_achieved:.2f}TF/s")
        print("-" * 40)


def mm_for_profiles(M,N,K,b=None, get_kernels=False, file= None):
    A = torch.randn((b, M, N), device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C = torch.randn((b, M, K), device='cuda', dtype=torch.float16)
    def f():
        return torch.nn.functional.linear(A, B, out=C)

    f()

file = "attention_kvq_transform_kernels.txt"

M = 4

for N in range(9472-64, 9472+64, 64):
    #get_flops_mm(M, N, N*3, b=2048, get_kernels=True, file = file)
    mm_for_profiles(M, N, N*3, b=2048, get_kernels=True, file = file)