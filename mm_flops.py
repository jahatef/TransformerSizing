import time
import torch


def benchmark_mm(m, n, k, num_iterations=100):
    A = torch.randn(m, n).half().to("cuda:0")
    B = torch.randn(n, k).half().to("cuda:0")
    C = torch.empty(m, k).half().to("cuda:0")
    num_warmup_iterations = 50
    for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            torch.mm(A, B, out=C)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations
    print(f"Elapsed time for {m}x{n}x{k}: {elapsed_time:.3f}")
    print(f"Throughput (in TFLOP/s) for {m}x{n}x{k}: {(2 * m * n * k) / (elapsed_time * 10**12):.3f}")
    print("-" * 80)




def benchmark_mm_b(m, n, k, b=None, num_iterations=100):
    
    B = torch.randn(k, n).half().to("cuda:0")
    if b is None:
        A = torch.randn(m, n).half().to("cuda:0")
        b=1
        C = torch.empty(m, k).half().to("cuda:0")
    else:
        A = torch.randn(b,m,n).half().to("cuda:0")
        C = torch.empty(b,m, k).half().to("cuda:0")
    num_warmup_iterations = 50
    for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            torch.nn.functional.linear(A, B, out=C)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations
    if b is None:
        print(f"Elapsed time for {m}x{n}x{k}: {elapsed_time:.3f}")
        print(f"Throughput (in TFLOP/s) for {m}x{n}x{k}: {(2 * m * n * k) / (elapsed_time * 10**12):.3f}")
    else:
        print(f"Elapsed time for {m}x{n}x{k}, b={b}: {elapsed_time:.4f}")
        print(f"Throughput (in TFLOP/s) for {m}x{n}x{k}, b={b}: "
          f"{(2 * b * m * n * k) / (elapsed_time * 10**12):.3f}") 
    print("-" * 80)

def benchmark_mm_concat(m, n, k, b=None, num_iterations=100):
    M = b*m
    A = torch.randn(M, n).half().to("cuda:0")
    b=1
    C = torch.empty(M, k).half().to("cuda:0")
    B = torch.randn(n, k).half().to("cuda:0")
    #C = torch.empty(m, k).half().to("cuda:0")
    num_warmup_iterations = 50
    for i in range(num_warmup_iterations + num_iterations):
        if i == num_warmup_iterations:
            start_time = time.time()
        with torch.no_grad():
            torch.nn.functional.linear(A, B.T , out=C)
        torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iterations
    print(f"Elapsed time for concat {M}x{n}x{k}: {elapsed_time:.3f}")
    print(f"Throughput (in TFLOP/s) for concat {M}x{n}x{k}: {(2 * m * n * k) / (elapsed_time * 10**12):.3f}")
    print("-" * 80)

if __name__ == '__main__':
    torch.cuda.set_device("cuda:0")

    # Figure 3. basicGemmMSweep.out
    #for log_size in range(5, 14):
    #    benchmark_mm(2**log_size, 4096, 2**log_size)

    # Figure 7. basicGemmKSweep.out
    #for k in range(64, 2**15, 64):
    #    benchmark_mm(2048, 2048, k)

    # Figure 8. basicGemmLargeKSweep.out
    #for k in range(1536, 6208, 64):
    #    benchmark_mm(2304, 4096, k)

    # m from 1024 to 10000.
    #for m in range(64, 2**15, 64):
    #    benchmark_mm(m, 2048, 2048)

    #n from 64 to 512
    #for n in range(64, 2**15, 64):
    #    benchmark_mm(2048,n,2048)
    

    #for nk in range( 64, 2**15, 64):
    #    benchmark_mm(2048, 4*nk, nk)


    #for mn in range(64, 4096, 8):
    #    benchmark_mm(mn,2048,mn)

    #batch vs concat
    #for n in range(64, 4096, 64):
    #    benchmark_mm_b(2048,n,2048, b=4)
    #    benchmark_mm_concat(2048, n, 2048, b=4)
    
    #profile linear projection
    #benchmark_mm_b(4,13056,13056,b=2048)

    #sweep nk
    #for logB in range(4,6):
    #    B = 2**logB
    #    for n in range(64, 2**15, 64):
    #        benchmark_mm_b(2048, n, 2048, b=B)

    #sweep nk in area of low speed
    #for hidden_size in range(22976,25024+64,64):
    #    benchmark_mm_b(4, hidden_size, hidden_size, b=2048)

    #profile separate arbitrary region
    
    #for hidden_size in range( 64, 2**15, 64):
    #    benchmark_mm_b(4, 3*hidden_size, hidden_size, b=2048)

    #h to 4h drop
    #for h in range(128,2**15,128): 
    #    benchmark_mm_b(2048,h, 3*h, b=4)

    #for h in range(128,2**15,128): 
    #    benchmark_mm_b(4,h, 3*h, b=2048)

    #for h in range(128,2**15,128): 
    #    benchmark_mm_b(4*2048,h, 3*h)

    b=4
    s=2048
    v=51200
    h=14336
    for h in range(14336-64, 14336 + 65): 
        benchmark_mm_b(b*s,h,v)
    h = 14336
    for v in range(51200-64, 51200 + 65): 
        benchmark_mm_b(b*s,h,v)