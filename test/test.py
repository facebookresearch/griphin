import time
import torch.multiprocessing as mp


def foo(rank, x):
    print(x*x)
    time.sleep(1)
    return x * x


def start_p():
    p = mp.Process(target=foo)
    p.start()


def start_batch_p(num_process):
    with mp.Pool(num_process) as pool:
        futs = [pool.apply_async(foo, args=(i, )) for i in range(100)]
        # results = [fut.get() for fut in futs]
        # print(results)
        pool.close()
        pool.join()


def test_batching():
    num_source = 11
    num_process = 5
    num_data = int(num_source / num_process)
    # assert num_source % num_process == 0

    for i in range(num_process-1):
        print(i * num_data, (i + 1) * num_data)

    print((num_process-1)*num_data, num_source)


if __name__ == '__main__':
    # start_batch_p(5)
    mp.spawn(foo, nprocs=10, args=(1,), join=True)
    # test_batching()


