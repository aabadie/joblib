"""
Benching joblib pickle I/O.

Warning: this is slow, and the benchs are easily offset by other disk
activity.
"""
import sys
import os
import time
import shutil
import numpy as np
import joblib

from joblib.disk import disk_used

try:
    from memory_profiler import memory_usage
except ImportError:
    memory_usage = None


def clear_out():
    """Clear output directory."""
    if os.path.exists('out'):
        shutil.rmtree('out')
    os.mkdir('out')


def kill_disk_cache():
    """Clear disk cache to avoid side effects."""
    if os.name == 'posix' and os.uname()[0] == 'Linux':
        try:
            os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')
        except IOError as e:
            if e.errno == 13:
                print('Please run me as root')
            else:
                raise e
    else:
        # Write ~100M to the disk
        file('tmp', 'w').write(np.random.random(2e7))


def memory_used(func, *args, **kwargs):
    """Compute memory usage of func."""
    if memory_usage is None:
        return None

    ref_mem = memory_usage(-1, interval=.2, timeout=1, max_usage=True)
    mem_use = memory_usage((func, args, kwargs),
                           max_usage=True)

    return mem_use[0] - ref_mem


def timeit(func, *args, **kwargs):
    """Compute the mean execution time of func based on 7 measures."""
    times = list()
    for _ in range(7):
        kill_disk_cache()
        t0 = time.time()
        out = func(*args, **kwargs)
        if 1:
            # Just time the function
            t1 = time.time()
            times.append(t1 - t0)
        else:
            # Compute a hash of the output, to estimate the time
            # necessary to access the elements: this is a better
            # estimate of the time to load with me mmapping.
            joblib.hash(out)
            t1 = time.time()
            joblib.hash(out)
            t2 = time.time()
            times.append(t2 - t0 - 2*(t2 - t1))
    times.sort()
    return np.mean(times[1:-1])


def generate_rand_dict(size):
    """Generate dictionnary with random values from list of keys."""
    ret = {}
    rnd = np.random.RandomState(0)
    randoms = rnd.random_sample((size))
    for key, random in zip(range(size), randoms):
        ret[str(key)] = random
    return ret


def print_line(dataset, strategy,
               write_time, read_time,
               mem_write, mem_read,
               disk_used):
    """Nice printing function."""
    print('% 15s, %12s, % 6.3f, % 7.4f, % 9.1f, % 9.1f, % 5.1f' % (
            dataset, strategy,
            write_time, read_time,
            mem_write, mem_read, disk_used))


def bench_compress(dataset, name='',
                   compress=None, cache_size=0):
    """Bench joblib dump and load functions, compress modes."""
    time_write = list()
    time_read = list()
    du = list()
    mem_read = list()
    mem_write = list()
    clear_out()
    #time_write = timeit(joblib.dump, dataset, 'out/test.pkl',
    #                    compress=compress, cache_size=cache_size)
    time_write = 0.0
    mem_write = memory_used(joblib.dump, dataset, 'out/test.pkl',
                            compress=compress, cache_size=cache_size)

    del dataset
    du = disk_used('out')/1024.
    #time_read = timeit(joblib.load, 'out/test.pkl')
    time_read = 0.0
    mem_read = memory_used(joblib.load, 'out/test.pkl')
    print_line(name, 'compress %i' % compress,
               time_write, time_read, mem_write, mem_read, du)


def bench_mmap(dataset, name='', cache_size=0, mmap_mode='r'):
    """Bench joblib dump and load functions, memmap modes."""
    time_write = list()
    time_read = list()
    du = list()
    clear_out()
    #time_write = timeit(joblib.dump, dataset, 'out/test.pkl')
    time_write = 0.0
    mem_write = memory_used(joblib.dump, dataset, 'out/test.pkl')
    del dataset

    #time_read = timeit(joblib.load, 'out/test.pkl', mmap_mode=mmap_mode)
    time_read = 0.0
    mem_read = memory_used(joblib.load, 'out/test.pkl', mmap_mode=mmap_mode)
    du = disk_used('out')/1024.
    print_line(name, 'mmap %s' % mmap_mode,
               time_write, time_read, mem_write, mem_read, du)


def run_bench(func, obj, name, **kwargs):
    """Run the benchmark function."""
    func(obj, name, **kwargs)


class BenchObj(object):
    """Dummy benchmark object for complex pickling."""

    pass


def run(args):
    """Run the full bench suite."""
    print('% 15s, %12s, % 6s, % 7s, % 9s, % 9s, % 5s' % (
            'Dataset', 'strategy', 'write', 'read',
            'mem write', 'mem read', 'disk'))

    compress_levels = args.compress
    mmap_mode = args.mmap
    dict_size = 10000
    a1_shape = (10000, 10000)
    a2_shape = (10000000, )

    if args.nifti:
        # Nifti images
        try:
            import nibabel
        except ImportError:
            print("nibabel is not installed skipping nifti file benchmark.")
        else:
            def load_nii(filename):
                img = nibabel.load(filename)
                return img.get_data(), img.get_affine()

            for name, nifti_file in (
                    ('MNI',
                     '/usr/share/fsl/data/atlases'
                     '/MNI/MNI-prob-1mm.nii.gz'),
                    ('Juelich',
                     '/usr/share/fsl/data/atlases'
                     '/Juelich/Juelich-prob-2mm.nii.gz'),
                    ):
                for c_order in (True, False):
                    name_d = '% 5s(%s)' % (name, 'C' if c_order else 'F')
                    for compress in compress_levels:
                        d = load_nii(nifti_file)

                        if c_order:
                            d = (np.ascontiguousarray(d[0]), d[1])

                        run_bench(bench_compress, d, name_d,
                                  compress=compress)
                        del d
                    d = load_nii(nifti_file)

                    if c_order:
                        d = (np.ascontiguousarray(d[0]), d[1])

                    run_bench(bench_mmap, d, name_d,
                              mmap_mode=mmap_mode)
                    del d

    # Generate random seed
    rnd = np.random.RandomState(0)

    if args.array:
        # numpy array
        name = '% 5s' % 'Big array'
        for compress in compress_levels:
            a1 = rnd.random_sample(a1_shape)
            run_bench(bench_compress, a1, name,
                      compress=compress)
            del a1

        a1 = rnd.random_sample(a1_shape)
        run_bench(bench_mmap, a1, name, mmap_mode=mmap_mode)
        del a1

    if args.arrays:
        # Complex object with 2 big arrays
        name = '% 5s' % '2 big arrays'
        for compress in compress_levels:
            obj = BenchObj()
            obj.a1 = rnd.random_sample(a1_shape)
            obj.a2 = rnd.random_sample(a2_shape)
            run_bench(bench_compress, obj, name, compress=compress)
            del obj

        obj = BenchObj()
        obj.a1 = rnd.random_sample(a1_shape)
        obj.a2 = rnd.random_sample(a2_shape)
        run_bench(bench_mmap, obj, name, mmap_mode=mmap_mode)
        del obj

    if args.dict:
        # Big dictionnary
        name = '% 5s' % 'Big dict'
        big_dict = generate_rand_dict(dict_size)
        run_bench(bench_compress, big_dict, name,
                  compress_levels=compress_levels)
        del dict
        big_dict = generate_rand_dict(dict_size)
        run_bench(bench_mmap, big_dict, name, mmap_mode=mmap_mode)
        del dict

    if args.combo:
        # 2 big arrays with one big dict
        name = '% 5s' % 'Dict/arrays'
        for compress in compress_levels:
            obj = BenchObj()
            obj.a1 = rnd.random_sample(a1_shape)
            obj.a2 = rnd.random_sample(a2_shape)
            obj.big_dict = big_dict
            run_bench(bench_compress, obj, name, compress=compress)
            del obj

        obj = BenchObj()
        obj.a1 = rnd.random_sample(a1_shape)
        obj.a2 = rnd.random_sample(a2_shape)
        obj.big_dict = big_dict
        run_bench(bench_mmap, obj, name, mmap_mode=mmap_mode)
        del obj

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Joblib benchmark script")
    parser.add_argument('--compress', nargs='+', type=int, default=(0, 3))
    parser.add_argument('--mmap', type=str, default='r')
    parser.add_argument("-n", "--nifti", action="store_true",
                        help="Benchmark Nifti data")
    parser.add_argument("-a", "--array", action="store_true",
                        help="Benchmark single big numpy array")
    parser.add_argument("-A", "--arrays", action="store_true",
                        help="Benchmark list of big numpy arrays")
    parser.add_argument("-d", "--dict", action="store_true",
                        help="Benchmark big dictionnary")
    parser.add_argument("-c", "--combo", action="store_true",
                        help="Benchmark big dictionnary + list of "
                             "big numpy arrays.")

    run(parser.parse_args())
