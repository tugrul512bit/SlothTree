# SlothTree

If a Ryzen CPU core is a bear, then a single CUDA pipeline is a sloth. If both compete to get apple from a tree, 1 at a time, the bear always wins by a large margin:

- bear: 1 apple per 15 seconds --> 240 apples per hour
- sloth: 1 apple per hour

But, when we have a lot of sloths:

- 5888 sloths: 5888 apples per hour

Sloths win.

---

Building tree:

Up to 40x faster than std::unordered_map for inserting same number of random elements.

Finding element:

Under development. Ideas: sorting the input should improve the searching because of reducing warp divergence. Sorting between each depth phase should also decrease warp divergence but these require only partial sorting. Perhaps sorting should start only after a certain depth and be fully parallel like sorting 10000 arrays of sizes 128. Without any sorting, maybe binning can help. But it requires extra space.

Sample code:
```C++
#include"tree.cuh"

int main()
{

    int n = 1000000;
    Sloth::Tree<int, int> tree;
    std::unordered_map<int, int> map;
    std::vector<int> key(n);
    std::vector<int> value(n);
    for (int i = 0; i < n; i++)
    {
        key[i] = rand() * rand() + rand();
        value[i] = i;
    }
    for (int i = 0; i < 10; i++)
    {
        tree.Build(key, value);

        map.clear();
        size_t t;
        {
            Sloth::Bench bench(&t);
            for (int j = 0; j < n; j++)
                map[key[j]] = value[j];
        }
        std::cout << "cpu: " << t / 1000000000.0 << "s" << std::endl;

    }

    int valueFound=-1;
    std::cout << " found: " << tree.FindKeyCpu(key[15],valueFound) << std::endl;
    std::cout << " found value: " << valueFound << " real value: " << value[15] << std::endl;
    return 0;
}

```

Output:
```
gpu: 0.0030451s
cpu: 0.163003s
gpu: 0.0021143s
cpu: 0.111421s
gpu: 0.0027288s
cpu: 0.0929716s
gpu: 0.0021829s
cpu: 0.101415s
gpu: 0.0020446s
cpu: 0.0716856s
gpu: 0.002245s
cpu: 0.0772123s
gpu: 0.0021036s
cpu: 0.0698298s
gpu: 0.0021068s
cpu: 0.0810439s
gpu: 0.0021056s
cpu: 0.0718065s
gpu: 0.0021467s
cpu: 0.0812151s
 found: 1
 found value: 15 real value: 15
```

# Building a Tree With CUDA

- Input array is taken as a single chunk
- Chunk is mapped to N number of tasks
- Each task is a single block of CUDA threads
- Single kernel launch computes all tasks
- Preprocessing: array's value range is computed with a min-max reduction. Reduction in registers, then reduction in shared memory then atomicMin and atomicMax once per block (N per task). This is written to root chunk's properties.
- Phase 1: tasks count occurences of keys falling in boundaries of child nodes (if there are 4 child nodes per parent, then there are 4 accumulators for 4 different key ranges that divide parent's range equally)
- Phase 2: tasks calculate offsets or starting index values of keys of child nodes
- Phase 3: tasks allocate enough space on target array and copy the contents of nodes (chunks) to target.
- Phase 4: tasks generate new tasks depending on node depth level, number of elements inside node and size of child node ranges
- Phase 5: book-keeping variables are resetted and loops to phase 1 until there are no tasks generated
- All allocations are only computed on a preallocated array by using atomicAdd on an offset variable.
- All objects are in form of struct-of-arrays because SOA is more efficient to read/write in parallel than AOS. When a field is required, only that information is accessed and memory bank conflicts, serializations are minimized.
- Tree of 1 million elements take roughly 2 milliseconds on a RTX4070 with 5888 CUDA pipelines while inserting same data to an std::unordered_map takes 70-90 milliseconds on a Ryzen7900 core. Despite having 40% warp occupancy and 50% of maximum in-flight warps, GPU is nearly 40 times faster to build a tree. 

![Top-down approach](https://github.com/tugrul512bit/SlothTree/blob/master/sloth-tree.drawio.png)
