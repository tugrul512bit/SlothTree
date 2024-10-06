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

Sample benchmark code:
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
        key[i] =  rand()*rand()+rand();
        value[i] = i;
    }

    // build tests
    for (int i = 0; i < 5; i++)
    {
        std::cout << "------------------------------------------------------------------------------------" << std::endl;
        int valueFound = -1;
        bool found = false;
        for (int i = 0; i < 5; i++)
        {
            tree.Build(key, value);
            
            valueFound = -1;
            found |= tree.FindKeyCpu(key[15], valueFound);
            if (valueFound == -1)
            {
                // try again to check if only "find" is failed or "build" is failed
                found = false;
                valueFound = -1;
                found |= tree.FindKeyCpu(key[15], valueFound);
                if (valueFound == -1)
                {
                    std::cout << "Build-failure!" << std::endl;
                    std::cout << " found: " << (found ? "yes" : "no") << std::endl;
                    std::cout << " found value: " << valueFound << " real value: " << value[15] << std::endl;
                    std::cout << "error: tree traversal could not find the key that was inserted." << std::endl;
                    return 0;
                }
                
                std::cout << "Find-failure!" << std::endl;
                std::cout << " found: " << (found?"yes":"no")<<std::endl;
                std::cout << " found value: " << valueFound << " real value: " << value[15] << std::endl;
                std::cout << "error: tree traversal could not find the key that was inserted." << std::endl;
                return 0;
            }
        }
        std::cout << " found: " << (found ? "yes" : "no") << std::endl;
        std::cout << " found value: " << valueFound << " real value: " << value[15] << std::endl;
        map.clear();
        size_t t;
        {
            Sloth::Bench bench(&t);
            for (int j = 0; j < n; j++)
                map[key[j]] = value[j];
        }
        std::cout << "build cpu: " << t / 1000000000.0 << "s" << std::endl;

    }


    std::cout << "1x brute-force cpu find: " << std::endl;


    // search tests
    // nSearch consumes too much video-memory. beware.
    // it allocates enough queue-space for nSearch CUDA threads
    // if 1 million keys are to be searched, do it in 100 steps using 10k chunks
    const int nSearch = 10000;
    std::vector<int> keys(nSearch);
    std::vector<int> values(nSearch);
    std::vector<char> conditions(nSearch);
    std::vector<int> valuesBruteForce(nSearch);
    std::vector<char> conditionsBruteForce(nSearch);
    for (int i = 0; i < nSearch; i++)
    {
        keys[i] = rand() * rand() + rand();        
    }


    for (int j = 0; j < 3; j++)
    {
        size_t t;
        {
            Sloth::Bench bench(&t);
            for (int j = 0; j < nSearch; j++)
            {
                bool cond = false;
                int val = -1;
                for (int i = 0; i < n; i++)
                {
                    if (keys[j] == key[i])
                    {
                        cond = true;
                        val = value[i];
                        break;
                    }
                }
                conditionsBruteForce[j] = cond;
                valuesBruteForce[j] = val;
            }
        }

        std::cout << "brute force find cpu: " << t / 1000000000.0 << "s" << std::endl;
        for (int i = 0; i < 15; i++)
        {
            {
                Sloth::Bench bench(&t);
                tree.FindKeyGpu(keys, values, conditions);
            }
            std::cout << "simple find gpu: " << t / 1000000000.0 << "s" << std::endl;
        }
        //checking error
        for (int i = 0; i < nSearch; i++)
        {
            bool fail = false;
            if (conditionsBruteForce[i] != conditions[i])
            {
                std::cout << "Error: tree-find failed (condition)!" << std::endl;
                fail = true;
            }

            if (valuesBruteForce[i] != values[i])
            {
                std::cout << "Error: tree-find failed (value)!" << std::endl;
                fail = true;
            }
            if (fail)
            {
                std::cout << "tree result: " << values[i] << " brute-force result: " << valuesBruteForce[i] << std::endl;
                std::cout << "tree condition: " << (int) conditions[i] << " brute-force condition: " <<  (int)conditionsBruteForce[i] << std::endl;
                return 0;
            }
        }
    }
    return 0;
}

```

Output:
```
------------------------------------------------------------------------------------
build gpu: 0.0078295s
build gpu: 0.0025396s
build gpu: 0.0024831s
build gpu: 0.0024197s
build gpu: 0.0024611s
 found: yes
 found value: 15 real value: 15
build cpu: 0.158406s

------------------------------------------------------------------------------------
build gpu: 0.0026996s
build gpu: 0.0025182s
build gpu: 0.0024496s
build gpu: 0.0024429s
build gpu: 0.0024122s
 found: yes
 found value: 15 real value: 15
build cpu: 0.0748112s
------------------------------------------------------------------------------------
brute force find cpu: 1.84401s
simple find gpu: 0.0561698s
simple find gpu: 0.0006346s
simple find gpu: 0.0006137s
simple find gpu: 0.0006114s
simple find gpu: 0.0006114s
simple find gpu: 0.0006114s
simple find gpu: 0.0006109s
simple find gpu: 0.0006105s
simple find gpu: 0.0006329s
simple find gpu: 0.0006256s
simple find gpu: 0.0006126s
simple find gpu: 0.0006103s
simple find gpu: 0.0006665s
simple find gpu: 0.0006237s
simple find gpu: 0.0006121s
brute force find cpu: 1.85026s
simple find gpu: 0.0006857s
simple find gpu: 0.0006161s
simple find gpu: 0.0006146s
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

# Separating Chunks

- Allocating chunks requires counting the number of elements. This is computed with parallel reduction in shared memory and warp shuffle. From 128 elements to 32 elements, shared memory is used. From 32 elements to 1 element, warp shuffle is used.
- Moving millions of elements to their target with only atomics-based offset computation is slow. Due to this, multiple elements are extracted from 128-sized regions by parallel stream-compaction.
- This is repeated for each child node chunk

![counting & compacting](https://github.com/tugrul512bit/SlothTree/blob/master/separating-chunks.drawio.png)
