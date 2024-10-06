# SlothTree

If a Ryzen CPU core is a bear, then a single CUDA pipeline is a sloth. If both compete to get apple from a tree, 1 at a time, the bear always wins by a large margin:

- bear: 1 apple per 15 seconds --> 240 apples per hour
- sloth: 1 apple per hour

But, when we have a lot of sloths:

- 5888 sloths: 5888 apples per hour

Sloths win.

---

Building tree:

Up to 30x faster than std::unordered_map for inserting same number of random elements.

Finding element (under development):

Up to 8x faster than std::unortered_map for finding multiple keys and 3000x faster than brute-force scanning array. 

Todo: optimize leaf nodes with sorting & binary-search.


Sample benchmark code:
```C++
#include"tree.cuh"
#include<random>
int main()
{
    // searching nSearch items in an array of n elements
    int n = 10000000;
    const int nSearch = 10000000;

    Sloth::Tree<int, int> tree;
    std::unordered_map<int, int> map;
    std::vector<int> key(n);
    std::vector<int> value(n);
    for (int i = 0; i < n; i++)
    {
        key[i] = i;
        value[i] = i;
    }
    unsigned int seed = 0;

    // unique random numbers generated
    std::shuffle(key.begin(), key.end(), std::default_random_engine(seed));

    std::cout << " =========================== Benchmarking Initialization ============================ " << std::endl;
    // build tests
    for (int i = 0; i < 5; i++)
    {
        size_t t;
        std::cout << "------------------------------------------------------------------------------------" << std::endl;
        int valueFound = -1;
        bool found = false;
        for (int i = 0; i < 100; i++)
        {
            {
                Sloth::Bench bench(&t);
                tree.Build(key, value);
            }
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
                std::cout << " found: " << (found ? "yes" : "no") << std::endl;
                std::cout << " found value: " << valueFound << " real value: " << value[15] << std::endl;
                std::cout << "error: tree traversal could not find the key that was inserted." << std::endl;
                return 0;
            }
            if(i%10 == 0)
                std::cout << "build gpu: " << t / 1000000000.0 << "s" << std::endl;
        }
        std::cout << " found: " << (found ? "yes" : "no") << std::endl;
        std::cout << " found value: " << valueFound << " real value: " << value[15] << std::endl;
        map.clear();
        
        {
            Sloth::Bench bench(&t);
            for (int j = 0; j < n; j++)
                map[key[j]] = value[j];
        }
        std::cout << "build std::unordered_map: " << t / 1000000000.0 << "s" << std::endl;

    }


    std::cout << " =========================== Benchmarking Search Operation ============================ " << std::endl;


    // search tests
    // nSearch consumes too much video-memory. beware.
    // it allocates enough queue-space for nSearch CUDA threads
    // if 1 million keys are to be searched, do it in 100 steps using 10k chunks

    std::vector<int> keys(nSearch);
    std::vector<int> values(nSearch);
    std::vector<char> conditions(nSearch);
    std::vector<int> valuesBruteForce(nSearch);
    std::vector<char> conditionsBruteForce(nSearch);
    for (int i = 0; i < nSearch; i++)
    {        
        keys[i] = i;
    }
    // searching doesn't require unique keys but same code was reused anyway.
    std::shuffle(keys.begin(), keys.end(), std::default_random_engine(seed));

    size_t t;
    for (int i = 0; i < 5; i++)
    {
        {
            Sloth::Bench bench(&t);
            for (int j = 0; j < nSearch; j++)
            {
                bool cond = false;
                int val = -1;
                auto it = map.find(keys[j]);
                if (it != map.end())
                {
                    val = it->second;
                    cond = true;
                }
                conditionsBruteForce[j] = cond;
                valuesBruteForce[j] = val;
            }

        }
        std::cout << "find std::unordered_map: " << t / 1000000000.0 << "s" << std::endl;

        for (int k = 0; k < 100; k++)
        {
            {
                Sloth::Bench bench(&t);
                tree.FindKeyGpu(keys, values, conditions);
            }
            if(k%10==0)
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
                std::cout << "tree condition: " << (int)conditions[i] << " brute-force condition: " << (int)conditionsBruteForce[i] << std::endl;
                return 0;
            }
        }
    }

    return 0;
}


```

Output:
```
 =========================== Benchmarking Initialization ============================
------------------------------------------------------------------------------------
build gpu: 0.0416458s
build gpu: 0.0302957s
build gpu: 0.0302053s
build gpu: 0.0301851s
build gpu: 0.0301825s
build gpu: 0.0301958s
build gpu: 0.0301058s
build gpu: 0.0302575s
build gpu: 0.0302568s
build gpu: 0.0301617s
 found: yes
 found value: 15 real value: 15
build std::unordered_map: 2.35114s
------------------------------------------------------------------------------------
build gpu: 0.0312252s
build gpu: 0.0302512s
build gpu: 0.0303013s
build gpu: 0.0302175s
build gpu: 0.0303988s
build gpu: 0.0302662s
build gpu: 0.0303072s
build gpu: 0.0302863s
build gpu: 0.0302872s
build gpu: 0.0304177s
 found: yes
 found value: 15 real value: 15
build std::unordered_map: 1.11404s
 =========================== Benchmarking Search Operation ============================
find std::unordered_map: 0.205048s
simple find gpu: 0.06739s
simple find gpu: 0.025621s
simple find gpu: 0.0256685s
simple find gpu: 0.0255597s
simple find gpu: 0.0260123s
simple find gpu: 0.0255821s
simple find gpu: 0.0256576s
simple find gpu: 0.0255937s
simple find gpu: 0.0256322s
simple find gpu: 0.0257908s
find std::unordered_map: 0.206452s
simple find gpu: 0.0273387s
simple find gpu: 0.0255418s
simple find gpu: 0.0256064s
simple find gpu: 0.025808s
simple find gpu: 0.0256079s
simple find gpu: 0.025541s
simple find gpu: 0.0255673s
simple find gpu: 0.0259154s
simple find gpu: 0.0255716s
simple find gpu: 0.0255605s
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

![Top-down approach](https://github.com/tugrul512bit/SlothTree/blob/master/sloth-tree.drawio.png)

# Separating Chunks

- Allocating chunks requires counting the number of elements. This is computed with parallel reduction in shared memory and warp shuffle. From 128 elements to 32 elements, shared memory is used. From 32 elements to 1 element, warp shuffle is used.
- Moving millions of elements to their target with only atomics-based offset computation is slow. Due to this, multiple elements are extracted from 128-sized regions by parallel stream-compaction.
- This is repeated for each child node chunk

![counting & compacting](https://github.com/tugrul512bit/SlothTree/blob/master/separating-chunks.drawio.png)
