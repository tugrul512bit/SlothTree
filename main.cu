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
