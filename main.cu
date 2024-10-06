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

    for (int i = 0; i < 10; i++)
    {
        std::cout << "------------------------------------------------------------------------------------" << std::endl;
        int valueFound = -1;
        bool found = false;
        for (int i = 0; i < 100; i++)
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
    size_t t;
    {
        Sloth::Bench bench(&t);
        for (int i = 0; i < n; i++)
        {
            if ((i != 15) && (key[i] == key[15]))
            {
                std::cout << "duplicate key(" << key[15] << ") found with value: " << value[i] << std::endl;
            }
        }
    }
    std::cout << "brute force find cpu: " << t / 1000000000.0 << "s" << std::endl;
    return 0;
}
