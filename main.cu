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
            key[i] = rand();//*rand()+rand();
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
    return 0;
}
