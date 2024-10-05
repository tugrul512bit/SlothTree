#include"tree.cuh"
int main()
{

        int n = 10000000;
        Sloth::Tree<int, int> tree;
        std::vector<int> key(n);
        std::vector<int> value(n);
        for (int i = 0; i < n; i++)
        {
            key[i] = rand();//*rand()+rand();
            value[i] = i;
        }
        for (int i = 0; i < 10; i++)
            tree.Build(key, value, true);
    
    return 0;
}
