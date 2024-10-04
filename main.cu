#include"tree.cuh"
int main()
{
    constexpr int n = 1024;
    Sloth::Tree<int, int> tree;
    std::vector<int> key(n);
    std::vector<int> value(n);
    for (int i = 0; i < n; i++)
    {
        key[i] = rand();
        value[i] = i;
    }
    for(int i=0;i<100;i++)
        tree.Build(key, value,true);

    return 0;
}
