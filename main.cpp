#include "bof.h"


int main()
{
    BuildDictionary(12,6);
    TrainingDataInit();

    TestClassify();

    return 0;
}
