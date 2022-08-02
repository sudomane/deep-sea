#include <criterion/criterion.h>

int function()
{
    return 1;
}

Test(sample, test)
{
    cr_expect(2 == 2, "Expected 2");
}

int main()
{
    return 0;
}