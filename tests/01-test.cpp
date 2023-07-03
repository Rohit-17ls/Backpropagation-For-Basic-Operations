#include <iostream>
#include "../Backpropagation.h"

int main(){



        Item a(-2.0 , true);
        Item b(3.0 , true);

        Item& c = a + b;
        Item& d = a*c - b;
        Item& e = b ^ d;

        e.backward();


        std::cout << "∂e/∂e = " << e.grad << std::endl;
        std::cout << "∂e/∂d = " << d.grad << std::endl;
        std::cout << "∂e/∂c = " << c.grad << std::endl;
        std::cout << "∂e/∂b = " << b.grad << std::endl;
        std::cout << "∂e/∂a = " << a.grad << std::endl;







        return 0;
}
