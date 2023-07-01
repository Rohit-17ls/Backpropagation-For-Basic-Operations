#include <iostream>
#include "./Backpropagation.h"


int main(){

	Item p(2.0, true);
	Item q(3.0, true);
	Item& r = p + q;
	Item& s = p*r - q;


	s.backward();

	std::cout << "∂s/∂s : " << s.grad << std::endl;
	std::cout << "∂s/∂r : " << r.grad << std::endl;
	std::cout << "∂s/∂q : " << q.grad << std::endl;
	std::cout << "∂s/∂p : " << p.grad << std::endl;
	std::cout << std::endl;

	s.reset_grads(0);

	r.backward();

	std::cout << "∂s/∂r : " << r.grad << std::endl;
	std::cout << "∂s/∂q : " << q.grad << std::endl;
	std::cout << "∂s/∂p : " << p.grad << std::endl;

	r.reset_grads(0);


	Item& t = p/s - r;
	Item& u = t + p;
	Item& v = p*u + r;

	v.backward();

	std::cout << "∂v/∂v : " << v.grad << std::endl;
	std::cout << "∂v/∂u : " << u.grad << std::endl;
	std::cout << "∂v/∂t : " << t.grad << std::endl;
	std::cout << "∂v/∂s : " << s.grad << std::endl;
	std::cout << "∂v/∂r : " << r.grad << std::endl;
	std::cout << "∂v/∂q : " << q.grad << std::endl;
	std::cout << "∂v/∂p : " << p.grad << std::endl;

	return 0;
}
