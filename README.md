# Backpropagation for Basic Operations
<h4>A simple implementation which can be adopted for gradient calculation in systems which involve a chain of computations that produce a complicated differentiable function (as in neural networks)</h4>

The differentiable function is obtained by performing a chain of computations on a special object type which tracks the sequence of operations by means of a <b>Dynamic Computation Graph</b> and is used to compute the gradients of the output of the function with respect to each of its parameters (parameter referes to objects that take part in the computation).

The gradients are calculated when the <b>backward</b> method is called on the object. The <b>backpropagation</b> is carried out by traversing backwards through the <b>Dynamic Compuation Graph</b> and the calculated gradients are stored in the objects and can be retrieved from the <b>grad</b> attribute once backward pass is done. 
