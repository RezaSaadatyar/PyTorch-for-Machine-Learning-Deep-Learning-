**PyTorch for Machin Learning & Deep Learning**

**PyTorch_Fundamentals:**<br/>
**1️⃣ CPU (Central Processing Unit) & GPU (Graphics Processing Unit):**
- `CPU`
  - Designed for general-purpose computing.
  - Optimized for sequential tasks.
  - Has a few powerful cores.
  - Excellent at handling complex logic and single-threaded applications.
- `GPU`
  - Designed for parallel processing.
  - Has thousands of smaller, less powerful cores.
  - GPUs offer far faster numerical computing than CPUs.
  - Optimized for tasks that can be divided into many independent calculations.
  - Excellent for tasks like matrix operations, which are common in deep learning.

Putting a tensor on GPU using `to(device)` (e.g. `some_tensor.to(device)`) returns a copy of that tensor, e.g. the same tensor will be on CPU and GPU. `some_tensor = some_tensor.to(device)`  

**2️⃣ N-d Tensor:** A tensor is a multi-dimensional array of numerical values. Tensor computation (like numpy) with strong GPU acceleration.
- `0-dimensional (Scalar):` A single number, e.g., 5, 3.14, -10. A <font color='red'><b>scalar</b></font> is a single number and in tensor-speak it's a zero dimension tensor.
- `1-dimensional (Vector):` A list of numbers, e.g., [1, 2, 3]. A <font color='blue'><b>vector</b></font> is a single dimension tensor but can contain many numbers.<br/>
- `2-dimensional (Matrix):` A table of numbers, e.g., [[1, 2], [3, 4]]. <font color='green'><b>MATRIX</b></font>  has two dimensions.
- `3-dimensional (or higher):` Like a "cube" of numbers or more complex higher-dimensional structures. These are common for representing images, videos, and more.

**3️⃣ Tensor datatypes:**<br/>
There are many different [tensor datatypes available in PyTorch](https://pytorch.org/docs/stable/tensors.html#data-types). Some are specific for CPU and some are better for GPU.<br/>
Generally if you see `torch.cuda` anywhere, the tensor is being used for GPU (since Nvidia GPUs use a computing toolkit called CUDA).<br/>
The most common type (and generally the default) is `torch.float32` or `torch.float`.<br/>

**4️⃣ Getting information from tensors:**<br/>
* `shape` - what shape is the tensor? (some operations require specific shape rules)
* `dtype` - what datatype are the elements within the tensor stored in?
* `device` - what device is the tensor stored on? (usually GPU or CPU)

**5️⃣ Math Operations:**<br/>
* Addition ⇒ `a+b `or `torh.add(a, b)`
* Substraction ⇒ `a-b `or `torh.sub(a, b)`
* Multiplication (element-wise) ⇒ `a*b `
* Division ⇒ `a/b `or `torh.div(a, b)`
* Matrix multiplication ⇒ "`@`" in Python is the symbol for matrix multiplication. [`torch.matmul()`](https://pytorch.org/docs/stable/generated/torch.matmul.html) or [`torch.mm()`](https://pytorch.org/docs/stable/generated/torch.mm.html)
  
**6️⃣ Special Arrays**<br/>
- zeros
- ones
- empty
- eye
- full<br/>

Using [`torch.zeros_like(input)`](https://pytorch.org/docs/stable/generated/torch.zeros_like.html) or [`torch.ones_like(input)`](https://pytorch.org/docs/1.9.1/generated/torch.ones_like.html) which return a tensor filled with zeros or ones in the same shape as the `input` respectively.

**7️⃣ Random Arrays**
- `torch.rand:` Create a n*m tensor filled with random numbers from a uniform distribution on the interval [0, 1)
- `torch.randn:` Create a n*m tensor filled with random numbers from a normal distribution with mean 0 and variance 1. 
- `torch.randint:` Create a n*m tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

`torch.randperm(value):` Create a random permutation of integers from 0 to value.<br/>
`torch.permute(input, dims):` Permute the original tensor to rearrange the axis order.

**8️⃣ Indexing & Slicing**
- `Indexing`
  - Accessing individual elements:  use integer indices to specify the position of the element you want to retrieve.
- `Slicing`
  - Extracting sub-tensors: Slicing allows you to extract a sub-part of your tensor by specifying a range of indices using the colon : operator.
    - `start:end` (exclusive end)
    - `start:` (from start to end of dimension)
    - `:end` (from beginning to end of dimension)
    - `:` (all elements)
    - `start:end:step` (start to end with given step)
  - Slicing with steps: You can include a step to skip elements in the slice. `start:end:step`

**9️⃣ `Unsqueeze & unsqueeze:`**
- The squeeze() method removes all singleton dimensions from a tensor. It will reduce the number of dimensions by removing the ones that have a size of 1.
- The unsqueeze() method adds a singleton dimension at a specified position in a tensor. It will increase the number of dimensions by adding a size of 1 dimension at a specific position.

**🔟 `PyTorch tensors & NumPy:`**
- [`torch.from_numpy(ndarray)`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html)  NumPy array → PyTorch tensor
- [`torch.Tensor.numpy()`](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)  PyTorch tensor → NumPy array.
----