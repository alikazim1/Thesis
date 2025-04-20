import React from "react";
import { FaBrain, FaCogs, FaCube, FaRocket } from "react-icons/fa";
import "./Fundamentals.css";
import Torch1Png from "./Img/torch1.png";
import PythonicPng from "./Img/Pythonic.png";
import GraphPng from "./Img/Graph.png";
import Torch from "./Img/torch.png";
import FloatingNumsPng from "./Img/floating_nums.png";
import TensorRepresentationPng from "./Img/tensor_representation.png";
import TensorPng from "./Img/Tensor.png";
import TensorStoragePng from "./Img/tensor_storage.png";
import GPUPng from "./Img/gpu.png";
import GPUParrallelizationPng from "./Img/gpu_parallization.png";
import InputOutputPng from "./Img/input_output.png";
import CodeIOBlock from "./CodeIOBlock";
import TrueFalseExercise from "./TrueFalseExercise.jsx";

const Fundamentals = () => {
  return (
    <div className="content">
      <h1 className="page-title">01. PyTorch Fundamentals</h1>
      <img src={Torch1Png} className="centered-image" alt="PyTorch Torch" />
      <section className="prerequisites">
        <h2 id="section1">Prerequisites</h2>
        <p>This guide assumes the following prerequisites:</p>
        <ul>
          <li>Basic/Intermediate Python</li>
          <li>Linear Algebra</li>
          <li>Optional: ML and CMD basics</li>
        </ul>
      </section>
      <section id="section2">
        <h1>Introduction to AI</h1>
        <p>
          Welcome to the comprehensive guide on PyTorch! This will take you from
          the fundamentals to advanced applications.
        </p>
        <p>
          Before diving into{" "}
          <a
            href="https://pytorch.org"
            target="_blank"
            rel="noopener noreferrer"
          >
            PyTorch
          </a>
          , let's explore what Artificial Intelligence (AI) is.
        </p>
        <div className="ai-definition">
          <strong>AI:</strong> AI refers to the simulation of human intelligence
          in machines that are programmed to think and act like humans.
        </div>
      </section>
      <section className="learning-objectives">
        <h2>Learning Objectives</h2>
        <ul>
          <li>Understand the basics of artificial intelligence.</li>
          <li>Learn how to install and set up PyTorch.</li>
          <li>
            Get familiar with essential concepts like Tensors and GPU
            acceleration.
          </li>
        </ul>
      </section>
      <section id="section3">
        <h1>What is PyTorch?</h1>
        <img src={GraphPng} className="deep-learning" alt="PyTorch Graph" />
        <p>
          PyTorch is an open-source machine learning library developed by
          Facebook's AI Research lab.
        </p>
        <ul>
          <li>
            <strong>Python Ecosystem Integration:</strong> Seamlessly integrates
            with other Python libraries like NumPy, SciPy, and scikit-learn.
          </li>
          <li>
            <strong>GPU Acceleration:</strong> Training on GPU enhances speed
            and efficiency for large-scale models.
          </li>
          <li>
            <strong>Research-Friendly:</strong> Simple and dynamic nature,
            making it widely adopted in AI research.
          </li>
          <li>
            <strong>Flexibility:</strong> Easy to use with dynamic computation
            graphs for flexibility in research and deployment.
          </li>
        </ul>
      </section>
      <img src={PythonicPng} className="deep-learning" alt="PyTorch Pythonic" />
      <section id="section4">
        <h1>Where is PyTorch Used?</h1>
        <ul>
          <li>
            <strong>Meta:</strong> Image and video recognition, content
            recognition, and AI features.
          </li>
          <li>
            <strong>Microsoft:</strong> Used in NLP for products like Cortana
            and Office 365.
          </li>
          <li>
            <strong>Amazon:</strong> AWS customers use PyTorch for building and
            scaling machine learning models.
          </li>
          <li>
            <strong>Tesla and Uber:</strong> PyTorch is used for motion planning
            in self-driving technologies and analyzing data from sensors.
          </li>
        </ul>
      </section>
      <img src={Torch} className="deep-learning" alt="PyTorch Torch" />
      <div className="torch">
        <h1 id="section5">Setting up Environment for PyTorch</h1>
        <div className="note">
          <p>
            <strong>Note:</strong> Before running any of the code in this
            notebook, you should have gone through the{" "}
            <a
              href="https://pytorch.org/get-started/locally/"
              target="_blank"
              rel="noopener noreferrer"
            >
              PyTorch setup steps
            </a>
            .
          </p>
          <p>
            <strong>
              If you're running on{" "}
              <a
                href="https://colab.research.google.com/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Google Colab
              </a>
            </strong>
            , everything should work (Google Colab comes with PyTorch and other
            libraries installed).
          </p>
          <p>
            Importing PyTorch and Checking its version If you are running
            PyTorch in your local host, Make sure to install it and other
            related libraries from
            <a href="https://pytorch.org/get-started/locally/">
              {" "}
              <b>here</b>.
            </a>
          </p>
          <div className="ai-definition">
            <strong>Google Colab:</strong> Now onwards we will use google colab
            for our code execution. If you do not have access to any GPU, you
            can use google colab for free. It provides free access to GPU and
            other functionalities.
          </div>

          <CodeIOBlock
            inputCode={`import torch
torch.__version__
      `}
            outputCode={`'2.3.0+cu121'  `}
          />

          <p>
            Amazing!! We got pytorch in our environment and its version is :
            2.3.0+cu121. For you it may be changed depending on the time.
          </p>
        </div>
      </div>
      <h1 id="section6">Tensor</h1>
      <p>
        Floating point numbers are the way a model deals with the information.
        We need a way to encode real-world data of the kind we want to process
        into something digestable by a model and then decode the output back to
        something we can understand and use for our purpose.
      </p>
      <img
        src={FloatingNumsPng}
        className="deep-learning"
        alt="Floating point numbers"
      />
      <p>
        Tensors are specialized data structure that are very similar to arrays
        or matrices. The dimensionality of a tensor coincides with the number of
        indexes used to refer to scaler values with in the tensor. A Tensor is
        fundamental building block in deep learning, just as a digit in
        fundamental building block of mathematics.
      </p>
      <img
        src={TensorRepresentationPng}
        className="deep-learning"
        alt="Tensor representation"
      />
      <p>
        <b>Why Tensor?</b>
        <br />
        <br />
        Python lists are designed for general purpose numerical operations. They
        lack operations like dot product of two vectors, making them inefficient
        for numerical data. In short, multi-dimentional lists are inefficient.
        For this reason data structures like PyTorch Tensors are introduced,
        which provide efficient low-level implementation of numberical data
        structures and related operations on them. Tensors are allocated in
        contagoius chunks of memory managed by torch.Storage instances. This
        contiguity is important for performance reasons especially when
        performing operations.
      </p>
      <img
        src={TensorStoragePng}
        className="deep-learning"
        alt="Tensor storage"
      />
      <p>
        The contiguity can be checked by the function{" "}
        <code>is_contiguous()</code> function.
      </p>
      <CodeIOBlock
        inputCode={`import torch
  a = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
  print(a.is_contiguous()) `}
        outputCode={`True `}
      />
      <b>Non-Contiguous Tensors</b>
      <p>
        Certain operations, such as transposing a tensor or using advanced
        indexing, can result in non-contigous tensors.
      </p>
      <CodeIOBlock
        inputCode={`b = a.T
print(b.is_contiguous())  `}
        outputCode={`
False  `}
      />
      <p>We can make the transposed tensor contiguous</p>
      <CodeIOBlock
        inputCode={`b = b.contiguous()
print(b.is_contiguous())  `}
        outputCode={`
True  `}
      />
      <p>
        In summary, PyTorch generally stores tensors in contiguous chunks of
        memory to optimize performance. However, certain operations can produce
        non-contiguous tensors, which may require conversion to a contiguous
        layout for efficient computation.
      </p>
      <p>
        Here you can see how the float values of tensor are stored in contagous
        memory. Even though the tensor reports itself as having three rows and
        two columns, the storage under the hood is contagious array of size 6.
      </p>
      <CodeIOBlock
        inputCode={`
 points = torch.tensor([[4.0,1.0], [5.0, 3.0], [2.0, 1.0]])
 points.storage()
       `}
        outputCode={` 4.0
 1.0
 5.0
 3.0
 2.0
 1.0
[torch.storage.TypedStorage(dtype=torch.float32, device=cpu) of size 6]  `}
      />
      <p>
        In PyTorch, we use tensors to encode the inputs and outputs of a model,
        as well as the model’s parameters.
      </p>
      <img
        src={InputOutputPng}
        className="deep-learning"
        alt="Tensors in PyTorch"
      />
      <div style={{ textAlign: "center" }}>
        <b>torch.Size([1, 3, 3])</b>
      </div>
      <h1>Initializing a Tensor</h1>
      <p>Tensors can be initialized in serveral ways.</p>
      <b>Directly from the data:</b>
      <CodeIOBlock
        inputCode={`import torch
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
x_data  `}
        outputCode={`tensor([[1, 2],
        [3, 4]])  `}
      />
      <b>From other tensor</b>
      <CodeIOBlock
        inputCode={`x_ones = torch.ones_like(x_data)
print(f"Ones Tensor:\\n{x_ones}\\n")
           
x_rand = torch.rand_like(x_data, dtype = torch.float)
print(f"Random Tensor: \n{x_rand}\n")`}
        outputCode={`Ones Tensor: 
 tensor([[1, 1],
        [1, 1]])

Random Tensor: 
tensor([[0.6515, 0.4408],
        [0.9148, 0.7340]])  `}
      />
      <b>With random or constant values</b>
      <CodeIOBlock
        inputCode={`
shape = (2,3,) # shape is a tuple of tensor dimensions
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \\n{rand_tensor}\\n")
print(f"Ones Tensor: \\n{ones_tensor}\\n")
print(f"Zeros Tensor: \\n{zeros_tensor}\\n")
  `}
        outputCode={`Random Tensor: 
tensor([[0.3686, 0.4562, 0.5344],
        [0.4988, 0.2458, 0.0761]])

Ones Tensor: 
tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor: 
tensor([[0., 0., 0.],
        [0., 0., 0.]])  `}
      />
      <b>Scaler Tensors</b>
      <p>
        PyTorch also recognizes scalers but these are tensor type. To check the
        dimention we use ndim attribute.
      </p>
      <CodeIOBlock
        inputCode={`scaler = torch.tensor(5)
scaler
scaler.ndim  `}
        outputCode={`tensor(5)
0  `}
      />
      <b>Attributes of Tensors</b>
      <p>
        The attributes of tensor describe its shape, datatype, and the device on
        which it is stored.
      </p>
      <CodeIOBlock
        inputCode={`tensor = torch.rand(5,6)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")  `}
        outputCode={`Shape of tensor: torch.Size([5, 6])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu  `}
      />
      <b>Run Time</b>
      <p>
        We can improve our runtime(40x-60x) by using GPU provided by Google
        colab. go to Runtime -) Change runtime type -) GPU PyTorch provides over
        100 tensor operations.
      </p>
      <b>GPU vs CPU</b>
      <p />
      GPU is a specialized processor designed to accelerate graphics rendering
      and parallel processing tasks. It has a large number of cores that can
      operate simultaneously, making it well-suited for tasks like deep
      learning. CPU is a general-purpose processor that handles a wide range of
      tasks, but it has fewer cores and is not as efficient for parallel
      processing as a GPU.
      <p />
      <b>GPU Terminator </b>{" "}
      <p>
        {" "}
        We will see latter that tensors have lots of operations. In training of
        a neural network millions of operations on tensors are required, which
        is more difficult for a single CPU to carryout. That's when GPU comes
        into action. GPU's are best due to their ability to handle massive
        computational requirements(parallel processing) for training of
        networks.
      </p>
      <img src={GPUPng} className="deep-learning" alt="GPU" />
      <CodeIOBlock
        inputCode={`device = 'gpu' if torch.cuda.is_available() else 'cpu'
print(device)  `}
        outputCode={`gpu  `}
      />
      <p>
        If device is 'gpu' we are good to go and if it is 'cpu' then go Runtime
        -) Change runtime type -) GPU and then run the code.
      </p>
      <b>Tensor Transfer</b>
      <p>We can move tensors to the gpu if it is available and vice versa.</p>
      <CodeIOBlock
        inputCode={`if torch.cuda.is_available():
  tensor = tensor.to('cuda')

print(tensor.device)  `}
        outputCode={`cuda:0  `}
      />
      <p>
        In contrast, tensors can represent anything in the universe. Once these
        tensors are embedded into the machine, we can manipulate them.
      </p>
      <h1 id="section7">Methods in Tensor</h1>
      <p>
        <b>randint():</b> The randint() method returns a tensor filled with
        random integers generated uniformly between low (inclusive) and high
        (exclusive) for a given shape. The shape is given by the user which can
        be a tuple or a list with non-negative members. The default value for
        low is 0. When only one int argument is passed, low gets the value 0, by
        default, and high gets the passed value. Like zeros() an empty tuple or
        list for the shape creates a tensor with zero dimension.
      </p>
      <CodeIOBlock
        inputCode={`import torch

#Syntax: torch.randint(,,) and the printed values might be different for you
randint_tensor = torch.randint(5, (3,3))
print(randint_tensor)
       `}
        outputCode={`tensor([[1, 1, 1],
        [2, 2, 4],
        [1, 4, 2]])  `}
      />
      <p>
        <b>complex():</b> The complex() method creates a complex tensor using
        real and imaginary parts provided as tensors. Both must have the same
        shape and data type.
      </p>
      <CodeIOBlock
        inputCode={`import torch

#Syntax: torch.complex(real, imag)
real = torch.rand(2, 2)
print(real)
imag = torch.rand(2, 2)
print(imag)
complex_tensor = torch.complex(real, imag)
print(complex_tensor)
  `}
        outputCode={`tensor([[0.4033, 0.7990],
        [0.2123, 0.6589]])
tensor([[0.9815, 0.1214],
        [0.6732, 0.3657]])
tensor([[0.4033+0.9815j, 0.7990+0.1214j],
        [0.2123+0.6732j, 0.6589+0.3657j]])`}
      />
      <p>
        <b>eye():</b> The eye() method returns a 2-D identity matrix with ones
        on the diagonal and zeros elsewhere. You can specify rows and columns.
      </p>
      <CodeIOBlock
        inputCode={`import torch

#Syntax: torch.eye(n, m)
n = m = 3
eye_tensor = torch.eye(n, m)
print(eye_tensor)
  `}
        outputCode={`tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])`}
      />
      <p>
        <b>zeros():</b> The zeros() method creates a tensor filled with zeros.
        Shape can be given as tuple, list or separate arguments.
      </p>
      <CodeIOBlock
        inputCode={`import torch

zeros_tensor = torch.zeros(3, 2)
print(zeros_tensor)
  `}
        outputCode={`tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])`}
      />
      <p>
        <b>rand():</b> The rand() method generates a tensor filled with random
        floats from a uniform distribution between 0 and 1.
      </p>
      <CodeIOBlock
        inputCode={`import torch

#Syntax: torch.rand()
rand_tensor = torch.rand(3, 3)
print(rand_tensor)
  `}
        outputCode={`tensor([[0.7786, 0.2453, 0.7261],
        [0.3529, 0.9573, 0.5311],
        [0.3924, 0.6010, 0.6432]])`}
      />
      <p>
        <b>ones():</b> The ones() method creates a tensor of specified shape
        where all elements are 1. Shape can be passed as tuple.
      </p>
      <CodeIOBlock
        inputCode={`import torch

#Syntax: torch.ones(shape)
ones_tensor = torch.ones((4, 4, 4))
print(ones_tensor)
  `}
        outputCode={`tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],
        ...
        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])`}
      />
      <p>
        <b>arange():</b> The arange() method returns a 1-D tensor with values
        starting from start (inclusive) to end (exclusive) with a step size.
      </p>
      <CodeIOBlock
        inputCode={`import torch

#Syntax: torch.arange(start, end, step)
arange_tensor = torch.arange(2, 20, 2)
print(arange_tensor)
  `}
        outputCode={`tensor([ 2,  4,  6,  8, 10, 12, 14, 16, 18])`}
      />
      <p>
        <b>full():</b> The full() method returns a tensor of given shape where
        all values are set to a specified constant.
      </p>
      <CodeIOBlock
        inputCode={`import torch

#Syntax: torch.full(shape, fill_value)
full_tensor = torch.full((3, 2), 3)
print(full_tensor)
  `}
        outputCode={`tensor([[3, 3],
        [3, 3],
        [3, 3]])`}
      />
      <p>
        <b>linspace():</b> The linspace() method returns a 1-D tensor with a
        specified number of evenly spaced values between start and end
        (inclusive).
      </p>
      <CodeIOBlock
        inputCode={`import torch

#Syntax: torch.linspace(start, end, steps)
linspace_tensor = torch.linspace(1, 7.75, 4)
print(linspace_tensor)
  `}
        outputCode={`tensor([1.0000, 3.5833, 6.1667, 7.7500])`}
      />
      <h1 id="section8">Tensor Operations</h1>
      <p>
        Every type of data(image, video,text,audio,milky-way galaxy etc) in
        PyTorch is represented by tensors. The deep learning model gains
        knowledge by examining those tensors and executing numerous
        operations(potentially millions or more) on them constructing a
        depiction of the patterns within the input data.
      </p>
      <p>
        <b>Addition:</b> You can add a number to a tensor element-wise using the
        `+` operator. The original tensor remains unchanged unless reassigned.
      </p>
      <CodeIOBlock
        inputCode={`import torch
tensor = torch.tensor([1,2,3])
tensor + 10`}
        outputCode={`tensor([11, 12, 13])`}
      />
      <p>
        <b>Multiplication:</b> You can multiply tensor elements with a scalar
        using the `*` operator. Like addition, the original tensor doesn't
        change unless reassigned.
      </p>
      <CodeIOBlock
        inputCode={`tensor = torch.tensor([1,2,3])
tensor * 10`}
        outputCode={`tensor([10, 20, 30])`}
      />
      <p>
        <b>Reassignment after subtraction:</b> Arithmetic operations require
        reassignment to update the tensor values. Here we subtract 10 and assign
        back.
      </p>
      <CodeIOBlock
        inputCode={`tensor = tensor - 10
tensor`}
        outputCode={`tensor([-9, -8, -7])`}
      />
      <blockquote>
        PyTorch also has built-in functions for multiplication and addition like{" "}
        <code>torch.mul()</code> and <code>torch.add()</code> to perform basic
        operations.
      </blockquote>
      <p>
        <b>Matrix Multiplication (@):</b> Matrix multiplication requires inner
        dimensions to match. Use <code>@</code> or <code>matmul()</code> for
        this operation.
      </p>
      <CodeIOBlock
        inputCode={`# Element-wise multiplication
a = torch.tensor([1,2,3])
b = a * a
print(b)

# Matrix multiplication
c = a.matmul(a)
print(c)

# Matrix multiplication using @
d = a @ a
print(d)`}
        outputCode={`tensor([1, 4, 9])
tensor(14)
tensor(14)`}
      />
      <p>
        <b>Dimension mismatch error:</b> Matrix multiplication will fail if the
        inner dimensions do not match. In such cases, you may need to transpose
        a matrix.
      </p>
      <CodeIOBlock
        inputCode={`tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]], dtype=torch.float32)

torch.matmul(tensor_A, tensor_B) # will error`}
        outputCode={`RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)`}
      />
      <p>
        <b>Fix with transpose:</b> Use <code>.T</code> to transpose one of the
        matrices so the inner dimensions match, allowing matrix multiplication
        to proceed.
      </p>
      <CodeIOBlock
        inputCode={`torch.matmul(tensor_A, tensor_B.T)`}
        outputCode={`tensor([[ 27.,  30.,  33.],
        [ 61.,  68.,  75.],
        [ 95., 106., 117.]])`}
      />
      <blockquote>
        You can visualize matrix multiplication here:{" "}
        <a href="http://matrixmultiplication.xyz/">
          http://matrixmultiplication.xyz/
        </a>
      </blockquote>
      <p>
        <b>Tensor Indexing:</b> PyTorch tensors support powerful indexing, such
        as slicing rows/columns and adding new dimensions using{" "}
        <code>[None]</code>.
      </p>
      <CodeIOBlock
        inputCode={`points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points`}
        outputCode={`tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])`}
      />
      <CodeIOBlock
        inputCode={`print(points[1:]) # All rows after the first`}
        outputCode={`tensor([[5., 3.],
        [2., 1.]])`}
      />
      <CodeIOBlock
        inputCode={`print(points[1:, :]) # All rows after the first, all columns`}
        outputCode={`tensor([[5., 3.],
        [2., 1.]])`}
      />
      <CodeIOBlock
        inputCode={`print(points[1:, 0]) # All rows after the first, first column`}
        outputCode={`tensor([5., 2.])`}
      />
      <CodeIOBlock
        inputCode={`print(points[None]) # Adds a dimension like unsqueeze`}
        outputCode={`tensor([[[4., 1.],
         [5., 3.],
         [2., 1.]]])`}
      />
      <h1>Finding min, max, sum etc</h1>
      <p>
        <b>Aggregation Functions:</b> PyTorch provides simple methods to
        summarize tensors using operations like <code>min()</code>,{" "}
        <code>max()</code>, <code>mean()</code>, and <code>sum()</code>.
      </p>
      <CodeIOBlock
        inputCode={`import torch
x = torch.arange(0, 100, 10)  # arange(start, end, step)
x`}
        outputCode={`tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])`}
      />
      <p>Performing basic aggregations:</p>
      <CodeIOBlock
        inputCode={`print(f"Minimum : {x.min()}")
print(f"Maximum : {x.max()}")
print(f"Mean : {x.type(torch.float32).mean()}")
print(f"Sum : {x.sum()}")`}
        outputCode={`Minimum : 0
Maximum : 90
Mean : 45.0
Sum : 450`}
      />
      <p>We can also use PyTorch's built-in functions directly:</p>
      <CodeIOBlock
        inputCode={`torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)`}
        outputCode={`(tensor(90), tensor(0), tensor(45.), tensor(450))`}
      />
      <p>
        <b>Positional Min/Max:</b> To get the index where the min or max value
        occurs, use <code>argmin()</code> and <code>argmax()</code>.
      </p>
      <CodeIOBlock
        inputCode={`tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")`}
        outputCode={`Tensor: tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
Index where max value occurs: 8
Index where min value occurs: 0`}
      />
      <p>
        <b>Tensor Data Types:</b> Tensors can have different data types, which
        impact performance and precision.
      </p>
      <CodeIOBlock
        inputCode={`tensor = torch.arange(10.0, 100.0, 10.0)
tensor.dtype`}
        outputCode={`torch.float32`}
      />
      <blockquote>
        The default data type for tensors is <code>float32</code>. Using{" "}
        <code>float16</code> can boost performance on GPU but reduces precision.
        Use <code>float64</code> if you need higher precision.
      </blockquote>
      <h1 id="section9">Playing More</h1>
      <p>
        <b>🧩 Tensor Dimension Manipulation:</b> Often we need to change the
        shape of a tensor <i>without modifying its values</i>. This is
        especially important in deep learning, where{" "}
        <b>matrix multiplication</b> requires shape alignment.
      </p>
      <blockquote>
        <b>💡 Why?</b> Because shape mismatches will lead to runtime errors in
        operations like matrix multiplication. Shape adjustment ensures that the
        correct elements align correctly across tensors.
      </blockquote>
      <CodeIOBlock
        inputCode={`import torch
x = torch.arange(1., 8.)
x, x.shape`}
        outputCode={`(tensor([1., 2., 3., 4., 5., 6., 7.]), torch.Size([7]))`}
      />
      <h3>📐 Reshape a Tensor</h3>
      <p>
        Use <code>torch.reshape()</code> to change dimensions.
      </p>
      <CodeIOBlock
        inputCode={`x_reshaped = x.reshape(1, 7)
x_reshaped, x_reshaped.shape`}
        outputCode={`(tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))`}
      />
      <h3>🔍 View vs Reshape</h3>
      <p>
        <code>torch.view()</code> returns a new view of the same tensor (shared
        memory). Modifying the view also modifies the original tensor.
      </p>
      <CodeIOBlock
        inputCode={`z = x.view(7, 1)
z, z.shape`}
        outputCode={`(tensor([[1.],
         [2.],
         [3.],
         [4.],
         [5.],
         [6.],
         [7.]]), torch.Size([7, 1]))`}
      />
      <p>
        Now if we change <code>z</code>, <code>x</code> changes too:
      </p>
      <CodeIOBlock
        inputCode={`z[:, 0] = 5
z, x`}
        outputCode={`(tensor([[5.],
         [5.],
         [5.],
         [5.],
         [5.],
         [5.],
         [5.]]),
tensor([5., 5., 5., 5., 5., 5., 5.]))`}
      />
      <blockquote>
        🔗{" "}
        <a
          href="https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch/54507446#54507446"
          target="_blank"
        >
          view vs reshape explained in detail
        </a>
      </blockquote>
      <h3>📚 Stacking Tensors</h3>
      <p>
        We can stack the same tensor multiple times using{" "}
        <code>torch.stack()</code>:
      </p>
      <CodeIOBlock
        inputCode={`x_stacked = torch.stack([x, x, x, x], dim=0)
x_stacked`}
        outputCode={`tensor([[5., 5., 5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5., 5., 5.]])`}
      />
      <h3>➖ Remove Dimensions</h3>
      <p>
        Use <code>torch.squeeze()</code> to remove a dimension of size 1:
      </p>
      <CodeIOBlock
        inputCode={`print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

x_squeezed = x_reshaped.squeeze()
print(f"\\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")`}
        outputCode={`Previous tensor: tensor([[5., 5., 5., 5., 5., 5., 5.]])
Previous shape: torch.Size([1, 7])

New tensor: tensor([5., 5., 5., 5., 5., 5., 5.])
New shape: torch.Size([7])`}
      />
      <h3>➕ Add Dimensions</h3>
      <p>
        You can add a dimension of size 1 using <code>torch.unsqueeze()</code>.
      </p>
      <blockquote>
        ✍️ <b>Homework:</b> Try out <code>torch.unsqueeze()</code> and{" "}
        <code>torch.permute()</code> for reordering tensor dimensions!
      </blockquote>
      <script src="https://gist.github.com/alikazim1/86d419c3b04f69d414e136dbb6ee2928.js"></script>
      <h1>Indexing of Tensors</h1>
      <h2>🔍 Indexing Tensors</h2>
      <p>
        Indexing in PyTorch is very similar to <code>NumPy</code> and Python
        lists — but with higher dimensions. It's crucial when you want to
        extract specific values or slices from a tensor. Interviews often test
        this!
      </p>
      <blockquote>
        💡 Use <code>:</code> to mean “all values” in a dimension. Combine
        multiple dimensions using commas.
      </blockquote>
      <h3>Examples</h3>
      <CodeIOBlock
        inputCode={`# Shape: (1, 3, 3)
x = torch.tensor([[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]])`}
        outputCode={`torch.Size([1, 3, 3])`}
      />
      <ul>
        <li>
          <b>All rows, 1st column (0th index of 1st dim):</b>
        </li>
      </ul>
      <CodeIOBlock inputCode={`x[:, 0]`} outputCode={`tensor([[1, 2, 3]])`} />
      <ul>
        <li>
          <b>All rows & columns, 2nd column of innermost dimension:</b>
        </li>
      </ul>
      <CodeIOBlock
        inputCode={`x[:, :, 1]`}
        outputCode={`tensor([[2, 5, 8]])`}
      />
      <ul>
        <li>
          <b>Get value at [0, 1, 1]:</b>
        </li>
      </ul>
      <CodeIOBlock inputCode={`x[:, 1, 1]`} outputCode={`tensor([5])`} />
      <ul>
        <li>
          <b>All innermost values of [0, 0]:</b>
        </li>
      </ul>
      <CodeIOBlock inputCode={`x[0, 0, :]`} outputCode={`tensor([1, 2, 3])`} />
      <blockquote>
        🧠 <b>Tip:</b> Always match dimensions in the order{" "}
        <code>[batch, row, column]</code> or <code>[dim0, dim1, dim2]</code> for
        3D tensors.
      </blockquote>
      <h1 id="section10">Interoperability with NumPy</h1>
      <h2>🔄 Interoperability with NumPy</h2>
      <p>
        PyTorch tensors can be easily converted to and from <b>NumPy arrays</b>,
        enabling us to leverage NumPy's vast array of functions. This seamless
        conversion helps combine the power of both libraries efficiently.
      </p>
      <h3>Examples</h3>
      <CodeIOBlock
        inputCode={`import torch
import numpy as np

# Create a tensor
points = torch.ones(3, 4)

# Convert tensor to NumPy array
points_np = points.numpy()`}
        outputCode={`array([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], dtype=float32)`}
      />
      <ul>
        <li>
          <b>Modifying the NumPy array also modifies the tensor:</b>
        </li>
      </ul>
      <CodeIOBlock
        inputCode={`# Modify NumPy array
points_np[0] = 5

# Check changes in both NumPy array and tensor
points_np, points`}
        outputCode={`(array([[5., 5., 5., 5.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], dtype=float32),
 tensor([[5., 5., 5., 5.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]])`}
      />
      <blockquote>
        🔧 <b>Tip:</b> NumPy arrays default to <code>float64</code>, but PyTorch
        prefers <code>float32</code>. To ensure compatibility, convert your
        NumPy arrays to PyTorch tensors with the appropriate type:
      </blockquote>
      <CodeIOBlock
        inputCode={`# Convert NumPy array (float64) to PyTorch tensor (float32)
tensor = torch.from_numpy(points_np).type(torch.float32)`}
        outputCode={`tensor([[5., 5., 5., 5.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])`}
      />
      <p>
        Now you can work with the tensor in the required format without losing
        performance or precision.
      </p>
      {/* Render the TrueFalseExercise component */}
      <TrueFalseExercise />
    </div>
  );
};

export default Fundamentals;
