import React from "react";
import { FaBrain, FaCogs, FaCube, FaRocket } from "react-icons/fa";
import ClassificatoinImage from "./Img/classification.png";
import Visualize from "./Img/visual.png";
import Sigmoid from "./Img/sigmoid.png";
import "./neuralNetworkClassification.css";
import Relu from "./Img/relu.png";
import Test from "./Img/test.png";
import Chicken from "./Img/chicken.png";
import Blobs from "./Img/blobs.png";
import ReluFunction from "./Img/ReluFunction.png";
import Plot from "./Img/plot.png";
import inputOutput from "./Img/inputoutput.png";
import Result from "./Img/result.png";
import CodeIOBlock from "./CodeIOBlock.jsx";
import modelResults from "./Img/modelResult.png";
import Workflow from "./Img/pytorchWorkflow.png";
import Circle from "./Img/circle.png";
import Graph from "./Img/graph.png";
import Playground from "./Img/playground.png";
import Traintest from "./Img/traintest.png";

const neuralNetworksClassification = () => {
  return (
    <div className="content">
      <h1 className="page-title">03. Neural Network Classification</h1>
      <section>
        <h2>What is Classification problem?</h2>
        <p>
          A classification problem involves predicting whether something belongs
          to one category or another.
        </p>

        <p>For example, you might want to:</p>
        <table>
          <thead>
            <tr>
              <th>Problem type</th>
              <th>What is it?</th>
              <th>Example</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Binary classification</td>
              <td>Target can be one of two options, e.g. yes or no</td>
              <td>
                Predict whether or not someone has heart disease based on their
                health parameters.
              </td>
            </tr>
            <tr>
              <td>Multi-class classification</td>
              <td>Target can be one of more than two options</td>
              <td>Decide whether a photo is of food, a person or a dog.</td>
            </tr>
            <tr>
              <td>Multi-label classification</td>
              <td>Target can be assigned more than one option</td>
              <td>
                Predict what categories should be assigned to a Wikipedia
                article (e.g. mathematics, science & philosophy).
              </td>
            </tr>
          </tbody>
        </table>

        <p>
          Classification, along with regression, is one of the most common
          machine learning problems. In this notebook, we'll explore
          classification with PyTorch, where we predict the class that a set of
          inputs belongs to.
        </p>

        <h2>What we are going to cover</h2>
        <p>
          In this notebook, we'll revisit the PyTorch workflow covered in
          Notebook 01.
        </p>

        <img src={Workflow} className="centered-image" />

        <p>
          This time, instead of predicting a straight line (a regression
          problem), we'll be tackling a classification problem.
        </p>

        <p>Specifically, we're going to cover:</p>
        <table>
          <thead>
            <tr>
              <th>Topic</th>
              <th>Contents</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0. Architecture of a classification neural network</td>
              <td>
                Neural networks can come in almost any shape or size, but they
                typically follow a similar floor plan.
              </td>
            </tr>
            <tr>
              <td>1. Getting binary classification data ready</td>
              <td>
                Data can be almost anything but to get started we're going to
                create a simple binary classification dataset.
              </td>
            </tr>
            <tr>
              <td>2. Building a PyTorch classification model</td>
              <td>
                Here we'll create a model to learn patterns in the data, we'll
                also choose a loss function, optimizer and build a training loop
                specific to classification.
              </td>
            </tr>
            <tr>
              <td>3. Fitting the model to data (training)</td>
              <td>
                We've got data and a model, now let's let the model (try to)
                find patterns in the (training) data.
              </td>
            </tr>
            <tr>
              <td>4. Making predictions and evaluating a model (inference)</td>
              <td>
                Our model's found patterns in the data, let's compare its
                findings to the actual (testing) data.
              </td>
            </tr>
            <tr>
              <td>5. Improving a model (from a model perspective)</td>
              <td>
                We've trained and evaluated a model but it's not working, let's
                try a few things to improve it.
              </td>
            </tr>
            <tr>
              <td>6. Non-linearity</td>
              <td>
                So far our model has only had the ability to model straight
                lines, what about non-linear (non-straight) lines?
              </td>
            </tr>
            <tr>
              <td>7. Replicating non-linear functions</td>
              <td>
                We used non-linear functions to help model non-linear data, but
                what do these look like?
              </td>
            </tr>
            <tr>
              <td>
                8. Putting it all together with multi-class classification
              </td>
              <td>
                Let's put everything we've done so far for binary classification
                together with a multi-class classification problem.
              </td>
            </tr>
          </tbody>
        </table>

        <h2>Where can you get help?</h2>
        <p>
          All the materials for this course are available on{" "}
          <a href="https://github.com/" target="_blank">
            GitHub
          </a>
          . If you encounter any issues, you can ask questions on the GitHub
          Discussions page. Additionally, the{" "}
          <a href="https://dev-discuss.pytorch.org/" target="_blank">
            PyTorch developer forums
          </a>{" "}
          are a valuable resource for all things PyTorch.
        </p>

        <h2>0. Architecture of a classification neural network</h2>

        <p>
          Before we get into writing code, let's look at the general
          architecture of a classification neural network.
        </p>
        <table>
          <thead>
            <tr>
              <th>Hyperparameter</th>
              <th>Binary Classification</th>
              <th>Multiclass Classification</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Input layer shape (in_features)</td>
              <td>
                Same as number of features (e.g. 5 for age, sex, height, weight,
                smoking status in heart disease prediction)
              </td>
              <td>Same as binary classification</td>
            </tr>
            <tr>
              <td>Hidden layer(s)</td>
              <td>Problem specific, minimum = 1, maximum = unlimited</td>
              <td>Same as binary classification</td>
            </tr>
            <tr>
              <td>Neurons per hidden layer</td>
              <td>Problem specific, generally 10 to 512</td>
              <td>Same as binary classification</td>
            </tr>
            <tr>
              <td>Output layer shape (out_features)</td>
              <td>1 (one class or the other)</td>
              <td>1 per class (e.g. 3 for food, person or dog photo)</td>
            </tr>
            <tr>
              <td>Hidden layer activation</td>
              <td>
                Usually ReLU (rectified linear unit) but can be many others
              </td>
              <td>Same as binary classification</td>
            </tr>
            <tr>
              <td>Output activation</td>
              <td>Sigmoid (torch.sigmoid in PyTorch)</td>
              <td>Softmax (torch.softmax in PyTorch)</td>
            </tr>
            <tr>
              <td>Loss function</td>
              <td>Binary crossentropy (torch.nn.BCELoss in PyTorch)</td>
              <td>Cross entropy (torch.nn.CrossEntropyLoss in PyTorch)</td>
            </tr>
            <tr>
              <td>Optimizer</td>
              <td>
                SGD (stochastic gradient descent), Adam (see torch.optim for
                more options)
              </td>
              <td>Same as binary classification</td>
            </tr>
          </tbody>
        </table>

        <p>
          The components of a classification neural network may vary depending
          on the problem, but this setup is a good starting point. We'll explore
          it hands-on throughout this notebook.
        </p>

        <h2>1. Getting binary classification data ready</h2>

        <p>
          Let's start by generating some data. We'll use the{" "}
          <code>make_circles()</code> method from Scikit-Learn to create two
          circles with differently colored dots.
        </p>

        <CodeIOBlock
          inputCode={`from sklearn.datasets import make_circles


# Make 1000 samples 
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values  `}
        />
        <p>Alright, now let's view the first 5 X and y values.</p>

        <CodeIOBlock
          inputCode={`print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")  `}
          outputCode={`First 5 X features:
[[ 0.75424625  0.23148074]
 [-0.75615888  0.15325888]
 [-0.81539193  0.17328203]
 [-0.39373073  0.69288277]
 [ 0.44220765 -0.89672343]]

First 5 y labels:
[1 1 1 1 0]  `}
        />
        <p>
          It seems like there are two X values for each Y value. Let's follow
          the data explorer's motto: "visualize, visualize, visualize." We'll
          put the data into a pandas DataFrame.
        </p>
        <CodeIOBlock
          inputCode={`# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
circles.head(10)  `}
          outputCode={`
X1	X2	label
0	0.754246	0.231481	1
1	-0.756159	0.153259	1
2	-0.815392	0.173282	1
3	-0.393731	0.692883	1
4	0.442208	-0.896723	0
5	-0.479646	0.676435	1
6	-0.013648	0.803349	1
7	0.771513	0.147760	1
8	-0.169322	-0.793456	1
9	-0.121486	1.021509	0
  `}
        />

        <p>
          Each pair of X features (X1 and X2) has a label (y) value of either 0
          or 1, indicating a binary classification problem. Let's count how many
          values belong to each class.
        </p>

        <CodeIOBlock
          inputCode={`# Check different labels
circles.label.value_counts()  `}
          outputCode={`label
1    500
0    500
Name: count, dtype: int64  `}
        />

        <p>500 each, nice and balanced. Let's plot them.</p>

        <CodeIOBlock
          inputCode={`# Visualize with a plot
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0], 
            y=X[:, 1], 
            c=y, 
            cmap=plt.cm.RdYlBu);  `}
        />

        <img src={Visualize} className="centered-image" />

        <p>
          Now that we have a problem, let's figure out how to build a PyTorch
          neural network to classify dots as either red (0) or blue (1).
        </p>

        <div className="note">
          <strong>Note:</strong>This dataset is often what's considered a toy
          problem (a problem that's used to try and test things out on) in
          machine learning. But it represents the major key of classification,
          you have some kind of data represented as numerical values and you'd
          like to build a model that's able to classify it, in our case,
          separate it into red or blue dots.
        </div>

        <h2>1.1 Input and output shapes</h2>

        <p>
          One of the most common errors in deep learning is shape mismatch. To
          avoid this, it's crucial to be familiar with the input and output
          shapes of your data. Always ask yourself: "What are the shapes of my
          inputs and outputs?" Let's explore this.
        </p>

        <CodeIOBlock
          inputCode={`# Check the shapes of our features and labels
X.shape, y.shape  `}
          outputCode={`((1000, 2), (1000,))  `}
        />
        <p>
          We have a match on the first dimension: 1000 X and 1000 y. Now, let's
          determine the second dimension of X. Viewing a single sample's values
          and shapes can help clarify the expected input and output shapes for
          our model.
        </p>

        <CodeIOBlock
          inputCode={`# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")  `}
          outputCode={`Values for one sample of X: [0.75424625 0.23148074] and the same for y: 1
Shapes for one sample of X: (2,) and the same for y: ()  `}
        />

        <p>
          This indicates that the second dimension of X represents two features
          (a vector), while y is a single feature (a scalar). So, we have two
          inputs for one output.
        </p>

        <h2>1.2 Turn data into tensors and create train and test splits</h2>
        <p>
          Now that we've examined the input and output shapes of our data, let's
          prepare it for use with PyTorch and modeling. We need to:
          <ol>
            <li>
              <strong>Convert our data to PyTorch tensors</strong>: Currently,
              our data is in NumPy arrays, but PyTorch works best with PyTorch
              tensors. We can use <code>torch.from_numpy()</code> or{" "}
              <code>torch.tensor()</code> for this conversion
              <a href="https://apxml.com/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/pytorch-numpy-interfacing">
                [1]
              </a>
              <a href="https://stackabuse.com/numpy-array-to-tensor-and-tensor-to-numpy-array-with-pytorch/">
                [2]
              </a>
              <a href="https://www.tutorialspoint.com/how-to-convert-a-numpy-array-to-tensor">
                [3]
              </a>
              .
            </li>
            <li>
              <strong>Split our data into training and test sets</strong>: We'll
              train our model on the training set to learn patterns between X
              and y, and then evaluate those patterns on the test dataset.
            </li>
          </ol>
        </p>

        <CodeIOBlock
          inputCode={`# Turn data into tensors
# Otherwise this causes issues with computations later on
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# View the first five samples
X[:5], y[:5]  `}
          outputCode={`(tensor([[ 0.7542,  0.2315],
         [-0.7562,  0.1533],
         [-0.8154,  0.1733],
         [-0.3937,  0.6929],
         [ 0.4422, -0.8967]]),
 tensor([1., 1., 1., 1., 0.]))  `}
        />

        <p>
          Now that our data is in tensor format, let's split it into training
          and test sets using Scikit-Learn's <code>train_test_split()</code>{" "}
          function. We'll set <code>test_size=0.2</code> to allocate 80% for
          training and 20% for testing. To ensure the split is reproducible,
          we'll use <code>random_state=42</code>.
        </p>

        <CodeIOBlock
          inputCode={`# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

len(X_train), len(X_test), len(y_train), len(y_test)  `}
          outputCode={`(800, 200, 800, 200)  `}
        />
        <p>
          Nice! Looks like we've now got 800 training samples and 200 testing
          samples.
        </p>

        <h2>2. Building a model</h2>
        <p>
          Now that we have our data ready, it's time to build a model. We'll
          break this down into several steps:
          <ol>
            <li>
              <strong>Set up device-agnostic code</strong>: This allows our
              model to run on either a CPU or GPU if available.
            </li>
            <li>
              <strong>Construct a model</strong>: We'll do this by subclassing{" "}
              <code>nn.Module</code>.
            </li>
            <li>
              <strong>Define a loss function and optimizer</strong>.
            </li>
            <li>
              <strong>Create a training loop</strong> (covered in the next
              section).
            </li>
          </ol>
          We've already covered these steps in Notebook 01, but now we'll adapt
          them for a classification dataset. Let's start by importing PyTorch
          and setting up device-agnostic code.
        </p>

        <CodeIOBlock
          inputCode={`# Standard PyTorch imports
import torch
from torch import nn

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device  `}
          outputCode={`'cuda'  `}
        />

        <p>
          Now that we have our device set up, let's create a model. We need a
          model that can take X as inputs and produce outputs in the shape of y.
          This is a supervised learning setup where the data guides the model on
          what outputs to produce for given inputs. To create this model, we'll
          need to handle the input and output shapes of X and y. Here's how
          we'll do it:
          <ol>
            <li>
              <strong>
                Subclass <code>nn.Module</code>
              </strong>
              : Most PyTorch models subclass <code>nn.Module</code>.
            </li>
            <li>
              <strong>
                Create <code>nn.Linear</code> layers
              </strong>
              : In the constructor, we'll create two linear layers capable of
              handling the input and output shapes of X and y.
            </li>
            <li>
              <strong>
                Define a <code>forward()</code> method
              </strong>
              : This method will contain the forward pass computation of the
              model.
            </li>
            <li>
              <strong>
                Instantiate the model and move it to the target device
              </strong>
              .
            </li>
          </ol>
        </p>
        <CodeIOBlock
          inputCode={`# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 feature (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)
model_0  `}
          outputCode={`CircleModelV0(
  (layer_1): Linear(in_features=2, out_features=5, bias=True)
  (layer_2): Linear(in_features=5, out_features=1, bias=True)
)  `}
        />
        <p>Let's break down what's happening here:</p>
        <ul>
          <li>
            <strong>
              First Layer (<code>self.layer_1</code>)
            </strong>
            : This layer takes 2 input features (<code>in_features=2</code>) and
            produces 5 output features (<code>out_features=5</code>). These 5
            outputs are known as hidden units or neurons. By transforming the
            input from 2 features to 5, the model can learn more complex
            patterns.
          </li>
          <li>
            <strong>Why Use Hidden Units?</strong>: Using more features allows
            the model to potentially learn better patterns. However, the number
            of hidden units is a hyperparameter, and there's no
            one-size-fits-all value. Generally, more units can be better, but
            too many can be detrimental. For our simple dataset, we'll keep it
            small.
          </li>
          <li>
            <strong>
              Second Layer (<code>self.layer_2</code>)
            </strong>
            : This layer must take the same number of input features as the
            previous layer's output features. So, <code>self.layer_2</code> has{" "}
            <code>in_features=5</code> (matching <code>self.layer_1</code>'s{" "}
            <code>out_features=5</code>) and outputs 1 feature (
            <code>out_features=1</code>), matching the shape of y.
          </li>
        </ul>
        <img src={Playground} className="centered-image" />

        <p>
          To visualize a similar classification neural network, you can create
          one on the{" "}
          <b>
            <a href="https://playground.tensorflow.org/" target="_blank">
              TensorFlow Playground
            </a>
          </b>{" "}
          website. This interactive tool allows you to experiment with different
          neural network configurations and see how they classify data points.
          Alternatively, you can achieve a similar setup using PyTorch's{" "}
          <code>nn.Sequential</code>. This method allows you to define a model
          by sequentially adding layers, which then perform a forward pass
          computation on the input data in the order they are defined.
        </p>

        <CodeIOBlock
          inputCode={`# Replicate CircleModelV0 with nn.Sequential
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

model_0  `}
          outputCode={`Sequential(
  (0): Linear(in_features=2, out_features=5, bias=True)
  (1): Linear(in_features=5, out_features=1, bias=True)
)  `}
        />

        <p>
          While <code>nn.Sequential</code> is simpler than subclassing{" "}
          <code>nn.Module</code>, it's limited to sequential computations. If
          you need more complex logic, you should define a custom{" "}
          <code>nn.Module</code> subclass. Now that we have a model, let's see
          what happens when we pass some data through it.
        </p>

        <CodeIOBlock
          inputCode={`# Make predictions with the model
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")  `}
          outputCode={`Length of predictions: 200, Shape: torch.Size([200, 1])
Length of test samples: 200, Shape: torch.Size([200])

First 10 predictions:
tensor([[0.0555],
        [0.0169],
        [0.2254],
        [0.0071],
        [0.3345],
        [0.3101],
        [0.1151],
        [0.1840],
        [0.2205],
        [0.0156]], device='cuda:0', grad_fn=<SliceBackward0>)

First 10 test labels:
tensor([1., 0., 1., 0., 1., 1., 0., 0., 1., 0.])  `}
        />
        <p>
          It seems like we have the same number of predictions as test labels,
          but the predictions don't match the form or shape of the test labels.
          We can address this with a few steps, which we'll explore later.
        </p>

        <h2>2.1 Loss function and optimizer</h2>
        <p>
          We've previously set up a loss function (also known as a criterion or
          cost function) and an optimizer in Notebook 01. However, different
          types of problems require different loss functions. For instance:
          <ul>
            <li>
              <strong>Regression</strong>: You might use Mean Absolute Error
              (MAE) loss.
            </li>
            <li>
              <strong>Binary Classification</strong>: Binary Cross Entropy is
              commonly used.
            </li>
          </ul>
          On the other hand, the same optimizer can often be applied across
          different problem types. For example, both the Stochastic Gradient
          Descent optimizer (<code>torch.optim.SGD()</code>) and the Adam
          optimizer (<code>torch.optim.Adam()</code>) are versatile and can be
          used in a range of scenarios.
        </p>
        <table>
          <thead>
            <tr>
              <th>Loss function/Optimizer</th>
              <th>Problem type</th>
              <th>PyTorch Code</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Stochastic Gradient Descent (SGD) optimizer</td>
              <td>Classification, regression, many others.</td>
              <td>
                <code>torch.optim.SGD()</code>
              </td>
            </tr>
            <tr>
              <td>Adam Optimizer</td>
              <td>Classification, regression, many others.</td>
              <td>
                <code>torch.optim.Adam()</code>
              </td>
            </tr>
            <tr>
              <td>Binary cross entropy loss</td>
              <td>Binary classification</td>
              <td>
                <code>torch.nn.BCELossWithLogits</code> or{" "}
                <code>torch.nn.BCELoss</code>
              </td>
            </tr>
            <tr>
              <td>Cross entropy loss</td>
              <td>Multi-class classification</td>
              <td>
                <code>torch.nn.CrossEntropyLoss</code>
              </td>
            </tr>
            <tr>
              <td>Mean absolute error (MAE) or L1 Loss</td>
              <td>Regression</td>
              <td>
                <code>torch.nn.L1Loss</code>
              </td>
            </tr>
            <tr>
              <td>Mean squared error (MSE) or L2 Loss</td>
              <td>Regression</td>
              <td>
                <code>torch.nn.MSELoss</code>
              </td>
            </tr>
          </tbody>
        </table>
        <p>
          Table of various loss functions and optimizers, there are more but
          these are some common ones you'll see. Since we're working with a
          binary classification problem, let's use a binary cross entropy loss
          function.
        </p>

        <div className="note">
          <strong>Note:</strong> Recall a loss function is what measures how
          wrong your model predictions are, the higher the loss, the worse your
          model. Also, PyTorch documentation often refers to loss functions as
          "loss criterion" or "criterion", these are all different ways of
          describing the same thing.
        </div>

        <p>
          PyTorch offers two binary cross-entropy implementations:
          <ol>
            <li>
              <strong>
                <code>torch.nn.BCELoss()</code>
              </strong>
              : This measures the binary cross entropy between the target and
              input. It expects the input to be probabilities, so you would
              typically apply a sigmoid function before using it[1][2].
            </li>
            <li>
              <strong>
                <code>torch.nn.BCEWithLogitsLoss()</code>
              </strong>
              : This includes a built-in sigmoid layer, making it more
              numerically stable than applying <code>nn.Sigmoid</code> followed
              by <code>BCELoss</code>. It's generally the better choice unless
              you need more control over the sigmoid application[2][4].
            </li>
          </ol>
          For our purposes, <code>torch.nn.BCEWithLogitsLoss()</code> is
          recommended. Let's create a loss function using{" "}
          <code>torch.nn.BCEWithLogitsLoss()</code> and an optimizer with{" "}
          <code>torch.optim.SGD()</code> at a learning rate of 0.1.
        </p>

        <CodeIOBlock
          inputCode={`# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)  `}
        />
        <p>
          Now, let's create an evaluation metric. While a loss function measures
          how wrong your model is, an evaluation metric can be seen as measuring
          how right it is. It provides a different perspective on your model's
          performance. For classification problems, one common evaluation metric
          is <strong>accuracy</strong>. <br></br>Accuracy is calculated by
          dividing the total number of correct predictions by the total number
          of predictions. For instance, if a model makes 99 correct predictions
          out of 100, its accuracy is 99%. Let's write a function to calculate
          accuracy.
        </p>

        <CodeIOBlock
          inputCode={`# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc  `}
        />

        <p>
          Excellent! We can now use this function whilst training our model to
          measure it's performance alongside the loss.
        </p>

        <h2>3. Train model</h2>
        <p>
          Now that we have our loss function and optimizer set up, let's proceed
          to train our model.
        </p>

        <h2>
          Going from raw model outputs to predicted labels ( logits -)
          prediction probabilities -) prediction labels )
        </h2>

        <p>
          Before diving into the training loop, let's examine what comes out of
          our model during the forward pass. This is defined by the{" "}
          <code>forward()</code> method. To do this, we'll pass some data
          through the model.
        </p>

        <CodeIOBlock
          inputCode={`# View the frist 5 outputs of the forward pass on the test data
y_logits = model_0(X_test.to(device))[:5]
y_logits  `}
          outputCode={`tensor([[0.0555],
        [0.0169],
        [0.2254],
        [0.0071],
        [0.3345]], device='cuda:0', grad_fn=<SliceBackward0>)  `}
        />

        <p>
          Since our model hasn't been trained, these outputs are basically
          random.
          <br />
          <br />
          But what are they?
          <br />
          <br />
          They're the output of our <code>forward()</code> method.
          <br />
          <br />
          This method implements two layers of <code>nn.Linear()</code>, which
          internally use the following equation:
          <br />
          <code>
            y = x · Weights<sup>T</sup> + bias
          </code>
          <br />
          <br />
          The raw outputs (unmodified) of this equation (<code>y</code>), and in
          turn the raw outputs of our model, are often referred to as{" "}
          <strong>logits</strong>.
          <br />
          <br />
          That's what our model is outputting above when it takes in the input
          data (<code>x</code> in the equation or <code>X_test</code> in the
          code): logits.
          <br />
          <br />
          However, these numbers are hard to interpret.
          <br />
          We'd like some numbers that are comparable to our truth labels.
          <br />
          <br />
          To convert our model's raw outputs (logits) into such a form, we can
          use the <strong>sigmoid activation function</strong>.
          <br />
          <br />
          Let's try it out.
        </p>

        <CodeIOBlock
          inputCode={`# Use sigmoid on model logits
y_pred_probs = torch.sigmoid(y_logits)
y_pred_probs  `}
          outputCode={`tensor([[0.5139],
        [0.5042],
        [0.5561],
        [0.5018],
        [0.5829]], device='cuda:0', grad_fn=<SigmoidBackward0>)  `}
        />

        <p>
          Okay, it seems like the outputs now have some kind of consistency
          (even though they're still random).
          <br />
          <br />
          They're now in the form of prediction probabilities (often referred to
          as <code>y_pred_probs</code>). In other words, the values now
          represent how confident the model is that a data point belongs to one
          class or another.
          <br />
          <br />
          In our case, since we're dealing with{" "}
          <a
            href="https://en.wikipedia.org/wiki/Binary_classification"
            target="_blank"
            rel="noopener noreferrer"
          >
            binary classification
          </a>
          , our ideal outputs are 0 or 1.
          <br />
          <br />
          So these values can be viewed as a decision boundary.
          <br />
          <br />
          The closer the value is to 0, the more the model believes the sample
          belongs to class 0; the closer to 1, the more the model believes the
          sample belongs to class 1.
          <br />
          <br />
          More specifically:
          <br />
          If <code>y_pred_probs &gt;= 0.5</code>, then <code>y = 1</code> (class
          1)
          <br />
          If <code>y_pred_probs &lt; 0.5</code>, then <code>y = 0</code> (class
          0)
          <br />
          <br />
          To convert our prediction probabilities into prediction labels, we can
          round the outputs of the sigmoid activation function.
        </p>

        <CodeIOBlock
          inputCode={`# Find the predicted labels (round the prediction probabilities)
y_preds = torch.round(y_pred_probs)

# In full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Get rid of extra dimension
y_preds.squeeze()  `}
          outputCode={`tensor([True, True, True, True, True], device='cuda:0')
tensor([1., 1., 1., 1., 1.], device='cuda:0', grad_fn=<SqueezeBackward0>)  `}
        />
        <p>
          Excellent! Now it looks like our model's predictions are in the same
          form as our truth labels (<code>y_test</code>).
        </p>

        <CodeIOBlock
          inputCode={`y_test[:5]  `}
          outputCode={`tensor([1., 0., 1., 0., 1.])  `}
        />

        <p>
          This means we'll be able to compare our model's predictions to the
          test labels to see how well it's performing.
          <br />
          <br />
          To recap, we converted our model's raw outputs (logits) to prediction
          probabilities using a sigmoid activation function.
          <br />
          <br />
          Then, we converted the prediction probabilities to prediction labels
          by rounding them.
        </p>
        <div className="note">
          <strong>Note:</strong> The use of the sigmoid activation function is
          often only for binary classification logits. For multi-class
          classification, we'll be looking at using the{" "}
          <strong>softmax activation function</strong> (this will come later
          on).
          <br />
          <br />
          Additionally, the use of the sigmoid activation function is not
          required when passing our model's raw outputs to the{" "}
          <code>nn.BCEWithLogitsLoss</code> (the "logits" in logits loss is
          because it works on the model's raw logits output). This is because it
          has a sigmoid function built-in.
        </div>

        <h2>3.2 Building a training and testing loop</h2>

        <p>
          Alright, we've discussed how to take our raw model outputs and convert
          them to prediction labels. Now, let's build a training loop.
          <br />
          <br />
          Let's start by training for 100 epochs and outputting the model's
          progress every 10 epochs.
        </p>

        <CodeIOBlock
          inputCode={`torch.manual_seed(42)

# Set the number of epochs
epochs = 100

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra \`1\` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train) 
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        console.log(\`Epoch: \${epoch} | Loss: \${loss:.5f}, Accuracy: \${acc:.2f}% | Test loss: \${test_loss:.5f}, Test acc: \${test_acc:.2f}%\`)`}
          outputCode={`Epoch: 0 | Loss: 0.70034, Accuracy: 50.00% | Test loss: 0.69484, Test acc: 52.50%
Epoch: 10 | Loss: 0.69718, Accuracy: 53.75% | Test loss: 0.69242, Test acc: 54.50%
Epoch: 20 | Loss: 0.69590, Accuracy: 51.12% | Test loss: 0.69161, Test acc: 53.50%
Epoch: 30 | Loss: 0.69530, Accuracy: 50.62% | Test loss: 0.69136, Test acc: 53.00%
Epoch: 40 | Loss: 0.69497, Accuracy: 49.75% | Test loss: 0.69131, Test acc: 53.50%
Epoch: 50 | Loss: 0.69474, Accuracy: 50.12% | Test loss: 0.69134, Test acc: 53.50%
Epoch: 60 | Loss: 0.69457, Accuracy: 49.88% | Test loss: 0.69139, Test acc: 53.50%
Epoch: 70 | Loss: 0.69442, Accuracy: 49.62% | Test loss: 0.69146, Test acc: 54.00%
Epoch: 80 | Loss: 0.69430, Accuracy: 49.62% | Test loss: 0.69153, Test acc: 54.50%
Epoch: 90 | Loss: 0.69418, Accuracy: 49.62% | Test loss: 0.69161, Test acc: 54.50%`}
        />

        <p>Hmm, what do you notice about the performance of our model?</p>
        <p>
          It looks like it went through the training and testing steps fine, but
          the results don't seem to have improved much.
        </p>
        <p>The accuracy barely moves above 50% on each data split.</p>
        <p>
          And because we're working with a balanced binary classification
          problem, this means our model is performing as well as random
          guessing. (With 500 samples of class 0 and class 1, a model predicting
          class 1 every single time would achieve 50% accuracy.)
        </p>

        <h2>4. Make predictions and evaluate the model</h2>

        <p>From the metrics, it looks like our model is random guessing.</p>
        <p>How could we investigate this further?</p>
        <p>I've got an idea.</p>
        <p>The data explorer's motto!</p>
        <p>
          <strong>"Visualize, visualize, visualize!"</strong>
        </p>
        <p>
          Let's make a plot of our model's predictions, the data it's trying to
          predict on, and the decision boundary it's creating for whether
          something is class 0 or class 1.
        </p>
        <p>
          To do so, we'll write some code to download and import the{" "}
          <code>helper_functions.py</code> script from the Learn PyTorch for
          Deep Learning repo.
        </p>
        <p>
          It contains a helpful function called{" "}
          <code>plot_decision_boundary()</code>, which creates a NumPy meshgrid
          to visually plot the different points where our model is predicting
          certain classes.
        </p>
        <p>
          We'll also import <code>plot_predictions()</code> which we wrote in
          notebook 01 to use later.
        </p>
        <CodeIOBlock
          inputCode={`import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary  `}
          outputCode={`helper_functions.py already exists, skipping download  `}
        />
        <CodeIOBlock
          inputCode={`# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)  `}
        />
        <img src={Traintest} className="centered-image" />

        <p>
          Oh wow, it seems like we've found the cause of the model's performance
          issue.
        </p>
        <p>
          It's currently trying to split the red and blue dots using a straight
          line...
        </p>
        <p>
          That explains the 50% accuracy. Since our data is circular, drawing a
          straight line can at best cut it down the middle.
        </p>
        <p>
          In machine learning terms, our model is <strong>underfitting</strong>,
          meaning it's not learning predictive patterns from the data.
        </p>
        <p>How could we improve this?</p>

        <h2>5. Improving a model (from a model perspective)</h2>
        <p>Let's try to fix our model's underfitting problem.</p>
        <p>
          Focusing specifically on the model (not the data), there are a few
          ways we could do this:
        </p>

        <table>
          <thead>
            <tr>
              <th>Model improvement technique</th>
              <th>What does it do?</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Add more layers</td>
              <td>
                Each layer potentially increases the learning capabilities of
                the model with each layer being able to learn some kind of new
                pattern in the data. More layers are often referred to as making
                your neural network deeper.
              </td>
            </tr>
            <tr>
              <td>Add more hidden units</td>
              <td>
                Similar to the above, more hidden units per layer means a
                potential increase in learning capabilities of the model. More
                hidden units are often referred to as making your neural network
                wider.
              </td>
            </tr>
            <tr>
              <td>Fitting for longer (more epochs)</td>
              <td>
                Your model might learn more if it had more opportunities to look
                at the data.
              </td>
            </tr>
            <tr>
              <td>Changing the activation functions</td>
              <td>
                Some data just can't be fit with only straight lines (like what
                we've seen), using non-linear activation functions can help with
                this (hint, hint).
              </td>
            </tr>
            <tr>
              <td>Change the learning rate</td>
              <td>
                Less model specific, but still related, the learning rate of the
                optimizer decides how much a model should change its parameters
                each step, too much and the model overcorrects, too little and
                it doesn't learn enough.
              </td>
            </tr>
            <tr>
              <td>Change the loss function</td>
              <td>
                Again, less model specific but still important, different
                problems require different loss functions. For example, a binary
                cross entropy loss function won't work with a multi-class
                classification problem.
              </td>
            </tr>
            <tr>
              <td>Use transfer learning</td>
              <td>
                Take a pretrained model from a problem domain similar to yours
                and adjust it to your own problem. We cover transfer learning in
                notebook 06.
              </td>
            </tr>
          </tbody>
        </table>

        <div className="note">
          <strong>Note:</strong> *because you can adjust all of these by hand,
          they're referred to as hyperparameters. And this is also where machine
          learning's half art half science comes in, there's no real way to know
          here what the best combination of values is for your project, best to
          follow the data scientist's motto of "experiment, experiment,
          experiment".
        </div>

        <p>
          Let's see what happens if we add an extra layer to our model, fit for
          longer (<code>epochs=1000</code> instead of <code>epochs=100</code>)
          and increase the number of hidden units from 5 to 10.
        </p>
        <p>
          We'll follow the same steps we did above but with a few changed
          hyperparameters.
        </p>

        <CodeIOBlock
          inputCode={`class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10) # extra layer
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x): # note: always make sure forward is spelt correctly!
        # Creating a model like this is the same as below, though below
        # generally benefits from speedups where possible.
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        # return z
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV1().to(device)
model_1  `}
          outputCode={`CircleModelV1(
  (layer_1): Linear(in_features=2, out_features=10, bias=True)
  (layer_2): Linear(in_features=10, out_features=10, bias=True)
  (layer_3): Linear(in_features=10, out_features=1, bias=True)
)  `}
        />

        <p>
          Now we've got a model, we'll recreate a loss function and optimizer
          instance, using the same settings as before.
        </p>

        <CodeIOBlock
          inputCode={`# loss_fn = nn.BCELoss() # Requires sigmoid on input
loss_fn = nn.BCEWithLogitsLoss() # Does not require sigmoid on input
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)  `}
        />
        <p>
          Great! Now that the model, optimizer, and loss function are ready,
          let's create a training loop.
        </p>
        <p>
          This time, we'll train for longer (<code>epochs=1000</code> instead of{" "}
          <code>epochs=100</code>) and see if it improves our model's
          performance.
        </p>

        <CodeIOBlock
          inputCode={`torch.manual_seed(42)

epochs = 1000 # Train for longer

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    ### Training
    # 1. Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
  `}
          outputCode={`Epoch: 0 | Loss: 0.69396, Accuracy: 50.88% | Test loss: 0.69261, Test acc: 51.00%
Epoch: 100 | Loss: 0.69305, Accuracy: 50.38% | Test loss: 0.69379, Test acc: 48.00%
Epoch: 200 | Loss: 0.69299, Accuracy: 51.12% | Test loss: 0.69437, Test acc: 46.00%
Epoch: 300 | Loss: 0.69298, Accuracy: 51.62% | Test loss: 0.69458, Test acc: 45.00%
Epoch: 400 | Loss: 0.69298, Accuracy: 51.12% | Test loss: 0.69465, Test acc: 46.00%
Epoch: 500 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69467, Test acc: 46.00%
Epoch: 600 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69468, Test acc: 46.00%
Epoch: 700 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69468, Test acc: 46.00%
Epoch: 800 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69468, Test acc: 46.00%
Epoch: 900 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69468, Test acc: 46.00%  `}
        />

        <p>
          What? Our model trained for longer and with an extra layer, but it
          still looks like it didn't learn any patterns better than random
          guessing.
        </p>
        <p>Let's visualize.</p>

        <CodeIOBlock
          inputCode={`# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)  `}
        />
        <img src={modelResults} className="centered-image" />

        <p>Hmmm.</p>
        <p>
          Our model is still drawing a straight line between the red and blue
          dots.
        </p>
        <p>
          If our model is drawing a straight line, could it model linear data?
          Like we did in notebook 01?
        </p>

        <h2>
          5.1 Preparing data to see if our model can model a straight line
        </h2>

        <p>
          Let's create some linear data to see if our model's able to model it
          and we're not just using a model that can't learn anything.
        </p>

        <CodeIOBlock
          inputCode={`# Create some data (same as notebook 01)
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias # linear regression formula

# Check the data
print(len(X_regression))
X_regression[:5], y_regression[:5]  `}
          outputCode={`100
(tensor([[0.0000],
         [0.0100],
         [0.0200],
         [0.0300],
         [0.0400]]),
 tensor([[0.3000],
         [0.3070],
         [0.3140],
         [0.3210],
         [0.3280]]))  `}
        />

        <p>Wonderful, now let's split our data into training and test sets.</p>

        <CodeIOBlock
          inputCode={`# Create train and test splits
train_split = int(0.8 * len(X_regression)) # 80% of data used for training set
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Check the lengths of each split
print(len(X_train_regression), 
    len(y_train_regression), 
    len(X_test_regression), 
    len(y_test_regression))  `}
          outputCode={`0 80 20 20  `}
        />

        <p>Beautiful, let's see how the data looks.</p>
        <p>
          To do so, we'll use the <code>plot_predictions()</code> function we
          created in notebook 01.
        </p>
        <p>
          It's contained within the <code>helper_functions.py</code> script on
          the Learn PyTorch for Deep Learning repo which we downloaded above.
        </p>

        <CodeIOBlock
          inputCode={`plot_predictions(train_data=X_train_regression,
    train_labels=y_train_regression,
    test_data=X_test_regression,
    test_labels=y_test_regression
);  `}
        />

        <img src={Graph} className="centered-image" />

        <h2>5.2 Adjusting model_1 to fit a straight line</h2>

        <p>
          Now that we have the data, let's recreate <code>model_1</code>, but
          this time with a loss function that's better suited for our regression
          data.
        </p>

        <CodeIOBlock
          inputCode={`# Same architecture as model_1 (but using nn.Sequential)
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

model_2  `}
          outputCode={`Sequential(
  (0): Linear(in_features=1, out_features=10, bias=True)
  (1): Linear(in_features=10, out_features=10, bias=True)
  (2): Linear(in_features=10, out_features=1, bias=True)
)  `}
        />

        <p>
          We'll set up the loss function as <code>nn.L1Loss()</code> (which is
          the same as mean absolute error) and the optimizer as{" "}
          <code>torch.optim.SGD()</code>.
        </p>

        <CodeIOBlock
          inputCode={`# Loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)  `}
        />

        <p>
          Now let's train the model using the regular training loop steps for{" "}
          <code>epochs=1000</code> (just like <code>model_1</code>).
        </p>

        <div className="note">
          <strong>Note:</strong>We've been writing similar training loop code
          over and over again. I've made it that way on purpose though, to keep
          practicing. However, do you have ideas how we could functionize this?
          That would save a fair bit of coding in the future. Potentially there
          could be a function for training and a function for testing.
        </div>

        <CodeIOBlock
          inputCode={`# Train the model
torch.manual_seed(42)

# Set the number of epochs
epochs = 1000

# Put data to target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

for epoch in range(epochs):
    ### Training 
    # 1. Forward pass
    y_pred = model_2(X_train_regression)
    
    # 2. Calculate loss (no accuracy since it's a regression problem, not classification)
    loss = loss_fn(y_pred, y_train_regression)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_2.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_pred = model_2(X_test_regression)
      # 2. Calculate the loss 
      test_loss = loss_fn(test_pred, y_test_regression)

    # Print out what's happening
    if epoch % 100 == 0: 
        print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")  `}
          outputCode={`Epoch: 0 | Train loss: 0.75986, Test loss: 0.54143
Epoch: 100 | Train loss: 0.09309, Test loss: 0.02901
Epoch: 200 | Train loss: 0.07376, Test loss: 0.02850
Epoch: 300 | Train loss: 0.06745, Test loss: 0.00615
Epoch: 400 | Train loss: 0.06107, Test loss: 0.02004
Epoch: 500 | Train loss: 0.05698, Test loss: 0.01061
Epoch: 600 | Train loss: 0.04857, Test loss: 0.01326
Epoch: 700 | Train loss: 0.06109, Test loss: 0.02127
Epoch: 800 | Train loss: 0.05599, Test loss: 0.01426
Epoch: 900 | Train loss: 0.05571, Test loss: 0.00603  `}
        />

        <p>
          Okay, unlike <code>model_1</code> on the classification data, it looks
          like <code>model_2</code>'s loss is actually going down.
        </p>
        <p>Let's plot its predictions to see if that's the case.</p>
        <p>
          And remember, since our model and data are using the target device,
          and this device may be a GPU, our plotting function uses{" "}
          <code>matplotlib</code>. However, <code>matplotlib</code> can't handle
          data on the GPU.
        </p>
        <p>
          To handle that, we'll send all of our data to the CPU using{" "}
          <code>.cpu()</code> when we pass it to <code>plot_predictions()</code>
          .
        </p>

        <CodeIOBlock
          inputCode={`# Turn on evaluation mode
model_2.eval()

# Make predictions (inference)
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

# Plot data and predictions with data on the CPU (matplotlib can't handle data on the GPU)
# (try removing .cpu() from one of the below and see what happens)
plot_predictions(train_data=X_train_regression.cpu(),
                 train_labels=y_train_regression.cpu(),
                 test_data=X_test_regression.cpu(),
                 test_labels=y_test_regression.cpu(),
                 predictions=y_preds.cpu());  `}
        />

        <img src={Result} className="centered-image" />

        <p>
          Alright, it looks like our model is able to do far better than random
          guessing on straight lines.
        </p>
        <p>This is a good thing.</p>
        <p>It means our model at least has some capacity to learn.</p>

        <div className="note">
          <strong>Note:</strong>A helpful troubleshooting step when building
          deep learning models is to start as small as possible to see if the
          model works before scaling it up. This could mean starting with a
          simple neural network (not many layers, not many hidden neurons) and a
          small dataset (like the one we've made) and then overfitting (making
          the model perform too well) on that small example before increasing
          the amount of data or the model size/design to reduce overfitting.
        </div>

        <h2>6.1 Recreating non-linear data (red and blue circles)</h2>

        <p>
          First, let's recreate the data to start off fresh. We'll use the same
          setup as before.
        </p>

        <CodeIOBlock
          inputCode={`# Make and plot data
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000

X, y = make_circles(n_samples=1000,
    noise=0.03,
    random_state=42,
)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu);  `}
        />

        <img src={Circle} className="centered-image" />
        <p>
          Nice! Now let's split it into training and test sets using 80% of the
          data for training and 20% for testing.
        </p>

        <CodeIOBlock
          inputCode={`# Convert to tensors and split into train and test sets
import torch
from sklearn.model_selection import train_test_split

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42
)

X_train[:5], y_train[:5]  `}
          outputCode={`(tensor([[ 0.6579, -0.4651],
         [ 0.6319, -0.7347],
         [-1.0086, -0.1240],
         [-0.9666, -0.2256],
         [-0.1666,  0.7994]]),
 tensor([1., 0., 0., 0., 1.]))  `}
        />

        <h2> 6.2 Building a model with non-linearity </h2>

        <p>Now here comes the fun part.</p>
        <p>
          What kind of pattern do you think you could draw with unlimited
          straight (linear) and non-straight (non-linear) lines?
        </p>
        <p>I bet you could get pretty creative.</p>
        <p>
          So far our neural networks have only been using linear (straight) line
          functions.
        </p>
        <p>But the data we've been working with is non-linear (circles).</p>
        <p>
          What do you think will happen when we introduce the capability for our
          model to use non-linear activation functions?
        </p>
        <p>Well, let's see.</p>
        <p>
          PyTorch has a bunch of ready-made non-linear activation functions that
          do similar but different things.
        </p>
        <p>
          One of the most common and best performing is ReLU (rectified
          linear-unit, <code>torch.nn.ReLU()</code>).
        </p>
        <p>
          Rather than talk about it, let's put it in our neural network between
          the hidden layers in the forward pass and see what happens.
        </p>

        <CodeIOBlock
          inputCode={`# Build model with non-linear activation function
from torch import nn
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)
print(model_3)  `}
          outputCode={`CircleModelV2(
  (layer_1): Linear(in_features=2, out_features=10, bias=True)
  (layer_2): Linear(in_features=10, out_features=10, bias=True)
  (layer_3): Linear(in_features=10, out_features=1, bias=True)
  (relu): ReLU()
)  `}
        />

        <img src={Relu} className="centered-image" />

        <p>
          <strong>
            A{" "}
            <a href="https://playground.tensorflow.org" target="_blank">
              classification neural network on TensorFlow Playground with ReLU
              activation
            </a>
          </strong>{" "}
          is a visual example of what a similar classification neural network to
          the one we've just built (using ReLU activation) looks like. Try
          creating one of your own on the{" "}
          <a href="https://playground.tensorflow.org" target="_blank">
            <strong>TensorFlow Playground website</strong>
          </a>
          .
        </p>

        <div className="resource">
          Question
          <p>
            Where should I put the non-linear activation functions when
            constructing a neural network? A rule of thumb is to put them in
            between hidden layers and just after the output layer, however,
            there is no set in stone option. As you learn more about neural
            networks and deep learning you'll find a bunch of different ways of
            putting things together. In the meantime, best to experiment,
            experiment, experiment.
          </p>
        </div>

        <p>
          Now that we've got a model ready to go, let's create a binary
          classification loss function as well as an optimizer.
        </p>

        <CodeIOBlock
          inputCode={`# Setup loss and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)  `}
        />

        <p>Wonderful!</p>

        <h2>6.3 Training a model with non-linearity</h2>

        <p>
          You know the drill, model, loss function, and optimizer are ready to
          go, so let's create a training and testing loop.
        </p>

        <CodeIOBlock
          inputCode={`# Fit the model
torch.manual_seed(42)
epochs = 1000

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_3(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calculate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")  `}
          outputCode={`Epoch: 0 | Loss: 0.69295, Accuracy: 50.00% | Test Loss: 0.69319, Test Accuracy: 50.00%
Epoch: 100 | Loss: 0.69115, Accuracy: 52.88% | Test Loss: 0.69102, Test Accuracy: 52.50%
Epoch: 200 | Loss: 0.68977, Accuracy: 53.37% | Test Loss: 0.68940, Test Accuracy: 55.00%
Epoch: 300 | Loss: 0.68795, Accuracy: 53.00% | Test Loss: 0.68723, Test Accuracy: 56.00%
Epoch: 400 | Loss: 0.68517, Accuracy: 52.75% | Test Loss: 0.68411, Test Accuracy: 56.50%
Epoch: 500 | Loss: 0.68102, Accuracy: 52.75% | Test Loss: 0.67941, Test Accuracy: 56.50%
Epoch: 600 | Loss: 0.67515, Accuracy: 54.50% | Test Loss: 0.67285, Test Accuracy: 56.00%
Epoch: 700 | Loss: 0.66659, Accuracy: 58.38% | Test Loss: 0.66322, Test Accuracy: 59.00%
Epoch: 800 | Loss: 0.65160, Accuracy: 64.00% | Test Loss: 0.64757, Test Accuracy: 67.50%
Epoch: 900 | Loss: 0.62362, Accuracy: 74.00% | Test Loss: 0.62145, Test Accuracy: 79.00%  `}
        />

        <p>Looks Great</p>

        <h2>
          6.4 Evaluating a model trained with non-linear activation functions
        </h2>

        <p>
          Remember how our circle data is non-linear? Well, let's see how our
          model's predictions look now that the model's been trained with
          non-linear activation functions.
        </p>

        <CodeIOBlock
          inputCode={`# Make predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
y_preds[:10], y[:10] # want preds in same format as truth labels  `}
          outputCode={`
(tensor([1., 0., 1., 0., 0., 1., 0., 0., 1., 0.], device='cuda:0'),
 tensor([1., 1., 1., 1., 0., 1., 1., 1., 1., 0.]))  `}
        />

        <CodeIOBlock
          inputCode={`# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train) # model_1 = no non-linearity
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity  `}
        />

        <img src={inputOutput} className="centered-image" />

        <p>Nice! It's not perfect, but it's definitely better than before.</p>
        <p>
          Want to improve the test accuracy even more? Check out section 5 for
          tips on enhancing the model.
        </p>

        <h2>7. Replicating non-linear activation functions</h2>
        <p>
          We've already seen how adding non-linear activation functions helps
          the model handle non-linear data.
        </p>

        <div className="note">
          <strong>Note:</strong> Much of the data you'll encounter in the wild
          is non-linear (or a combination of linear and non-linear). Right now
          we've been working with dots on a 2D plot. But imagine if you had
          images of plants you'd like to classify, there's a lot of different
          plant shapes. Or text from Wikipedia you'd like to summarize, there's
          lots of different ways words can be put together (linear and
          non-linear patterns).
        </div>

        <p>
          We've already seen how adding non-linear activation functions helps
          the model handle non-linear data.
        </p>

        <CodeIOBlock
          inputCode={`# Create a toy tensor (similar to the data going into our model(s))
A = torch.arange(-10, 10, 1, dtype=torch.float32)
A  `}
          outputCode={`tensor([-10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0.,   1.,
          2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.])  `}
        />

        <p>Wonderful, now let's plot it.</p>

        <CodeIOBlock
          inputCode={`# Visualize the toy tensor
plt.plot(A);  `}
        />
        <img src={Plot} className="centered-image" />

        <p>A straight line, nice.</p>
        <p>
          Now, let's see how the <strong>ReLU activation function</strong>{" "}
          influences it.
        </p>
        <p>
          Instead of using PyTorch's ReLU (<code>torch.nn.ReLU</code>), we'll
          recreate it ourselves.
        </p>
        <p>
          The ReLU function turns all negative values to 0 and leaves the
          positive values as they are.
        </p>

        <CodeIOBlock
          inputCode={`# Create ReLU function by hand 
def relu(x):
  return torch.maximum(torch.tensor(0), x) # inputs must be tensors

# Pass toy tensor through ReLU function
relu(A)  `}
          outputCode={`tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7.,
        8., 9.])  `}
        />

        <p>
          It looks like our <strong>ReLU function</strong> worked, all of the
          negative values are now zeros.
        </p>
        <p>Let's plot them.</p>

        <CodeIOBlock
          inputCode={`# Plot ReLU activated toy tensor
plt.plot(relu(A));  `}
        />

        <img src={ReluFunction} className="centered-image" />
        <p>
          Nice! That looks exactly like the shape of the ReLU function on the{" "}
          <a
            href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)"
            target="_blank"
            rel="noopener noreferrer"
          >
            Wikipedia page for ReLU
          </a>
          .
        </p>
        <p>
          How about we try the <strong>sigmoid function</strong> we've been
          using?
        </p>
        <p>The sigmoid function formula goes like so:</p>
        <p>
          <code>out_i = 1 / (1 + Math.exp(-input_i))</code>
        </p>
        <p>
          Or using <em>x</em> as input:
        </p>
        <p>
          <code>S(x) = 1 / (1 + Math.exp(-x_i))</code>
        </p>
        <p>
          Where <em>S</em> stands for sigmoid, <em>e</em> stands for
          exponential, and <em>i</em> stands for a particular element in a
          tensor.
        </p>
        <p>
          Let's build a function to replicate the sigmoid function with PyTorch.
        </p>

        <CodeIOBlock
          inputCode={`# Create a custom sigmoid function
def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

# Test custom sigmoid on toy tensor
sigmoid(A)  `}
          outputCode={`tensor([4.5398e-05, 1.2339e-04, 3.3535e-04, 9.1105e-04, 2.4726e-03, 6.6929e-03,
        1.7986e-02, 4.7426e-02, 1.1920e-01, 2.6894e-01, 5.0000e-01, 7.3106e-01,
        8.8080e-01, 9.5257e-01, 9.8201e-01, 9.9331e-01, 9.9753e-01, 9.9909e-01,
        9.9966e-01, 9.9988e-01])  `}
        />

        <p>
          Woah, those values look a lot like prediction probabilities we've seen
          earlier, let's see what they look like visualized.
        </p>

        <CodeIOBlock
          inputCode={`# Plot sigmoid activated toy tensor
plt.plot(sigmoid(A));  `}
        />
        <img src={Sigmoid} className="centered-image" />

        <p>Looking good! We've gone from a straight line to a curved line.</p>
        <p>
          Now there's plenty more non-linear activation functions that exist in
          PyTorch that we haven't tried.
        </p>
        <p>But these two are two of the most common.</p>
        <p>
          And the point remains, what patterns could you draw using an unlimited
          amount of linear (straight) and non-linear (not straight) lines?
        </p>
        <p>Almost anything, right?</p>
        <p>
          That's exactly what our model is doing when we combine linear and
          non-linear functions.
        </p>
        <p>
          Instead of telling our model what to do, we give it tools to figure
          out how to best discover patterns in the data.
        </p>
        <p>And those tools are linear and non-linear functions.</p>

        <h2>
          8. Putting things together by building a multi-class PyTorch model
        </h2>

        <p>We've covered a fair bit.</p>
        <p>
          But now let's put it all together using a multi-class classification
          problem.
        </p>
        <p>
          Recall a binary classification problem deals with classifying
          something as one of two options (e.g. a photo as a cat photo or a dog
          photo) whereas a multi-class classification problem deals with
          classifying something from a list of more than two options (e.g.
          classifying a photo as a cat, a dog, or a chicken).
        </p>

        <img src={Chicken} className="centered-image" />
        <p>
          Example of binary vs. multi-class classification: Binary deals with
          two classes (one thing or another), whereas multi-class classification
          can deal with any number of classes over two. For example, the popular
          ImageNet-1k dataset is used as a computer vision benchmark and has
          1000 classes.
        </p>

        <h2>8.1 Creating multi-class classification data</h2>
        <p>
          To begin a multi-class classification problem, let's create some
          multi-class data.
        </p>
        <p>
          To do so, we can leverage Scikit-Learn's <code>make_blobs()</code>{" "}
          method.
        </p>
        <p>
          This method will create however many classes (using the{" "}
          <code>centers</code> parameter) we want.
        </p>
        <p>Specifically, let's do the following:</p>
        <ul>
          <li>
            Create some multi-class data with <code>make_blobs()</code>.
          </li>
          <li>
            Turn the data into tensors (the default of <code>make_blobs()</code>{" "}
            is to use NumPy arrays).
          </li>
          <li>
            Split the data into training and test sets using{" "}
            <code>train_test_split()</code>.
          </li>
          <li>Visualize the data.</li>
        </ul>

        <CodeIOBlock
          inputCode={`# Import dependencies
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
    n_features=NUM_FEATURES, # X features
    centers=NUM_CLASSES, # y labels 
    cluster_std=1.5, # give the clusters a little shake up (try changing this to 1.0, the default)
    random_state=RANDOM_SEED
)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
print(X_blob[:5], y_blob[:5])

# 3. Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
    y_blob,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# 4. Plot data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu);  `}
          outputCode={`tensor([[-8.4134,  6.9352],
        [-5.7665, -6.4312],
        [-6.0421, -6.7661],
        [ 3.9508,  0.6984],
        [ 4.2505, -0.2815]]) tensor([3, 2, 2, 1, 1])  `}
        />

        <img src={Blobs} className="centered-image" />

        <p>Nice! Looks like we've got some multi-class data ready to go.</p>
        <p>Let's build a model to separate the coloured blobs.</p>

        <div className="resource">
          Question
          <p>
            Does this dataset need non-linearity? Or could you draw a succession
            of straight lines to separate it?
          </p>
        </div>

        <h2>8.2 Building a multi-class classification model</h2>
        <p>We've created a few models in PyTorch so far.</p>
        <p>
          You might also be starting to get an idea of how flexible neural
          networks are.
        </p>
        <p>
          How about we build one similar to model_3 but this is still capable of
          handling multi-class data?
        </p>
        <p>
          To do so, let's create a subclass of <code>nn.Module</code> that takes
          in three hyperparameters:
        </p>
        <ul>
          <li>
            <code>input_features</code> - the number of X features coming into
            the model.
          </li>
          <li>
            <code>output_features</code> - the ideal numbers of output features
            we'd like (this will be equivalent to NUM_CLASSES or the number of
            classes in your multi-class classification problem).
          </li>
          <li>
            <code>hidden_units</code> - the number of hidden neurons we'd like
            each hidden layer to use.
          </li>
        </ul>
        <p>
          Since we're putting things together, let's setup some device agnostic
          code (we don't have to do this again in the same notebook, it's only a
          reminder).
        </p>
        <p>
          Then we'll create the model class using the hyperparameters above.
        </p>

        <CodeIOBlock
          inputCode={`# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device  `}
          outputCode={`'cuda'  `}
        />

        <CodeIOBlock
          inputCode={`from torch import nn

# Build model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
            nn.Linear(in_features=hidden_units, out_features=output_features), # how many classes are there?
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)

# Create an instance of BlobModel and send it to the target device
model_4 = BlobModel(input_features=NUM_FEATURES, 
                    output_features=NUM_CLASSES, 
                    hidden_units=8).to(device)
model_4  `}
          outputCode={`BlobModel(
  (linear_layer_stack): Sequential(
    (0): Linear(in_features=2, out_features=8, bias=True)
    (1): Linear(in_features=8, out_features=8, bias=True)
    (2): Linear(in_features=8, out_features=4, bias=True)
  )
)  `}
        />

        <p>
          Excellent! Our multi-class model is ready to go, let's create a loss
          function and optimizer for it.
        </p>

        <h2>
          8.3 Creating a loss function and optimizer for a multi-class PyTorch
          model
        </h2>

        <p>
          Since we're working on a multi-class classification problem, we'll use
          the <code>nn.CrossEntropyLoss()</code> method as our loss function.
        </p>
        <p>
          And we'll stick with using <code>SGD</code> with a learning rate of
          0.1 for optimizing our <code>model_4</code> parameters.
        </p>

        <CodeIOBlock
          inputCode={`# Create loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), 
                            lr=0.1) # exercise: try changing the learning rate here and seeing what happens to the model's performance  `}
        />

        <h2>
          8.4 Getting prediction probabilities for a multi-class PyTorch model
        </h2>

        <p>
          Alright, we've got a loss function and optimizer ready, and we're
          ready to train our model. But before we do, let's do a single forward
          pass with our model to see if it works.
        </p>

        <CodeIOBlock
          inputCode={`# Perform a single forward pass on the data (we'll need to put it to the target device for it to work)
model_4(X_blob_train.to(device))[:5]  `}
          outputCode={`tensor([[-1.2711, -0.6494, -1.4740, -0.7044],
        [ 0.2210, -1.5439,  0.0420,  1.1531],
        [ 2.8698,  0.9143,  3.3169,  1.4027],
        [ 1.9576,  0.3125,  2.2244,  1.1324],
        [ 0.5458, -1.2381,  0.4441,  1.1804]], device='cuda:0',
       grad_fn=<SliceBackward0>)  `}
        />

        <p>What's coming out here?</p>
        <p>It looks like we get one value per feature of each sample.</p>
        <p>Let's check the shape to confirm.</p>

        <CodeIOBlock
          inputCode={`# How many elements in a single prediction sample?
model_4(X_blob_train.to(device))[0].shape, NUM_CLASSES   `}
          outputCode={`(torch.Size([4]), 4)  `}
        />

        <p>
          Wonderful, our model is predicting one value for each class that we
          have.
        </p>
        <p>Do you remember what the raw outputs of our model are called?</p>
        <p>
          Hint: it rhymes with "frog splits" (no animals were harmed in the
          creation of these materials).
        </p>
        <p>If you guessed logits, you'd be correct.</p>
        <p>
          So right now our model is outputting logits, but what if we wanted to
          figure out exactly which label it was giving the sample?
        </p>
        <p>
          As in, how do we go from logits &rarr; prediction probabilities &rarr;
          prediction labels just like we did with the binary classification
          problem?
        </p>
        <p>That's where the softmax activation function comes into play.</p>
        <p>
          The softmax function calculates the probability of each prediction
          class being the actual predicted class compared to all other possible
          classes.
        </p>
        <p>If this doesn't make sense, let's see in code.</p>

        <CodeIOBlock
          inputCode={`# Make prediction logits with model
y_logits = model_4(X_blob_test.to(device))

# Perform softmax calculation on logits across dimension 1 to get prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1) 
print(y_logits[:5])
print(y_pred_probs[:5])  `}
          outputCode={`tensor([[-1.2549, -0.8112, -1.4795, -0.5696],
        [ 1.7168, -1.2270,  1.7367,  2.1010],
        [ 2.2400,  0.7714,  2.6020,  1.0107],
        [-0.7993, -0.3723, -0.9138, -0.5388],
        [-0.4332, -1.6117, -0.6891,  0.6852]], device='cuda:0',
       grad_fn=<SliceBackward0>)
tensor([[0.1872, 0.2918, 0.1495, 0.3715],
        [0.2824, 0.0149, 0.2881, 0.4147],
        [0.3380, 0.0778, 0.4854, 0.0989],
        [0.2118, 0.3246, 0.1889, 0.2748],
        [0.1945, 0.0598, 0.1506, 0.5951]], device='cuda:0',
       grad_fn=<SliceBackward0>)  `}
        />

        <p>Hmm, what's happened here?</p>
        <p>
          It may still look like the outputs of the softmax function are jumbled
          numbers (and they are, since our model hasn't been trained and is
          predicting using random patterns) but there's a very specific thing
          different about each sample.
        </p>
        <p>
          After passing the logits through the softmax function, each individual
          sample now adds to 1 (or very close to).
        </p>
        <p>Let's check.</p>

        <CodeIOBlock
          inputCode={`# Sum the first sample output of the softmax activation function 
torch.sum(y_pred_probs[0])  `}
          outputCode={`tensor(1., device='cuda:0', grad_fn=<SumBackward0>)  `}
        />
        <p>
          These prediction probabilities are essentially saying how much the
          model thinks the target X sample (the input) maps to each class.
        </p>
        <p>
          Since there's one value for each class in <code>y_pred_probs</code>,
          the index of the highest value is the class the model thinks the
          specific data sample most belongs to.
        </p>
        <p>
          We can check which index has the highest value using{" "}
          <code>torch.argmax()</code>.
        </p>

        <CodeIOBlock
          inputCode={`# Which class does the model think is *most* likely at the index 0 sample?
print(y_pred_probs[0])
print(torch.argmax(y_pred_probs[0]))  `}
          outputCode={`tensor([0.1872, 0.2918, 0.1495, 0.3715], device='cuda:0',
       grad_fn=<SelectBackward0>)
tensor(3, device='cuda:0')  `}
        />

        <p>
          You can see the output of <code>torch.argmax()</code> returns 3, so
          for the features (X) of the sample at index 0, the model is predicting
          that the most likely class value (y) is 3.
        </p>
        <p>
          Of course, right now this is just random guessing so it's got a 25%
          chance of being right (since there's four classes). But we can improve
          those chances by training the model.
        </p>

        <div className="note">
          <strong>Note:</strong> To summarize the above, a model's raw output is
          referred to as logits. For a multi-class classification problem, to
          turn the logits into prediction probabilities, you use the softmax
          activation function (torch.softmax). The index of the value with the
          highest prediction probability is the class number the model thinks is
          most likely given the input features for that sample (although this is
          a prediction, it doesn't mean it will be correct).
        </div>

        <h2>
          8.5 Creating a training and testing loop for a multi-class PyTorch
          model
        </h2>

        <p>
          Alright, now we've got all of the preparation steps out of the way,
          let's write a training and testing loop to improve and evaluate our
          model.
        </p>
        <p>
          We've done many of these steps before so much of this will be
          practice.
        </p>
        <p>
          The only difference is that we'll be adjusting the steps to turn the
          model outputs (logits) to prediction probabilities (using the softmax
          activation function) and then to prediction labels (by taking the
          argmax of the output of the softmax activation function).
        </p>
        <p>
          Let's train the model for <code>epochs=100</code> and evaluate it
          every 10 epochs.
        </p>

        <CodeIOBlock
          inputCode={`# Fit the model
torch.manual_seed(42)

# Set number of epochs
epochs = 100

# Put data to target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    ### Training
    model_4.train()

    # 1. Forward pass
    y_logits = model_4(X_blob_train) # model outputs raw logits 
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # go from logits -> prediction probabilities -> prediction labels
    # print(y_logits)
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_blob_train) 
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_4.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_4(X_blob_test)
      test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
      # 2. Calculate test loss and accuracy
      test_loss = loss_fn(test_logits, y_blob_test)
      test_acc = accuracy_fn(y_true=y_blob_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")   `}
          outputCode={`
Epoch: 0 | Loss: 1.04324, Acc: 65.50% | Test Loss: 0.57861, Test Acc: 95.50%
Epoch: 10 | Loss: 0.14398, Acc: 99.12% | Test Loss: 0.13037, Test Acc: 99.00%
Epoch: 20 | Loss: 0.08062, Acc: 99.12% | Test Loss: 0.07216, Test Acc: 99.50%
Epoch: 30 | Loss: 0.05924, Acc: 99.12% | Test Loss: 0.05133, Test Acc: 99.50%
Epoch: 40 | Loss: 0.04892, Acc: 99.00% | Test Loss: 0.04098, Test Acc: 99.50%
Epoch: 50 | Loss: 0.04295, Acc: 99.00% | Test Loss: 0.03486, Test Acc: 99.50%
Epoch: 60 | Loss: 0.03910, Acc: 99.00% | Test Loss: 0.03083, Test Acc: 99.50%
Epoch: 70 | Loss: 0.03643, Acc: 99.00% | Test Loss: 0.02799, Test Acc: 99.50%
Epoch: 80 | Loss: 0.03448, Acc: 99.00% | Test Loss: 0.02587, Test Acc: 99.50%
Epoch: 90 | Loss: 0.03300, Acc: 99.12% | Test Loss: 0.02423, Test Acc: 99.50%  `}
        />

        <h2>
          8.6 Making and evaluating predictions with a PyTorch multi-class model
        </h2>

        <p>It looks like our trained model is performing pretty well.</p>
        <p>
          But to make sure of this, let's make some predictions and visualize
          them.
        </p>

        <CodeIOBlock
          inputCode={`# Make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

# View the first 10 predictions
y_logits[:10]  `}
          outputCode={`tensor([[  4.3377,  10.3539, -14.8948,  -9.7642],
        [  5.0142, -12.0371,   3.3860,  10.6699],
        [ -5.5885, -13.3448,  20.9894,  12.7711],
        [  1.8400,   7.5599,  -8.6016,  -6.9942],
        [  8.0727,   3.2906, -14.5998,  -3.6186],
        [  5.5844, -14.9521,   5.0168,  13.2890],
        [ -5.9739, -10.1913,  18.8655,   9.9179],
        [  7.0755,  -0.7601,  -9.5531,   0.1736],
        [ -5.5918, -18.5990,  25.5309,  17.5799],
        [  7.3142,   0.7197, -11.2017,  -1.2011]], device='cuda:0')  `}
        />

        <p>
          It looks like our model's predictions are still in the raw logit form.
        </p>
        <p>
          To evaluate them properly, we need to convert them into the same
          format as our labels (<code>y_blob_test</code>), which are in integer
          form.
        </p>
        <p>
          We can do this by first using <code>torch.softmax()</code> to convert
          the logits into prediction probabilities, and then using{" "}
          <code>torch.argmax()</code> to get the predicted class labels for each
          sample.
        </p>

        <div className="note">
          <strong>Note:</strong> It's possible to skip the torch.softmax()
          function and go straight from predicted logits -) predicted labels by
          calling torch.argmax() directly on the logits. For example, y_preds =
          torch.argmax(y_logits, dim=1), this saves a computation step (no
          torch.softmax()) but results in no prediction probabilities being
          available to use.
        </div>

        <CodeIOBlock
          inputCode={`# Turn predicted logits in prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)

# Turn prediction probabilities into prediction labels
y_preds = y_pred_probs.argmax(dim=1)

# Compare first 10 model preds and test labels
print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")  `}
          outputCode={`Predictions: tensor([1, 3, 2, 1, 0, 3, 2, 0, 2, 0], device='cuda:0')
Labels: tensor([1, 3, 2, 1, 0, 3, 2, 0, 2, 0], device='cuda:0')
Test accuracy: 99.5%  `}
        />

        <p>
          Great! Now our model's predictions are in the same format as our test
          labels.
        </p>
        <p>
          Let's visualize them using <code>plot_decision_boundary()</code>.
          Since our data is on the GPU, we'll need to move it to the CPU for use
          with matplotlib (don't worry, <code>plot_decision_boundary()</code>{" "}
          does this automatically for us).
        </p>

        <CodeIOBlock
          inputCode={`plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)  `}
        />

        <img src={Test} className="centered-image" />

        <h2>9. More classification evaluation metrics</h2>

        <p>
          So far, we've only covered a couple of ways of evaluating a
          classification model (accuracy, loss, and visualizing predictions).
        </p>
        <p>
          These are some of the most common methods you'll come across and are a
          good starting point.
        </p>
        <p>
          However, you may want to evaluate your classification model using more
          metrics such as the following:
        </p>

        <table>
          <thead>
            <tr>
              <th>Metric name/Evaluation method</th>
              <th>Definition</th>
              <th>Code</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Accuracy</td>
              <td>
                Out of 100 predictions, how many does your model get correct?
                E.g. 95% accuracy means it gets 95/100 predictions correct.
              </td>
              <td>
                torchmetrics.Accuracy() or sklearn.metrics.accuracy_score()
              </td>
            </tr>
            <tr>
              <td>Precision</td>
              <td>
                Proportion of true positives over total number of samples.
                Higher precision leads to less false positives (model predicts 1
                when it should've been 0).
              </td>
              <td>
                torchmetrics.Precision() or sklearn.metrics.precision_score()
              </td>
            </tr>
            <tr>
              <td>Recall</td>
              <td>
                Proportion of true positives over total number of true positives
                and false negatives (model predicts 0 when it should've been 1).
                Higher recall leads to less false negatives.
              </td>
              <td>torchmetrics.Recall() or sklearn.metrics.recall_score()</td>
            </tr>
            <tr>
              <td>F1-score</td>
              <td>
                Combines precision and recall into one metric. 1 is best, 0 is
                worst.
              </td>
              <td>torchmetrics.F1Score() or sklearn.metrics.f1_score()</td>
            </tr>
            <tr>
              <td>Confusion matrix</td>
              <td>
                Compares the predicted values with the true values in a tabular
                way, if 100% correct, all values in the matrix will be top left
                to bottom right (diagonal line).
              </td>
              <td>
                torchmetrics.ConfusionMatrix or
                sklearn.metrics.plot_confusion_matrix()
              </td>
            </tr>
            <tr>
              <td>Classification report</td>
              <td>
                Collection of some of the main classification metrics such as
                precision, recall and f1-score.
              </td>
              <td>sklearn.metrics.classification_report()</td>
            </tr>
          </tbody>
        </table>

        <p>
          Scikit-Learn (a popular and world-class machine learning library) has
          many implementations of the above metrics and if you're looking for a
          PyTorch-like version, check out <strong>TorchMetrics</strong>,
          especially the TorchMetrics classification section.
        </p>

        <p>
          Let's try the <code>torchmetrics.Accuracy</code> metric out.
        </p>

        <CodeIOBlock
          inputCode={`try:
    from torchmetrics import Accuracy
except:
    !pip install torchmetrics==0.9.3 # this is the version we're using in this notebook (later versions exist here: https://torchmetrics.readthedocs.io/en/stable/generated/CHANGELOG.html#changelog)
    from torchmetrics import Accuracy

# Setup metric and make sure it's on the target device
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)

# Calculate accuracy
torchmetrics_accuracy(y_preds, y_blob_test)  `}
          outputCode={`tensor(0.9950, device='cuda:0')  `}
        />

        <h1> The End!</h1>
      </section>
    </div>
  );
};

export default neuralNetworksClassification;
