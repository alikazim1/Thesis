import React from "react";
import linePlot from "./Img/workflow.png";
import CodeIOBlock from "../Fundamentals/CodeIOBlock";
import "./workflowFundamentals.css";
import dataPreprocessing from "./Img/dataPreprocessing.png";
import { data } from "autoprefixer";
import Plot from "./Img/plot.png";
import PredPlot from "./Img/predPlot.png";
import LinearModel from "./Img/linearModel.png";
import plotPred from "./Img/plot_pred.png";
import ModelBuild from "./Img/modelBuild.png";
import PlotPrediction from "./Img/plotPrediction.png";
import GraphPred from "./Img/graphPred.png";
import Loss from "./Img/loss.png";
import TrainingLoop from "./Img/trainingLoop.png";
import PyTorchTestingLoop from "./Img/pytorchTestingLoop.png";
import TrainingLoss from "./Img/trainingLoss.png";

const WorkflowFundamentals = () => {
  return (
    <div className="content">
      <h1 className="page-title">02. Workflow Fundamentals</h1>

      <section>
        <h2>The Essence of Machine Learning</h2>
        <p>
          The core idea of machine learning and deep learning is to take past
          data, build an algorithm (like a neural network) that identifies
          patterns in that data, and use those learned patterns to make
          predictions about new, unseen data.
        </p>
      </section>

      <img
        src={linePlot}
        alt="Simple ML workflow with data and prediction"
        className="centered-image"
      />

      <section>
        <p>
          We'll use this workflow to predict a simple straight line, but the
          workflow steps can be repeated and adapted depending on the problem
          you're working on.
        </p>

        <p>Specifically, we're going to cover:</p>

        <table className="workflow-table">
          <thead>
            <tr>
              <th>Topic</th>
              <th>Contents</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1. Getting data ready</td>
              <td>
                Data can be almost anything but to get started we're going to
                create a simple straight line
              </td>
            </tr>
            <tr>
              <td>2. Building a model</td>
              <td>
                We'll create a model to learn patterns in the data, choose a
                loss function, optimizer, and build a training loop.
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
              <td>5. Saving and loading a model</td>
              <td>
                You may want to use your model elsewhere, or come back to it
                later, here we'll cover that.
              </td>
            </tr>
            <tr>
              <td>6. Putting it all together</td>
              <td>Let's take all of the above and combine it.</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section>
        <p>
          <strong>Where can you get help?</strong> There's also the{" "}
          <a
            href="https://discuss.pytorch.org/"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "#4a90e2", textDecoration: "underline" }}
          >
            PyTorch Developer Forums
          </a>
          , a very helpful place for all things PyTorch.
        </p>

        <p>
          Let's start by putting what we're covering into a dictionary to
          reference later.
        </p>

        <div className="dictionary-block">
          <p>
            <strong>what_were_covering =</strong>
          </p>
          <ul>
            <li>1: "data (prepare and load)"</li>
            <li>2: "build model"</li>
            <li>3: "fitting the model to data (training)"</li>
            <li>4: "making predictions and evaluating a model (inference)"</li>
            <li>5: "saving and loading a model"</li>
            <li>6: "putting it all together"</li>
          </ul>
        </div>

        <p>Now let's import what we'll need for this module.</p>

        <p>
          We're going to get <code>torch</code>, <code>torch.nn</code> (
          <strong>nn</strong> stands for neural network and this package
          contains the building blocks for creating neural networks in PyTorch),
          and <code>matplotlib</code>.
        </p>

        <CodeIOBlock
          inputCode={`import torch
from torch import nn  # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
torch.__version__`}
          outputCode={`'1.12.1+cu113'`}
        />
      </section>

      <section>
        <h2>1. Data (preparing and loading)</h2>
        <p>
          I want to stress that <strong>"data"</strong> in machine learning can
          be almost anything you can imagine. A table of numbers (like a big
          Excel spreadsheet), images of any kind, videos (YouTube has lots of
          data!), audio files like songs or podcasts, protein structures, text,
          and more.
        </p>

        <img src={dataPreprocessing} className="centered-image" />
        <p>Machine learning can be thought of as a two-step process:</p>
        <ol>
          <li>
            Convert your data, no matter what form it takes, into numerical
            representations.
          </li>
          <li>
            Pick or create a model that can learn these representations as
            accurately as possible.
          </li>
        </ol>
        <p>Sometimes, these two steps can happen simultaneously.</p>
        <p>But what if you don’t have data?</p>
        <p>
          That’s the situation we’re in right now—no data. But don’t worry, we
          can generate some.
        </p>
        <p>Let's create our data as a simple straight line.</p>
        <p>
          We’ll use linear regression to generate the data with known parameters
          (things the model can learn), and then we’ll apply PyTorch to build a
          model that estimates these parameters using gradient descent.
        </p>
        <p>
          Don’t worry if some of these terms aren’t clear yet. We’ll see them in
          action, and I’ll provide additional resources below where you can
          learn more.
        </p>
        <CodeIOBlock
          inputCode={`# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]  `}
          outputCode={`(tensor([[0.0000],
         [0.0200],
         [0.0400],
         [0.0600],
         [0.0800],
         [0.1000],
         [0.1200],
         [0.1400],
         [0.1600],
         [0.1800]]),
 tensor([[0.3000],
         [0.3140],
         [0.3280],
         [0.3420],
         [0.3560],
         [0.3700],
         [0.3840],
         [0.3980],
         [0.4120],
         [0.4260]]))  `}
        />
        <p>
          Great! Now it's time to build a model that can learn the connection
          between <code>X</code> (input features) and <code>y</code> (target
          labels).
        </p>
        <h2>1. Split data into training and test sets</h2>
        <p>
          We've got some data. But before building a model, we need to split it
          into separate parts.
        </p>
        <p>
          One of the most important steps in a machine learning project is
          dividing the dataset into a <strong>training set</strong> and a{" "}
          <strong>test set</strong> (sometimes also a{" "}
          <strong>validation set</strong> if needed).
        </p>
        <p>Each split has its own purpose:</p>
        <table className="workflow-table">
          <thead>
            <tr>
              <th>Split</th>
              <th>Purpose</th>
              <th>Amount of total data</th>
              <th>How often is it used?</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Training set</td>
              <td>
                The model learns from this data (like the course materials you
                study during the semester).
              </td>
              <td>~60–80%</td>
              <td>Always</td>
            </tr>
            <tr>
              <td>Validation set</td>
              <td>
                The model gets tuned on this data (like the practice exam you
                take before the final exam).
              </td>
              <td>~10–20%</td>
              <td>Often but not always</td>
            </tr>
            <tr>
              <td>Testing set</td>
              <td>
                The model gets evaluated on this data to test what it has
                learned (like the final exam you take at the end of the
                semester).
              </td>
              <td>~10–20%</td>
              <td>Always</td>
            </tr>
          </tbody>
        </table>
        <br></br>
        <p>
          For now, we'll use just a <strong>training set</strong> and a{" "}
          <strong>test set</strong>. This means our model will have one portion
          of the data to learn from and another to be evaluated on.
        </p>
        <p>
          We can create these sets by splitting our <code>X</code> and{" "}
          <code>y</code> tensors accordingly.
        </p>

        <div className="note">
          <strong>Note:</strong> When dealing with real-world data, this step is
          typically done right at the start of a project (the test set should
          always be kept separate from all other data). We want our model to
          learn from training data and then evaluate it on test data to get an
          indication of how well it generalizes to unseen examples.
        </div>

        <CodeIOBlock
          inputCode={`# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)  `}
          outputCode={`(40, 40, 10, 10)  `}
        />

        <p>
          Wonderful, we've got <strong>40 samples</strong> for training (
          <code>X_train</code> & <code>y_train</code>) and{" "}
          <strong>10 samples</strong> for testing (<code>X_test</code> &{" "}
          <code>y_test</code>).
        </p>

        <p>
          The model we're going to create will try to learn the relationship
          between <code>X_train</code> and <code>y_train</code>, and then we’ll
          evaluate how well it learned by testing it on <code>X_test</code> and{" "}
          <code>y_test</code>.
        </p>

        <p>But right now, our data is just numbers on a page.</p>

        <p>Let's create a function to visualize it.</p>

        <CodeIOBlock
          inputCode={`def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});
  plot_predictions();  `}
        />

        <img src={Plot} className="centered-image" />
        <p>Epic!</p>
        <p>
          Now instead of just being numbers on a page, our data forms a straight
          line.
        </p>
        <div className="note">
          <strong>Note:</strong> Here's a great rule to follow when working with
          data: <em>"visualize, visualize, visualize!"</em>
        </div>
        <p>
          Whenever you're turning data into numbers, try to visualize it if you
          can — it really helps with understanding what's going on.
        </p>
        <p>
          Machines work best with numbers. And while we humans can handle
          numbers too, we usually understand things better when we can see them.
        </p>

        <h2>2. Build Model</h2>
        <p>
          Now that we've got some data, let's build a model to use the blue dots
          to predict the green dots.
        </p>
        <p>We're going to jump right in.</p>
        <p>We'll write the code first and then explain everything.</p>
        <p>
          Let's replicate a standard linear regression model using pure PyTorch.
        </p>
        <CodeIOBlock
          inputCode={`# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)  `}
        />

        <p>
          Alright, there's a fair bit going on above, but let's break it down
          bit by bit.
        </p>

        <div className="resource">
          <strong>Resource:</strong> We'll be using Python classes to create
          bits and pieces for building neural networks. If you're unfamiliar
          with Python class notation, I'd recommend reading
          <a
            href="https://realpython.com/python3-object-oriented-programming/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Real Python's Object-Oriented Programming in Python 3 guide
          </a>{" "}
          a few times.
        </div>

        <h2>PyTorch Model Building Essentials</h2>
        <p>
          PyTorch has four main modules for building neural networks. These are:
          <code>torch.nn</code>, <code>torch.optim</code>,{" "}
          <code>torch.utils.data.Dataset</code>, and{" "}
          <code>torch.utils.data.DataLoader</code>. For now, we'll focus on the
          first two and cover the others later (though their purpose might be
          easy to guess).
        </p>

        <table>
          <thead>
            <tr>
              <th>PyTorch Module</th>
              <th>What Does It Do?</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>torch.nn</td>
              <td>
                Contains all of the building blocks for computational graphs
                (essentially a series of computations executed in a particular
                way).
              </td>
            </tr>
            <tr>
              <td>torch.nn.Parameter</td>
              <td>
                Stores tensors that can be used with nn.Module. If{" "}
                <code>requires_grad=True</code>, gradients are calculated
                automatically for updating model parameters via gradient
                descent, often referred to as "autograd".
              </td>
            </tr>
            <tr>
              <td>torch.nn.Module</td>
              <td>
                The base class for all neural network modules. All building
                blocks for neural networks are subclasses of this. Models should
                subclass <code>nn.Module</code> and implement a{" "}
                <code>forward()</code> method.
              </td>
            </tr>
            <tr>
              <td>torch.optim</td>
              <td>
                Contains various optimization algorithms that help model
                parameters stored in <code>nn.Parameter</code> adjust to improve
                gradient descent and reduce loss.
              </td>
            </tr>
            <tr>
              <td>def forward()</td>
              <td>
                All <code>nn.Module</code> subclasses require a{" "}
                <code>forward()</code> method, which defines the computation
                performed on the data passed to the module (e.g., the linear
                regression formula).
              </td>
            </tr>
          </tbody>
        </table>

        <p>If the previous explanation sounds complex, think of it this way:</p>
        <ul>
          <li>
            <strong>nn.Module</strong> contains the larger building blocks
            (layers) of the neural network.
          </li>
          <li>
            <strong>nn.Parameter</strong> holds the smaller parameters like
            weights and biases, which are used within <code>nn.Module</code>.
          </li>
          <li>
            <strong>forward()</strong> defines how the larger blocks perform
            calculations on input data (tensors) inside <code>nn.Module</code>.
          </li>
          <li>
            <strong>torch.optim</strong> provides optimization methods to
            improve the parameters in <code>nn.Parameter</code>, helping the
            model better represent input data.
          </li>
        </ul>
        <img src={ModelBuild} className="centered-image" />
        <p>
          The basic building blocks of creating a PyTorch model involve
          subclassing <code>nn.Module</code>. For objects that subclass{" "}
          <code>nn.Module</code>, the <code>forward()</code> method must be
          defined.
        </p>

        <div className="resource">
          <p>
            <strong>Resource:</strong> See more of these essential modules and
            their use cases in the
            <a
              href="https://pytorch.org/tutorials/beginner/nn_tutorial.html"
              target="_blank"
            >
              PyTorch Cheat Sheet
            </a>
            .
          </p>
        </div>
        <p>
          Now that we've covered the basics, let's create a model instance using
          the class we've made and check its parameters using{" "}
          <code>.parameters()</code>.
        </p>

        <CodeIOBlock
          inputCode={`# Set manual seed since nn.Parameter are randomly initialized
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Check the nn.Parameter(s) within the nn.Module subclass we created
list(model_0.parameters())  `}
          outputCode={`[Parameter containing:
 tensor([0.3367], requires_grad=True),
 Parameter containing:
 tensor([0.1288], requires_grad=True)]  `}
        />

        <p>
          We can also check the state (what the model contains) of the model
          using <code>.state_dict()</code>.
        </p>

        <CodeIOBlock
          inputCode={`# List named parameters 
model_0.state_dict()  `}
          outputCode={`OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])  `}
        />

        <p>
          Notice how the values for weights and bias from{" "}
          <code>model_0.state_dict()</code> appear as random float tensors? This
          is because we initialized them using <code>torch.randn()</code>.
        </p>
        <p>
          Essentially, we want to start with random parameters and update them
          to better fit our data, eventually matching the hardcoded weight and
          bias values we set when creating our straight line data.
        </p>

        <div className="resource">
          <p>
            <b>Exercise</b>: Try changing the torch.manual_seed() value two
            cells above, see what happens to the weights and bias values.
          </p>
        </div>

        <p>
          Since our model starts with random values, it will have poor
          predictive power initially.
        </p>
        <h2>Making Predictions Using torch.inference_mode()</h2>
        <p>
          To check the model's predictions, we can pass the test data{" "}
          <code>X_test</code> and see how closely it predicts{" "}
          <code>y_test</code>.
        </p>
        <p>
          When we pass data to our model, it goes through the{" "}
          <code>forward()</code> method and produces a result based on the
          computations we've defined.
        </p>
        <p>Now, let's make some predictions.</p>

        <CodeIOBlock
          inputCode={`# Make predictions with model
with torch.inference_mode(): 
    y_preds = model_0(X_test)

# Note: in older PyTorch code you might also see torch.no_grad()
# with torch.no_grad():
#   y_preds = model_0(X_test)  `}
        />

        <p>
          <code>torch.inference_mode()</code> is a context manager used for
          making predictions with a model. It disables unnecessary features like
          gradient tracking, which is only needed during training. This makes
          the forward pass faster, improving performance when making
          predictions.
        </p>

        <div className="note">
          <strong>Note:</strong> In older PyTorch code, you may also see
          torch.no_grad() being used for inference. While torch.inference_mode()
          and torch.no_grad() do similar things, torch.inference_mode() is
          newer, potentially faster and preferred. See this Tweet from PyTorch
          for more.
        </div>
        <p>Let's take a look at the predictions we've made.</p>

        <CodeIOBlock
          inputCode={`# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")  `}
          outputCode={`Number of testing samples: 10
Number of predictions made: 10
Predicted values:
tensor([[0.3982],
        [0.4049],
        [0.4116],
        [0.4184],
        [0.4251],
        [0.4318],
        [0.4386],
        [0.4453],
        [0.4520],
        [0.4588]])  `}
        />

        <p>
          Each testing sample has one prediction value because our data maps one
          X value to one y value. However, machine learning models can handle
          more complex relationships, such as multiple X values mapping to
          multiple y values. Let's visualize our predictions using the{" "}
          <code>plot_predictions()</code> function.
        </p>

        <CodeIOBlock inputCode={`plot_predictions(predictions=y_preds)  `} />

        <img src={PlotPrediction} className="centered-image" />

        <CodeIOBlock
          inputCode={`y_test - y_preds  `}
          outputCode={`tensor([[0.4618],
        [0.4691],
        [0.4764],
        [0.4836],
        [0.4909],
        [0.4982],
        [0.5054],
        [0.5127],
        [0.5200],
        [0.5272]])  `}
        />

        <p>
          The predictions don't look great, which is expected since our model is
          using random parameters. It hasn't learned from the blue dots to
          predict the green dots. Now it's time to improve this.
        </p>

        <h2>3. Train Model</h2>
        <p>
          Currently, our model makes predictions using random parameters,
          essentially guessing. To improve this, we need to update its internal
          parameters (weights and biases) to better represent the data. While we
          could hard-code the values, it's more interesting to let the model
          learn them itself.
        </p>
        <h2>Creating a Loss Function and Optimizer in PyTorch</h2>
        <p>
          To update our model's parameters automatically, we need to add a loss
          function and an optimizer to our setup.
        </p>
        <p>These are a loss function and an optimizer.</p>
        <p>Here are their roles:</p>
        <table>
          <thead>
            <tr>
              <th>Function</th>
              <th>What Does It Do?</th>
              <th>Where Does It Live in PyTorch?</th>
              <th>Common Values</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Loss function</td>
              <td>
                Measures how wrong your model's predictions (e.g.,{" "}
                <code>y_preds</code>) are compared to the truth labels (e.g.,{" "}
                <code>y_test</code>). Lower values are better.
              </td>
              <td>
                PyTorch has plenty of built-in loss functions in{" "}
                <code>torch.nn</code>.
              </td>
              <td>
                Mean absolute error (MAE) for regression problems (
                <code>torch.nn.L1Loss()</code>). Binary cross entropy for binary
                classification problems (<code>torch.nn.BCELoss()</code>).
              </td>
            </tr>
            <tr>
              <td>Optimizer</td>
              <td>
                Tells your model how to update its internal parameters to best
                lower the loss.
              </td>
              <td>
                You can find various optimization function implementations in{" "}
                <code>torch.optim</code>.
              </td>
              <td>
                Stochastic gradient descent (SGD) (
                <code>torch.optim.SGD()</code>). Adam optimizer (
                <code>torch.optim.Adam()</code>).
              </td>
            </tr>
          </tbody>
        </table>
        <p>
          Let's create a loss function and optimizer to improve our model. The
          choice of these depends on the problem type. Common choices include
          the SGD or Adam optimizer, and the MAE loss function for regression
          tasks or binary cross-entropy for classification tasks. Since we're
          predicting numbers, we'll use MAE (available as{" "}
          <code>torch.nn.L1Loss()</code> in PyTorch).
        </p>
        <img src={Loss} className="centered-image" />
        <br />
        <p>
          Mean Absolute Error (MAE), implemented in PyTorch as{" "}
          <code>torch.nn.L1Loss</code>, calculates the average absolute
          difference between predictions and actual values. We'll use Stochastic
          Gradient Descent (SGD) as our optimizer, defined as{" "}
          <code>torch.optim.SGD(params, lr)</code>. Here, <code>params</code>{" "}
          are the model parameters to optimize (like weights and biases), and{" "}
          <code>lr</code> is the learning rate, which determines how quickly the
          optimizer updates these parameters. A higher learning rate can lead to
          larger updates, while a lower rate results in smaller updates. Common
          initial learning rates are 0.01, 0.001, or 0.0001, and these can be
          adjusted over time.
        </p>

        <CodeIOBlock
          inputCode={`# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))  `}
        />

        <h2>Creating an optimization loop in PyTorch</h2>
        <p>
          Now that we have a loss function and an optimizer, it's time to create
          a training loop and a testing loop. The training loop involves the
          model learning from the training data by iterating through each
          sample. The testing loop evaluates the model's performance on unseen
          test data, checking how well it has learned the patterns.
        </p>

        <h2>PyTorch training loop</h2>
        <p>For the training loop, we'll build the following steps:</p>
        <table>
          <thead>
            <tr>
              <th>Number</th>
              <th>Step Name</th>
              <th>What Does It Do?</th>
              <th>Code Example</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>Forward pass</td>
              <td>
                The model goes through all of the training data once, performing
                its <code>forward()</code> function calculations.
              </td>
              <td>
                <code>model(x_train)</code>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>Calculate the loss</td>
              <td>
                The model's outputs (predictions) are compared to the ground
                truth and evaluated to see how wrong they are.
              </td>
              <td>
                <code>loss = loss_fn(y_pred, y_train)</code>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>Zero gradients</td>
              <td>
                The optimizer's gradients are set to zero (they are accumulated
                by default) so they can be recalculated for the specific
                training step.
              </td>
              <td>
                <code>optimizer.zero_grad()</code>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Perform backpropagation on the loss</td>
              <td>
                Computes the gradient of the loss with respect to every model
                parameter to be updated (each parameter with{" "}
                <code>requires_grad=True</code>). This is known as
                backpropagation, hence "backwards".
              </td>
              <td>
                <code>loss.backward()</code>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>Update the optimizer (gradient descent)</td>
              <td>
                Update the parameters with <code>requires_grad=True</code> with
                respect to the loss gradients to improve them.
              </td>
              <td>
                <code>optimizer.step()</code>
              </td>
            </tr>
          </tbody>
        </table>
        <br></br>
        <img src={TrainingLoop} className="centered-image" />

        <div className="note">
          <strong>Note:</strong>
          <p>
            <strong>Note:</strong> The above is just one example of how the
            steps could be ordered or described. With experience, you'll find
            that PyTorch training loops can be quite flexible.
          </p>
          <p>
            Regarding the ordering of things, the above is a good default order,
            but you may encounter slightly different orders. Some general rules
            to follow:
          </p>
          <ul>
            <li>
              Calculate the loss (<code>loss = ...</code>) before performing
              backpropagation on it (<code>loss.backward()</code>).
            </li>
            <li>
              Zero gradients (<code>optimizer.zero_grad()</code>) before
              computing the gradients of the loss with respect to every model
              parameter (<code>loss.backward()</code>).
            </li>
            <li>
              Step the optimizer (<code>optimizer.step()</code>) after
              performing backpropagation on the loss (
              <code>loss.backward()</code>).
            </li>
          </ul>
        </div>

        <h2>PyTorch testing loop</h2>
        <p>
          As for the testing loop (evaluating our model), the typical steps
          include:
        </p>
        <table>
          <thead>
            <tr>
              <th>Number</th>
              <th>Step name</th>
              <th>What does it do?</th>
              <th>Code example</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>Forward pass</td>
              <td>
                The model goes through all of the testing data once, performing
                its forward() function calculations.
              </td>
              <td>
                <code>model(x_test)</code>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>Calculate the loss</td>
              <td>
                The model's outputs (predictions) are compared to the ground
                truth and evaluated to see how wrong they are.
              </td>
              <td>
                <code>loss = loss_fn(y_pred, y_test)</code>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>Calulate evaluation metrics (optional)</td>
              <td>
                Alongside the loss value you may want to calculate other
                evaluation metrics such as accuracy on the test set.
              </td>
              <td></td>
            </tr>
          </tbody>
        </table>
        <p>
          The testing loop doesn't include backpropagation (
          <code>loss.backward()</code>) or updating the optimizer (
          <code>optimizer.step()</code>). This is because the model's parameters
          are already set during training. In testing, we only need the output
          from the forward pass.
        </p>

        <img src={PyTorchTestingLoop} className="centered-image" />
        <br></br>
        <p>
          Let's combine everything and train our model for 100 epochs,
          evaluating it every 10 epochs.
        </p>

        <CodeIOBlock
          inputCode={`torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")  `}
          outputCode={`Epoch: 0 | MAE Train Loss: 0.31288138031959534 | MAE Test Loss: 0.48106518387794495 
Epoch: 10 | MAE Train Loss: 0.1976713240146637 | MAE Test Loss: 0.3463551998138428 
Epoch: 20 | MAE Train Loss: 0.08908725529909134 | MAE Test Loss: 0.21729660034179688 
Epoch: 30 | MAE Train Loss: 0.053148526698350906 | MAE Test Loss: 0.14464017748832703 
Epoch: 40 | MAE Train Loss: 0.04543796554207802 | MAE Test Loss: 0.11360953003168106 
Epoch: 50 | MAE Train Loss: 0.04167863354086876 | MAE Test Loss: 0.09919948130846024 
Epoch: 60 | MAE Train Loss: 0.03818932920694351 | MAE Test Loss: 0.08886633068323135 
Epoch: 70 | MAE Train Loss: 0.03476089984178543 | MAE Test Loss: 0.0805937647819519 
Epoch: 80 | MAE Train Loss: 0.03132382780313492 | MAE Test Loss: 0.07232122868299484 
Epoch: 90 | MAE Train Loss: 0.02788739837706089 | MAE Test Loss: 0.06473556160926819   `}
        />
        <p>
          Oh would you look at that! Looks like our loss is going down with
          every epoch, let's plot it to find out.
        </p>
        <CodeIOBlock
          inputCode={`# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();  `}
        />

        <img src={TrainingLoss} className="centered-image" />

        <p>
          The loss curves show the loss decreasing over time. Lower loss means
          the model is less wrong. This happened because our loss function and
          optimizer updated the model's internal parameters (weights and bias)
          to better match the data's patterns. Let's check the model's{" "}
          <code>.state_dict()</code> to see how close it got to the original
          weights and bias values.
        </p>

        <CodeIOBlock
          inputCode={`# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")  `}
          outputCode={`The model learned the following values for weights and bias:
OrderedDict([('weights', tensor([0.5784])), ('bias', tensor([0.3513]))])

And the original values for weights and bias are:
weights: 0.7, bias: 0.3  `}
        />
        <p>
          Our model successfully approximated the original weight and bias
          values. It would likely get even closer with more training.
        </p>

        <div className="note">
          <strong>Exercise:</strong>Try changing the epochs value above to 200,
          what happens to the loss curves and the weights and bias parameter
          values of the model?
        </div>

        <p>
          The model might not guess the values perfectly, especially with
          complex datasets, but a close approximation is often enough to achieve
          great results. This is the core idea of machine learning: instead of
          manually finding ideal values, we train models to discover them
          programmatically.
        </p>

        <h2>4. Making predictions with a trained PyTorch model (inference) </h2>

        <p>
          To make predictions with a trained PyTorch model, remember these three
          steps:
        </p>
        <ol>
          <li>
            <strong>Set the Model to Evaluation Mode</strong>: Use{" "}
            <code>model.eval()</code>.
          </li>
          <li>
            <strong>Use Inference Mode</strong>: Wrap predictions with{" "}
            <code>with torch.inference_mode():</code>.
          </li>
          <li>
            <strong>Use the Same Device</strong>: Ensure both data and model are
            on the same device (e.g., both on GPU or CPU).
          </li>
        </ol>
        <p>
          The first two steps disable unnecessary training calculations for
          faster inference, and the third prevents cross-device errors.
        </p>

        <CodeIOBlock
          inputCode={`# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)
y_preds  `}
          outputCode={`tensor([[0.8141],
        [0.8256],
        [0.8372],
        [0.8488],
        [0.8603],
        [0.8719],
        [0.8835],
        [0.8950],
        [0.9066],
        [0.9182]])  `}
        />
        <p>
          Now that we've made predictions with our trained model, let's see how
          they look.
        </p>

        <CodeIOBlock inputCode={`plot_predictions(predictions=y_preds)  `} />
        <img src={GraphPred} className="centered-image" />

        <h2>5. Saving and loading a PyTorch model</h2>
        <p>
          If you've trained a PyTorch model, you might want to save it for use
          in another application or to save your progress. There are three main
          methods for saving and loading PyTorch models.
        </p>
        <table>
          <thead>
            <tr>
              <th>PyTorch method</th>
              <th>What does it do?</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>torch.save</td>
              <td>
                Saves a serialized object to disk using Python's pickle utility.
                Models, tensors and various other Python objects like
                dictionaries can be saved using torch.save.
              </td>
            </tr>
            <tr>
              <td>torch.load</td>
              <td>
                Uses pickle's unpickling features to deserialize and load
                pickled Python object files (like models, tensors or
                dictionaries) into memory. You can also set which device to load
                the object to (CPU, GPU etc).
              </td>
            </tr>
            <tr>
              <td>torch.nn.Module.load_state_dict</td>
              <td>
                Loads a model's parameter dictionary (model.state_dict()) using
                a saved state_dict() object.
              </td>
            </tr>
          </tbody>
        </table>
        <div className="note">
          <strong>Note:</strong> As stated in Python's pickle documentation, the
          pickle module is not secure. That means you should only ever unpickle
          (load) data you trust. That goes for loading PyTorch models as well.
          Only ever use saved PyTorch models from sources you trust.
        </div>

        <h2>Saving a PyTorch model's state_dict()</h2>
        <p>
          The recommended way to save and load a model for inference is by using
          the model's <code>state_dict()</code>. Here's how you can do it:
        </p>
        <ol>
          <li>
            Create a directory for saving models using Python's{" "}
            <code>pathlib</code> module.
          </li>
          <li>Define a file path for saving the model.</li>
          <li>
            Use <code>torch.save()</code> to save the model's{" "}
            <code>state_dict()</code> to the specified file.
          </li>
        </ol>

        <div className="note">
          <strong>Note:</strong>It's common convention for PyTorch saved models
          or objects to end with .pt or .pth, like saved_model_01.pth.
        </div>

        <CodeIOBlock
          inputCode={`from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)   `}
          outputCode={`Saving model to: models/01_pytorch_workflow_model_0.pth  `}
        />

        <CodeIOBlock
          inputCode={`# Check the saved file path
!ls -l models/01_pytorch_workflow_model_0.pth  `}
          outputCode={`
-rw-rw-r-- 1 daniel daniel 1063 Nov 10 16:07 models/01_pytorch_workflow_model_0.pth  `}
        />

        <h2>Loading a saved PyTorch model's state_dict()</h2>
        <p>
          Now that we have a saved model's <code>state_dict()</code> at{" "}
          <code>models/01_pytorch_workflow_model_0.pth</code>, we can load it
          using <code>torch.nn.Module.load_state_dict(torch.load(f))</code>,
          where <code>f</code> is the file path. We use{" "}
          <code>torch.load()</code> inside <code>load_state_dict()</code>{" "}
          because we only saved the model's parameters, not the entire model. We
          load the <code>state_dict()</code> and pass it to a new model
          instance.
        </p>
        <p>
          We're using the flexible method of saving and loading just the{" "}
          <code>state_dict()</code>, which is a dictionary of model parameters.
          Let's test it by creating another instance of{" "}
          <code>LinearRegressionModel()</code>, a subclass of{" "}
          <code>torch.nn.Module</code>, which has the built-in{" "}
          <code>load_state_dict()</code> method.
        </p>

        <CodeIOBlock
          inputCode={`# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))  `}
          outputCode={`<All keys matched successfully>  `}
        />

        <p>
          Now, let's test our loaded model by making predictions on the test
          data. Remember the rules for performing inference with PyTorch models?
        </p>

        <CodeIOBlock
          inputCode={`# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model  `}
        />

        <p>
          Now we've made some predictions with the loaded model, let's see if
          they're the same as the previous predictions.
        </p>

        <CodeIOBlock
          inputCode={`# Compare previous model predictions with loaded model predictions (these should be the same)
y_preds == loaded_model_preds  `}
          outputCode={`tensor([[True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True]])  `}
        />

        <p>
          The loaded model's predictions match the previous model's predictions,
          showing that our model is saving and loading correctly.
        </p>

        <div className="note">
          <strong>Note:</strong>
          There are more methods to save and load PyTorch models but I'll leave
          these for extra-curriculum and further reading. See the PyTorch guide
          for saving and loading models for more.
        </div>

        <h2>6. Putting it all together</h2>
        <p>
          We've covered a lot so far. Now, let's put everything together and
          make our code device-agnostic, so it uses a GPU if available or
          defaults to the CPU. We'll start by importing the necessary libraries.
        </p>

        <div className="note">
          <strong>Note:</strong>If you're using Google Colab, to setup a GPU, go
          to Runtime -) Change runtime type -) Hardware acceleration -) GPU. If
          you do this, it will reset the Colab runtime and you will lose saved
          variables.
        </div>

        <CodeIOBlock
          inputCode={`# Import PyTorch and matplotlib
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
torch.__version__  `}
          outputCode={`Using device: cuda  `}
        />

        <p>
          Let's make our code device-agnostic by setting{" "}
          <code>device="cuda"</code> if available, otherwise defaulting to{" "}
          <code>device="cpu"</code>.
        </p>

        <CodeIOBlock
          inputCode={`# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  `}
          outputCode={`Using device: cuda  `}
        />
        <p>
          If you have access to a GPU, it should be used for computations.
          Otherwise, a CPU will be used. This is fine for small datasets but
          will be slower for larger ones.
        </p>

        <h2>6.1 Data</h2>
        <p>
          Let's create some data. First, we'll set some weight and bias values.
          Then, we'll generate X values from 0 to 1. Finally, we'll use these X
          values along with the weight and bias to calculate y using the linear
          regression formula: <code>y = weight * X + bias</code>.
        </p>

        <CodeIOBlock
          inputCode={`# Create weight and bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) # without unsqueeze, errors will happen later on (shapes within linear layers)
y = weight * X + bias 
X[:10], y[:10]  `}
          outputCode={`(tensor([[0.0000],
         [0.0200],
         [0.0400],
         [0.0600],
         [0.0800],
         [0.1000],
         [0.1200],
         [0.1400],
         [0.1600],
         [0.1800]]),
 tensor([[0.3000],
         [0.3140],
         [0.3280],
         [0.3420],
         [0.3560],
         [0.3700],
         [0.3840],
         [0.3980],
         [0.4120],
         [0.4260]]))  `}
        />

        <p>
          Now that we have data, let's split it into training and test sets
          using an 80/20 split: 80% for training and 20% for testing.
        </p>

        <CodeIOBlock
          inputCode={`# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)  `}
          outputCode={`(40, 40, 10, 10)  `}
        />

        <p>Excellent, let's visualize them to make sure they look okay.</p>

        <CodeIOBlock
          inputCode={`# Note: If you've reset your runtime, this function won't work, 
# you'll have to rerun the cell above where it's instantiated.
plot_predictions(X_train, y_train, X_test, y_test)  `}
        />

        <img src={plotPred} className="centered-image" />

        <h2>6.2 Building a PyTorch linear model</h2>
        <p>
          Now, let's create a model. We'll use{" "}
          <code>nn.Linear(in_features, out_features)</code> to define it
          automatically. Since our data has one input feature (X) and one output
          feature (y), we'll set both <code>in_features</code> and{" "}
          <code>out_features</code> to 1.
        </p>
        <img src={LinearModel} className="centered-image" />

        <p>
          Creating a linear regression model using <code>nn.Parameter</code>{" "}
          versus using <code>nn.Linear</code>. There are many examples of
          pre-built computations in the <code>torch.nn</code> module, including
          popular neural network layers.
        </p>
        <CodeIOBlock
          inputCode={`# Subclass nn.Module to make our model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)
    
    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Set the manual seed when creating the model (this isn't always needed but is used for demonstrative purposes, try commenting it out and seeing what happens)
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1, model_1.state_dict()  `}
          outputCode={`(LinearRegressionModelV2(
   (linear_layer): Linear(in_features=1, out_features=1, bias=True)
 ),
 OrderedDict([('linear_layer.weight', tensor([[0.7645]])),
              ('linear_layer.bias', tensor([0.8300]))]))  `}
        />

        <p>
          Notice that <code>model_1.state_dict()</code> shows the{" "}
          <code>nn.Linear()</code> layer has created random weight and bias
          parameters. Now, let's move our model to a GPU if available. We can
          change the device using <code>.to(device)</code>. First, let's check
          the model's current device.
        </p>

        <CodeIOBlock
          inputCode={`# Check model device
next(model_1.parameters()).device  `}
          outputCode={`device(type='cpu')  `}
        />

        <p>
          It looks like the model is on the CPU by default. Let's move it to the
          GPU if available.
        </p>
        <CodeIOBlock
          inputCode={`# Set model to GPU if it's available, otherwise it'll default to CPU
model_1.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
next(model_1.parameters()).device  `}
          outputCode={`device(type='cuda', index=0)  `}
        />

        <p>
          Thanks to our device-agnostic code, the above cell will work whether a
          GPU is available or not. If you have a CUDA-enabled GPU, you should
          see an output like <code>device(type='cuda', index=0)</code>.
        </p>

        <h2>6.3 Training a PyTorch model</h2>
        <p>
          Now, let's build a training and testing loop. We'll use{" "}
          <code>nn.L1Loss()</code> as our loss function and{" "}
          <code>torch.optim.SGD()</code> as our optimizer. The optimizer needs
          the model's parameters (<code>model.parameters()</code>) to adjust
          them during training. We'll use a learning rate of 0.01, which worked
          well previously.
        </p>

        <CodeIOBlock
          inputCode={`# Create loss function
loss_fn = nn.L1Loss()

# Create optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), # optimize newly created model's parameters
                            lr=0.01)  `}
        />

        <CodeIOBlock
          inputCode={`torch.manual_seed(42)

# Set the number of epochs 
epochs = 1000 

# Put data on the available device
# Without this, error will happen (not all model/data on device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    ### Training
    model_1.train() # train mode is on by default after construction

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Testing
    model_1.eval() # put the model in evaluation mode for testing (inference)
    # 1. Forward pass
    with torch.inference_mode():
        test_pred = model_1(X_test)
    
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")  `}
          outputCode={`Epoch: 0 | Train loss: 0.5551779866218567 | Test loss: 0.5739762187004089
Epoch: 100 | Train loss: 0.006215683650225401 | Test loss: 0.014086711220443249
Epoch: 200 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
Epoch: 300 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
Epoch: 400 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
Epoch: 500 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
Epoch: 600 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
Epoch: 700 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
Epoch: 800 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
Epoch: 900 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882  `}
        />
        <div className="note">
          <strong>Note:</strong>Due to the random nature of machine learning,
          you will likely get slightly different results (different loss and
          prediction values) depending on whether your model was trained on CPU
          or GPU. This is true even if you use the same random seed on either
          device. If the difference is large, you may want to look for errors,
          however, if it is small (ideally it is), you can ignore it.
        </div>

        <p>
          The loss looks quite low. Let's check the parameters our model has
          learned and compare them to the original hard-coded parameters.
        </p>

        <CodeIOBlock
          inputCode={`# Find our model's learned parameters
from pprint import pprint # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html 
print("The model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")  `}
          outputCode={`The model learned the following values for weights and bias:
OrderedDict([('linear_layer.weight', tensor([[0.6968]], device='cuda:0')),
             ('linear_layer.bias', tensor([0.3025], device='cuda:0'))])

And the original values for weights and bias are:
weights: 0.7, bias: 0.3  `}
        />

        <p>
          Our model is very close to perfect! However, in practice, you rarely
          know the ideal parameters beforehand. If you did, machine learning
          wouldn't be as exciting. Plus, real-world problems often involve
          millions of parameters, making it much easier to let a computer figure
          them out.
        </p>
        <h2>6.4 Making predictions with a trained PyTorch model</h2>
        <p>
          Now that we have a trained model, let's switch it to evaluation mode
          and make some predictions.
        </p>
        <CodeIOBlock
          inputCode={`# Turn model into evaluation mode
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test)
y_preds  `}
          outputCode={`tensor([[0.8600],
        [0.8739],
        [0.8878],
        [0.9018],
        [0.9157],
        [0.9296],
        [0.9436],
        [0.9575],
        [0.9714],
        [0.9854]], device='cuda:0')  `}
        />
        <p>
          If you're making predictions with data on the GPU, you might see{" "}
          <code>device='cuda:0'</code>, indicating that the data is on the first
          available GPU (CUDA device 0). If you use multiple GPUs in the future,
          this number could be higher. Now, let's plot our model's predictions.
        </p>
        <div className="note">
          <strong>Note:</strong>Many data science libraries such as pandas,
          matplotlib and NumPy aren't capable of using data that is stored on
          GPU. So you might run into some issues when trying to use a function
          from one of these libraries with tensor data not stored on the CPU. To
          fix this, you can call .cpu() on your target tensor to return a copy
          of your target tensor on the CPU.
        </div>
        <CodeIOBlock
          inputCode={`# plot_predictions(predictions=y_preds) # -> won't work... data not on CPU

# Put data on the CPU and plot it
plot_predictions(predictions=y_preds.cpu())  `}
        />
        <img src={PredPlot} className="centered-image" />

        <p>
          Wow, look at those red dots! They align almost perfectly with the
          green dots. The extra epochs really helped improve the model's
          accuracy.
        </p>

        <h2>6.5 Saving and loading a PyTorch model</h2>
        <p>
          We're happy with our model's predictions, so let's save it to a file
          for later use.
        </p>

        <CodeIOBlock
          inputCode={`from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)   `}
          outputCode={`Saving model to: models/01_pytorch_workflow_model_1.pth  `}
        />

        <p>
          To ensure everything worked well, let's load the model back in. We'll
          create a new instance of <code>LinearRegressionModelV2()</code>, load
          the model state dictionary using{" "}
          <code>torch.nn.Module.load_state_dict()</code>, and move the new model
          instance to the target device for device-agnostic code.
        </p>

        <CodeIOBlock
          inputCode={`# Instantiate a fresh instance of LinearRegressionModelV2
loaded_model_1 = LinearRegressionModelV2()

# Load model state dict 
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
loaded_model_1.to(device)

print(f"Loaded model:\n{loaded_model_1}")
print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")  `}
          outputCode={`Loaded model:
LinearRegressionModelV2(
  (linear_layer): Linear(in_features=1, out_features=1, bias=True)
)
Model on device:
cuda:0  `}
        />

        <p>
          Now, let's evaluate the loaded model to see if its predictions match
          those made before saving.
        </p>

        <CodeIOBlock
          inputCode={`# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)
y_preds == loaded_model_1_preds  `}
          outputCode={`tensor([[True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True]], device='cuda:0')  `}
        />

        <p>
          Everything checks out! Well done! You've now built and trained your
          first two neural network models in PyTorch. It's time to practice your
          skills.
        </p>
      </section>
    </div>
  );
};

export default WorkflowFundamentals;
