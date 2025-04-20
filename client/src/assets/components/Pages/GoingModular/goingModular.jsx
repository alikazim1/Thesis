import React from "react";
import { FaBrain, FaCogs, FaCube, FaRocket } from "react-icons/fa";
import work from "./Img/workflow.png";
import script from "./Img/script.png";
import cell from "./Img/cell.png";
import modular from "./Img/modular.png";
import size from "./Img/size.png";
import data from "./Img/data.png";
import CodeIOBlock from "./CodeIOBlock.jsx";

const goingModular = () => {
  return (
    <div className="content">
      <h1 className="page-title">06. Going Modular</h1>

      <section>
        <p>
          This section answers the question, "how do I turn my notebook code
          into Python scripts?"
        </p>

        <p>
          To do so, we're going to turn the most useful code cells in notebook
          04. PyTorch Custom Datasets into a series of Python scripts saved to a
          directory called <code>going_modular</code>.
        </p>

        <h2>What is going modular?</h2>
        <p>
          Going modular involves turning notebook code (from a Jupyter Notebook
          or Google Colab notebook) into a series of different Python scripts
          that offer similar functionality.
        </p>

        <p>
          For example, we could turn our notebook code from a series of cells
          into the following Python files:
        </p>

        <ul>
          <li>
            <code>data_setup.py</code> - a file to prepare and download data if
            needed.
          </li>
          <li>
            <code>engine.py</code> - a file containing various training
            functions.
          </li>
          <li>
            <code>model_builder.py</code> or <code>model.py</code> - a file to
            create a PyTorch model.
          </li>
          <li>
            <code>train.py</code> - a file to leverage all other files and train
            a target PyTorch model.
          </li>
          <li>
            <code>utils.py</code> - a file dedicated to helpful utility
            functions.
          </li>
        </ul>

        <div className="note">
          <strong>Note:</strong>The naming and layout of the above files will
          depend on your use case and code requirements. Python scripts are as
          general as individual notebook cells, meaning, you could create one
          for almost any kind of functionality.
        </div>

        <h2>Why would you want to go modular?</h2>

        <p>
          Notebooks are fantastic for iteratively exploring and running
          experiments quickly.
        </p>

        <p>
          However, for larger scale projects you may find Python scripts more
          reproducible and easier to run.
        </p>

        <p>
          Though this is a debated topic, as companies like Netflix have shown
          how they use notebooks for production code.
        </p>

        <p>
          <strong>Production code</strong> is code that runs to offer a service
          to someone or something.
        </p>

        <p>
          For example, if you have an app running online that other people can
          access and use, the code running that app is considered production
          code.
        </p>

        <p>
          And libraries like fast.ai's nb-dev (short for notebook development)
          enable you to write whole Python libraries (including documentation)
          with Jupyter Notebooks.
        </p>

        <p>
          <strong>Pros and cons of notebooks vs Python scripts</strong>
        </p>
        <p>
          There's arguments for both sides. But this list sums up a few of the
          main topics.
        </p>

        <table>
          <thead>
            <tr>
              <th>Pros</th>
              <th>Cons</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>Notebooks</strong>
              </td>
              <td>
                <strong>Notebooks</strong>
              </td>
            </tr>
            <tr>
              <td>Easy to experiment/get started</td>
              <td>Versioning can be hard</td>
            </tr>
            <tr>
              <td>Easy to share (e.g. a link to a Google Colab notebook)</td>
              <td>Hard to use only specific parts</td>
            </tr>
            <tr>
              <td>Very visual</td>
              <td>Text and graphics can get in the way of code</td>
            </tr>
            <tr>
              <td>
                <strong>Python scripts</strong>
              </td>
              <td>
                <strong>Python scripts</strong>
              </td>
            </tr>
            <tr>
              <td>
                Can package code together (saves rewriting similar code across
                different notebooks)
              </td>
              <td>
                Experimenting isn't as visual (usually have to run the whole
                script rather than one cell)
              </td>
            </tr>
            <tr>
              <td>Can use git for versioning</td>
              <td></td>
            </tr>
            <tr>
              <td>Many open source projects use scripts</td>
              <td></td>
            </tr>
            <tr>
              <td>
                Larger projects can be run on cloud vendors (not as much support
                for notebooks)
              </td>
              <td></td>
            </tr>
          </tbody>
        </table>

        <h2>Workflow</h2>
        <p>
          I typically begin machine learning projects using <code>Jupyter</code>{" "}
          or <code>Google Colab</code> notebooks for rapid experimentation and
          visualization. Once I have a working solution, I transfer the most
          useful code segments into <code>Python scripts</code>.
        </p>

        <img src={work} className="centered-image" />
        <p>
          There are various workflows for writing machine learning code. Some
          people prefer to begin with scripts, while others, like me, prefer
          starting with notebooks and transitioning to scripts later on.
        </p>

        <h2>PyTorch in the wild</h2>
        <p>
          During your journey, you'll encounter many code repositories for
          PyTorch-based ML projects, which provide instructions on how to run
          the PyTorch code using Python scripts.
        </p>
        <p>
          For instance, you may be instructed to run code like the following in
          a terminal or command line to train a model:
        </p>
        <CodeIOBlock
          inputCode={`python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS`}
        />
        <img src={size} className="centered-image" />

        <p>
          Running a PyTorch <code>train.py</code> script in the command line
          with various hyperparameter settings.
        </p>
        <p>
          In this case, <code>train.py</code> is the target Python script, and
          it likely contains functions to train a PyTorch model.
        </p>
        <p>
          The <code>--model</code>, <code>--batch_size</code>, <code>--lr</code>
          , and <code>--num_epochs</code> are known as argument flags.
        </p>
        <p>
          You can set these to any values you like, and if they are compatible
          with <code>train.py</code>, they will work. If not, an error will
          occur.
        </p>
        <p>
          For example, let's say we wanted to train our TinyVGG model from
          notebook 04 for 10 epochs, with a batch size of 32 and a learning rate
          of 0.001:
        </p>

        <CodeIOBlock
          inputCode={`python train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 10`}
        />

        <p>
          You can set up any number of these argument flags in your{" "}
          <code>train.py</code> script to suit your needs.
        </p>
        <p>
          The PyTorch blog post for training state-of-the-art computer vision
          models uses this approach.
        </p>

        <img src={script} className="centered-image" />
        <p>
          Here's a PyTorch command-line training script recipe for training
          state-of-the-art computer vision models using 8 GPUs. Source:{" "}
          <a href="https://pytorch.org/blog">PyTorch blog</a>.
        </p>

        <h2>What we're going to cover</h2>
        <p>
          The main concept of this section is to convert useful notebook code
          cells into reusable Python files. This approach helps us avoid writing
          the same code repeatedly.
        </p>

        <p>There are two notebooks for this section:</p>
        <ul>
          <li>
            <strong>05. Going Modular: Part 1 (cell mode)</strong>: This
            notebook runs as a traditional Jupyter Notebook/Google Colab
            notebook and is a condensed version of notebook 04.
          </li>
          <li>
            <strong>05. Going Modular: Part 2 (script mode)</strong>: This
            notebook is the same as Part 1 but with added functionality to turn
            each of the major sections into Python scripts, such as{" "}
            <code>data_setup.py</code> and <code>train.py</code>.
          </li>
        </ul>

        <p>
          This document focuses on the code cells in{" "}
          <strong>05. Going Modular: Part 2 (script mode)</strong>, specifically
          the ones with <code>%%writefile ...</code> at the top.
        </p>

        <h2>Why two parts?</h2>
        <p>
          Sometimes, the best way to learn something is by comparing it to
          something else. By running each notebook side by side, you'll be able
          to observe how they differ, and that's where the key learnings come
          from.
        </p>

        <img src={cell} className="centered-image" />

        <p>
          By running the two notebooks for section 05 side by side, you'll
          notice that the script mode notebook includes additional code cells
          that convert the code from the cell mode notebook into Python scripts.
        </p>

        <h2>What we are working towards</h2>
        <p>By the end of this section, we aim to achieve two things:</p>
        <ol>
          <li>
            The ability to train the model we built in notebook 04 (Food Vision
            Mini) with a single line of code on the command line:{" "}
            <code>python train.py</code>.
          </li>
          <li>A directory structure of reusable Python scripts, such as:</li>
        </ol>
        <img src={modular} className="centered-image" />

        <p>
          <strong>Things to note:</strong>
        </p>
        <ul>
          <li>
            <strong>Docstrings</strong>: Writing reproducible and understandable
            code is essential. With this in mind, each of the functions/classes
            we'll be placing into scripts has been designed using Google's
            Python docstring style.
          </li>
          <li>
            <strong>Imports at the top of scripts</strong>: Since each Python
            script we create can be considered a small program on its own, all
            scripts will import their necessary modules at the beginning. For
            example:
          </li>
        </ul>
        <CodeIOBlock
          inputCode={`# Import modules required for train.py
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms`}
        />

        <h2>0. Cell mode vs. script mode</h2>

        <p>
          A <strong>cell mode notebook</strong>, like{" "}
          <code>05. Going Modular Part 1 (cell mode)</code>, is a notebook run
          normally where each cell is either code or markdown.
        </p>

        <p>
          A <strong>script mode notebook</strong>, such as{" "}
          <code>05. Going Modular Part 2 (script mode)</code>, is very similar
          to a cell mode notebook. However, many of the code cells are converted
          into Python scripts.
        </p>

        <div className="note">
          <strong>Note:</strong> You don't need to create Python scripts via a
          notebook, you can create them directly through an IDE (integrated
          developer environment) such as VS Code. Having the script mode
          notebook as part of this section is just to demonstrate one way of
          going from notebooks to Python scripts.
        </div>
        <h2>1. Get data</h2>
        <p>
          The process of obtaining the data in each of the 05 notebooks is the
          same as in notebook 04.
        </p>

        <p>
          A request is made to GitHub using Python's <code>requests</code>{" "}
          module to download a .zip file, which is then unzipped.
        </p>

        <CodeIOBlock
          inputCode={`import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...") 
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")`}
        />

        <img src={data} className="centered-image" />

        <h2>2. Create Datasets and DataLoaders (data_setup.py)</h2>
        <p>
          Once we have the data, we can convert it into PyTorch Datasets and
          DataLoaders (one for training data and another for testing data).
        </p>
        <p>
          We encapsulate the relevant Dataset and DataLoader creation code into
          a function called <code>create_dataloaders()</code>.
        </p>
        <p>
          Then, we write this function to a file using the command{" "}
          <code>%%writefile going_modular/data_setup.py</code>.
        </p>
        <CodeIOBlock
          inputCode={`%%writefile going_modular/data_setup.py
"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names`}
        />
        <p>
          If we want to create DataLoaders, we can now use the function defined
          in <code>data_setup.py</code> like this:
        </p>

        <CodeIOBlock
          inputCode={`# Import data_setup.py
from going_modular import data_setup

# Create train/test dataloader and get class names as a list
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(...)`}
        />

        <h2>3. Making a model (model_builder.py)</h2>
        <p>
          Over the past few notebooks (notebook 03 and notebook 04), we've built
          the TinyVGG model multiple times. Therefore, it makes sense to place
          the model in a file so we can reuse it easily. Let's write the{" "}
          <code>TinyVGG()</code> model class into a script using the line{" "}
          <code>%%writefile going_modular/model_builder.py</code>:
        </p>

        <CodeIOBlock
          inputCode={`%%writefile going_modular/model_builder.py
"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )

  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion`}
        />

        <p>
          Now instead of coding the TinyVGG model from scratch every time, we
          can import it using:
        </p>

        <CodeIOBlock
          inputCode={`import torch
# Import model_builder.py
from going_modular import model_builder
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate an instance of the model from the "model_builder.py" script
torch.manual_seed(42)
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=10, 
                              output_shape=len(class_names)).to(device)`}
        />

        <h2>
          4. Creating train_step() and test_step() functions and train() to
          combine them
        </h2>

        <p>We wrote several training functions in notebook 04:</p>
        <ul>
          <li>
            <code>train_step()</code> - Takes in a model, a DataLoader, a loss
            function, and an optimizer, and trains the model on the DataLoader.
          </li>
          <li>
            <code>test_step()</code> - Takes in a model, a DataLoader, and a
            loss function, and evaluates the model on the DataLoader.
          </li>
          <li>
            <code>train()</code> - Performs both training and evaluation for a
            given number of epochs and returns a results dictionary.
          </li>
        </ul>
        <p>
          Since these functions are the core of our model training, we can place
          them into a Python script called <code>engine.py</code> with the line{" "}
          <code>%%writefile going_modular/engine.py</code>:
        </p>

        <CodeIOBlock
          inputCode={`%%writefile going_modular/engine.py
"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          # Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results`}
        />

        <p>
          Now we've got the engine.py script, we can import functions from it
          via:
        </p>

        <CodeIOBlock
          inputCode={`# Import engine.py
from going_modular import engine

# Use train() by calling it from engine.py
engine.train(...)`}
        />

        <h2>5. Creating a function to save the model (utils.py)</h2>

        <p>Often, you'll want to save a model during or after training.</p>
        <p>
          Since we've written the code to save a model several times in previous
          notebooks, it makes sense to turn it into a reusable function and save
          it to a file.
        </p>
        <p>
          It's common practice to store helper functions in a file called{" "}
          <code>utils.py</code> (short for utilities).
        </p>
        <p>
          Let's save our <code>save_model()</code> function to a file called{" "}
          <code>utils.py</code> with the line{" "}
          <code>%%writefile going_modular/utils.py</code>:
        </p>

        <CodeIOBlock
          inputCode={`%%writefile going_modular/utils.py
"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)`}
        />

        <p>
          Now if we wanted to use our save_model() function, instead of writing
          it all over again, we can import it and use it via:
        </p>

        <CodeIOBlock
          inputCode={`# Import utils.py
from going_modular import utils

# Save a model to file
save_model(model=...
           target_dir=...,
           model_name=...)`}
        />

        <h2>6. Train, evaluate and save the model (train.py)</h2>
        <p>
          As previously discussed, you'll often come across PyTorch repositories
          that consolidate all their functionality into a single{" "}
          <code>train.py</code> file.
        </p>
        <p>
          This file essentially serves the purpose of saying "train the model
          using whatever data is available."
        </p>
        <p>
          In our <code>train.py</code> file, we'll integrate all the
          functionality from the other Python scripts we've created and use it
          to train a model.
        </p>
        <p>
          This will allow us to train a PyTorch model using a single line of
          code in the command line:
        </p>

        <CodeIOBlock inputCode={`python train.py`} />
        <p>
          To create <code>train.py</code>, we'll go through the following steps:
        </p>
        <ol>
          <li>
            Import the necessary dependencies, including <code>torch</code>,{" "}
            <code>os</code>, <code>torchvision.transforms</code>, and all the
            scripts from the <code>going_modular</code> directory, such as{" "}
            <code>data_setup</code>, <code>engine</code>,{" "}
            <code>model_builder</code>, and <code>utils</code>.
          </li>
          <p>
            <em>
              Note: Since <code>train.py</code> will be inside the{" "}
              <code>going_modular</code> directory, we can import the other
              modules using <code>import ...</code> rather than{" "}
              <code>from going_modular import ...</code>.
            </em>
          </p>

          <li>
            Set up various hyperparameters, such as batch size, number of
            epochs, learning rate, and number of hidden units (these could
            potentially be set in the future using Python's{" "}
            <code>argparse</code>).
          </li>

          <li>Set up the training and test directories.</li>

          <li>Implement device-agnostic code.</li>

          <li>Create the necessary data transformations.</li>

          <li>
            Create the DataLoaders using <code>data_setup.py</code>.
          </li>

          <li>
            Build the model using <code>model_builder.py</code>.
          </li>

          <li>Set up the loss function and optimizer.</li>

          <li>
            Train the model using <code>engine.py</code>.
          </li>

          <li>
            Save the model using <code>utils.py</code>.
          </li>
        </ol>
        <p>
          Finally, we can create the file from a notebook cell using the
          following line:
        </p>
        <pre>
          <code>%%writefile going_modular/train.py</code>
        </pre>

        <CodeIOBlock
          inputCode={`%%writefile going_modular/train.py
"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")`}
        />

        <p>
          <strong>Woohoo!</strong> Now we can train a PyTorch model by running
          the following command on the command line:
        </p>
        <pre>
          <code>python train.py</code>
        </pre>

        <p>
          Doing this will leverage all of the other code scripts we've created.
        </p>
        <p>
          Additionally, we could adjust our train.py file to use argument flag
          inputs with Python's <code>argparse</code> module. This would allow us
          to provide different hyperparameter settings, as we discussed
          previously:
        </p>
        <pre>
          <code>
            python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr
            LEARNING_RATE --num_epochs NUM_EPOCHS
          </code>
        </pre>

        <h2>The End</h2>
      </section>
    </div>
  );
};

export default goingModular;
