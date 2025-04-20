import React from "react";
import { FaBrain, FaCogs, FaCube, FaRocket } from "react-icons/fa";
import food from "./Img/food.png";
import Custom from "./Img/custom.png";
import items from "./Img/items.png";
import error from "./Img/error.png";
import bigpizza from "./Img/bigpizza.png";
import foods from "./Img/foods.png";
import fitting from "./Img/fitting.png";
import lossplot from "./Img/lossplot.png";
import man from "./Img/man.png";
import graphs from "./Img/graphs.png";
import dataload from "./Img/data load.png";
import graph from "./Img/graph.png";
import pizza from "./Img/pizza.png";
import plot from "./Img/plot.png";
import pizzachannels from "./Img/pizzachannels.png";
import CodeIOBlock from "./CodeIOBlock.jsx";

const customDatasets = () => {
  return (
    <div className="content">
      <h1 className="page-title">05. Custom Datasets</h1>

      <section>
        <p>
          In the last notebook (Notebook 03), we learned how to build computer
          vision models using PyTorch's built-in <b>FashionMNIST</b> dataset.
        </p>

        <p>
          The steps we followed are common in many machine learning problems:
        </p>
        <ol>
          <li>Find a dataset</li>
          <li>Convert the data into numbers</li>
          <li>Build or use a model to find patterns and make predictions</li>
        </ol>

        <p>
          While PyTorch offers many built-in datasets for machine learning
          tasks, you’ll often need to use your own custom dataset.
        </p>

        <h2>What is a custom dataset?</h2>
        <section>
          <p>
            A <b>custom dataset</b> is a set of data related to the specific
            problem you're working on.
          </p>

          <p>It can be made up of almost anything. For example:</p>
          <ul>
            <li>
              If you're building a food image classification app like{" "}
              <b>Nutrify</b>, the dataset could be food images.
            </li>
            <li>
              If you're working on a model to detect positive or negative text
              reviews, the dataset could be customer reviews and their ratings.
            </li>
            <li>
              If you're building a sound classification app, the dataset could
              be sound samples with their labels.
            </li>
            <li>
              If you're creating a recommendation system, the dataset could
              include examples of products that customers have purchased.
            </li>
          </ul>
        </section>

        <img src={Custom} className="centered-image" />

        <p>
          PyTorch has built-in functions for loading custom datasets using
          libraries like <b>TorchVision</b>, <b>TorchText</b>, <b>TorchAudio</b>
          , and <b>TorchRec</b>.
        </p>

        <p>
          But sometimes, these functions might not be enough for your specific
          needs.
        </p>

        <p>
          In that case, you can create your own dataset by subclassing{" "}
          <code>torch.utils.data.Dataset</code> and customizing it however you
          like.
        </p>

        <h2>What we are going to cover</h2>

        <p>
          We're going to apply the PyTorch workflow from notebook 01 and
          notebook 02 to a computer vision problem.
        </p>

        <p>
          Instead of using an in-built PyTorch dataset, we'll be using our own
          dataset of pizza, steak, and sushi images.
        </p>

        <p>
          The goal is to load these images and then build a model to train and
          predict on them.
        </p>

        <img src={dataload} className="centered-image" />

        <p>
          What we're going to build: We'll use <b>torchvision.datasets</b> as
          well as our own custom <code>Dataset</code> class to load images of
          food, and then we'll build a PyTorch computer vision model to classify
          them.
        </p>
        <table>
          <tr>
            <th>Topic</th>
            <th>Contents</th>
          </tr>
          <tr>
            <td>0. Importing PyTorch and setting up device-agnostic code</td>
            <td>
              Let's get PyTorch loaded and then follow best practice to setup
              our code to be device-agnostic.
            </td>
          </tr>
          <tr>
            <td>1. Get data</td>
            <td>
              We're going to be using our own custom dataset of pizza, steak and
              sushi images.
            </td>
          </tr>
          <tr>
            <td>2. Become one with the data (data preparation)</td>
            <td>
              At the beginning of any new machine learning problem, it's
              paramount to understand the data you're working with. Here we'll
              take some steps to figure out what data we have.
            </td>
          </tr>
          <tr>
            <td>3. Transforming data</td>
            <td>
              Often, the data you get won't be 100% ready to use with a machine
              learning model, here we'll look at some steps we can take to
              transform our images so they're ready to be used with a model.
            </td>
          </tr>
          <tr>
            <td>4. Loading data with ImageFolder (option 1)</td>
            <td>
              PyTorch has many in-built data loading functions for common types
              of data. ImageFolder is helpful if our images are in standard
              image classification format.
            </td>
          </tr>
          <tr>
            <td>5. Loading image data with a custom Dataset</td>
            <td>
              What if PyTorch didn't have an in-built function to load data
              with? This is where we can build our own custom subclass of{" "}
              <code>torch.utils.data.Dataset</code>.
            </td>
          </tr>
          <tr>
            <td>6. Other forms of transforms (data augmentation)</td>
            <td>
              Data augmentation is a common technique for expanding the
              diversity of your training data. Here we'll explore some of
              torchvision's in-built data augmentation functions.
            </td>
          </tr>
          <tr>
            <td>7. Model 0: TinyVGG without data augmentation</td>
            <td>
              By this stage, we'll have our data ready, let's build a model
              capable of fitting it. We'll also create some training and testing
              functions for training and evaluating our model.
            </td>
          </tr>
          <tr>
            <td>8. Exploring loss curves</td>
            <td>
              Loss curves are a great way to see how your model is
              training/improving over time. They're also a good way to see if
              your model is underfitting or overfitting.
            </td>
          </tr>
          <tr>
            <td>9. Model 1: TinyVGG with data augmentation</td>
            <td>
              By now, we've tried a model without, how about we try one with
              data augmentation?
            </td>
          </tr>
          <tr>
            <td>10. Compare model results</td>
            <td>
              Let's compare our different models' loss curves and see which
              performed better and discuss some options for improving
              performance.
            </td>
          </tr>
          <tr>
            <td>11. Making a prediction on a custom image</td>
            <td>
              Our model is trained on a dataset of pizza, steak and sushi
              images. In this section we'll cover how to use our trained model
              to predict on an image outside of our existing dataset.
            </td>
          </tr>
        </table>

        <h2>0. Importing PyTorch and setting up device-agnostic code</h2>

        <CodeIOBlock
          inputCode={`import torch
from torch import nn

# Note: this notebook requires torch >= 1.10.0
torch.__version__  `}
          outputCode={`'1.12.1+cu113'  `}
        />

        <p>
          And now let's follow best practice and setup device-agnostic code.
        </p>

        <div className="note">
          <strong>Note:</strong>If you're using Google Colab, and you don't have
          a GPU turned on yet, it's now time to turn one on via Runtime -)
          Change runtime type -) Hardware accelerator -) GPU. If you do this,
          your runtime will likely reset and you'll have to run all of the cells
          above by going Runtime -) Run before.
        </div>

        <CodeIOBlock
          inputCode={`# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device`}
          outputCode={`'cuda'`}
        />

        <h2>1. Get data</h2>
        <p>First things first, we need some data.</p>

        <p>
          And like any good cooking show, some data has already been prepared
          for us.
        </p>

        <p>
          We're going to start small, because we're not looking to train the
          biggest model or use the biggest dataset yet.
        </p>

        <p>
          Machine learning is an iterative process: start small, get something
          working, and increase when necessary.
        </p>

        <p>
          The data we're going to be using is a subset of the <b>Food101</b>{" "}
          dataset.
        </p>

        <p>
          <b>Food101</b> is a popular computer vision benchmark, containing
          1,000 images of 101 different kinds of foods, totaling 101,000 images
          (75,750 train and 25,250 test).
        </p>

        <p>Can you think of 101 different foods?</p>

        <p>Can you think of a computer program to classify 101 foods?</p>

        <p>I can. A machine learning model!</p>

        <p>
          Specifically, a PyTorch computer vision model, as we covered in
          notebook 03.
        </p>

        <p>
          But instead of 101 food classes, we're going to start with 3: pizza,
          steak, and sushi.
        </p>

        <p>
          And instead of 1,000 images per class, we're going to start with a
          random 10% (start small, increase when necessary).
        </p>

        <p>
          If you'd like to see where the data came from, check out the following
          resources:
        </p>
        <ul>
          <li>
            <a href="https://example.com">
              Original Food101 dataset and paper website
            </a>
          </li>
          <li>
            <a href="https://example.com">
              torchvision.datasets.Food101 - the version of the data I
              downloaded for this notebook
            </a>
          </li>
          <li>
            <a href="https://example.com">
              extras/04_custom_data_creation.ipynb - a notebook I used to format
              the Food101 dataset for this notebook
            </a>
          </li>
          <li>
            <a href="https://example.com">
              data/pizza_steak_sushi.zip - the zip archive of pizza, steak, and
              sushi images from Food101, created with the notebook linked above
            </a>
          </li>
        </ul>

        <p>Let's write some code to download the formatted data from GitHub.</p>

        <div className="note">
          {" "}
          The dataset we're about to use has been pre-formatted for what we'd
          like to use it for. However, you'll often have to format your own
          datasets for whatever problem you're working on. This is a regular
          practice in the machine learning world.
          <strong>Note:</strong>
        </div>

        <CodeIOBlock
          inputCode={`import requests
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
        zip_ref.extractall(image_path)`}
          outputCode={`data/pizza_steak_sushi directory exists.`}
        />

        <h2>2. Become one with the data (data preparation)</h2>
        <p>Dataset downloaded!</p>

        <p>Time to become one with it.</p>

        <p>This is another important step before building a model.</p>
        <p>
          Data preparation is crucial. Before building a model, become one with
          the data. Ask: What am I trying to do here?{" "}
          <i>Source: @mrdbourke Twitter</i>.
        </p>

        <p>What does inspecting the data and becoming one with it mean?</p>

        <p>
          Before starting a project or building a model, it's important to
          understand the data you're working with.
        </p>

        <p>
          In our case, we have images of pizza, steak, and sushi in standard
          image classification format.
        </p>

        <p>
          The image classification format organizes images into separate
          directories named after each class.
        </p>

        <p>
          For example, all images of pizza are stored in the <code>pizza/</code>{" "}
          directory.
        </p>

        <p>
          This format is common across many image classification benchmarks,
          including ImageNet (one of the most popular computer vision benchmark
          datasets).
        </p>

        <p>
          You can see an example of the storage format below. The image numbers
          are arbitrary.
        </p>
        <CodeIOBlock
          inputCode={`pizza_steak_sushi/ <- overall dataset folder
    train/ <- training images
        pizza/ <- class name as folder name
            image01.jpeg
            image02.jpeg
            ...
        steak/
            image24.jpeg
            image25.jpeg
            ...
        sushi/
            image37.jpeg
            ...
    test/ <- testing images
        pizza/
            image101.jpeg
            image102.jpeg
            ...
        steak/
            image154.jpeg
            image155.jpeg
            ...
        sushi/
            image167.jpeg
            ...`}
        />

        <p>
          The goal will be to take this data storage structure and turn it into
          a dataset usable with PyTorch.
        </p>

        <div className="note">
          <strong>Note:</strong>The structure of the data you work with will
          vary depending on the problem you're working on. But the premise still
          remains: become one with the data, then find a way to best turn it
          into a dataset compatible with PyTorch.
        </div>

        <p>
          We can inspect what's in our data directory by writing a small helper
          function to walk through each of the subdirectories and count the
          files present.
        </p>

        <p>
          To do so, we'll use Python's in-built <code>os.walk()</code>.
        </p>

        <CodeIOBlock
          inputCode={`import os
def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
walk_through_dir(image_path)`}
          outputCode={`There are 2 directories and 1 images in 'data/pizza_steak_sushi'.
There are 3 directories and 0 images in 'data/pizza_steak_sushi/test'.
There are 0 directories and 19 images in 'data/pizza_steak_sushi/test/steak'.
There are 0 directories and 31 images in 'data/pizza_steak_sushi/test/sushi'.
There are 0 directories and 25 images in 'data/pizza_steak_sushi/test/pizza'.
There are 3 directories and 0 images in 'data/pizza_steak_sushi/train'.
There are 0 directories and 75 images in 'data/pizza_steak_sushi/train/steak'.
There are 0 directories and 72 images in 'data/pizza_steak_sushi/train/sushi'.
There are 0 directories and 78 images in 'data/pizza_steak_sushi/train/pizza'.`}
        />
        <p>Excellent!</p>

        <p>
          It looks like we've got about 75 images per training class and 25
          images per testing class.
        </p>

        <p>That should be enough to get started.</p>

        <p>
          Remember, these images are subsets of the original Food101 dataset.
        </p>

        <p>You can see how they were created in the data creation notebook.</p>

        <p>While we're at it, let's set up our training and testing paths.</p>

        <CodeIOBlock
          inputCode={`# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir`}
          outputCode={`(PosixPath('data/pizza_steak_sushi/train'),
 PosixPath('data/pizza_steak_sushi/test'))`}
        />

        <h2>2.1 Visualize an image</h2>

        <p>Okay, we've seen how our directory structure is formatted.</p>

        <p>
          Now, in the spirit of the data explorer, it's time to visualize,
          visualize, visualize!
        </p>

        <p>Let's write some code to:</p>
        <ul>
          <li>
            Get all of the image paths using <code>pathlib.Path.glob()</code> to
            find all of the files ending in <code>.jpg</code>.
          </li>
          <li>
            Pick a random image path using Python's <code>random.choice()</code>
            .
          </li>
          <li>
            Get the image class name using <code>pathlib.Path.parent.stem</code>
            .
          </li>
          <li>
            Since we're working with images, open the random image path using{" "}
            <code>PIL.Image.open()</code> (PIL stands for Python Image Library).
          </li>
          <li>Show the image and print some metadata.</li>
        </ul>

        <CodeIOBlock
          inputCode={`import random
from PIL import Image

# Set seed
random.seed(42) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")
img`}
          outputCode={`Random image path: data/pizza_steak_sushi/test/pizza/2124579.jpg
Image class: pizza
Image height: 384
Image width: 512`}
        />
        <img src={pizza} className="centered-image" />
        <p>
          We can do the same with <code>matplotlib.pyplot.imshow()</code>,
          except we have to convert the image to a NumPy array first.
        </p>

        <CodeIOBlock
          inputCode={`import numpy as np
import matplotlib.pyplot as plt

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);`}
        />

        <img src={pizzachannels} className="centered-image" />
        <h2>3. Transforming data</h2>
        <p>Now, what if we wanted to load our image data into PyTorch?</p>

        <p>Before we can use our image data with PyTorch, we need to:</p>
        <ul>
          <li>
            Turn it into tensors (numerical representations of our images).
          </li>
          <li>
            Turn it into a <code>torch.utils.data.Dataset</code> and
            subsequently a <code>torch.utils.data.DataLoader</code>, we'll call
            these <b>Dataset</b> and <b>DataLoader</b> for short.
          </li>
        </ul>

        <p>
          There are several different kinds of pre-built datasets and dataset
          loaders for PyTorch, depending on the problem you're working on.
        </p>
        <table>
          <thead>
            <tr>
              <th>Problem Space</th>
              <th>Pre-built Datasets and Functions</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Vision</td>
              <td>torchvision.datasets</td>
            </tr>
            <tr>
              <td>Audio</td>
              <td>torchaudio.datasets</td>
            </tr>
            <tr>
              <td>Text</td>
              <td>torchtext.datasets</td>
            </tr>
            <tr>
              <td>Recommendation system</td>
              <td>torchrec.datasets</td>
            </tr>
          </tbody>
        </table>
        <p>
          Since we're working with a vision problem, we'll be looking at{" "}
          <code>torchvision.datasets</code> for our data loading functions as
          well as <code>torchvision.transforms</code> for preparing our data.
        </p>

        <p>Let's import some base libraries.</p>
        <CodeIOBlock
          inputCode={`import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms`}
        />

        <h2>Transforming data with torchvision.transforms</h2>
        <p>
          We've got folders of images, but before we can use them with PyTorch,
          we need to convert them into tensors.
        </p>

        <p>
          One of the ways we can do this is by using the{" "}
          <code>torchvision.transforms</code> module.
        </p>

        <p>
          <code>torchvision.transforms</code> contains many pre-built methods
          for formatting images, turning them into tensors, and even
          manipulating them for data augmentation (the practice of altering data
          to make it harder for a model to learn, which we'll explore later on).
        </p>

        <p>
          To get experience with <code>torchvision.transforms</code>, let's
          write a series of transform steps that:
        </p>
        <ul>
          <li>
            Resize the images using <code>transforms.Resize()</code> (from about
            512x512 to 64x64, the same shape as the images on the CNN Explainer
            website).
          </li>
          <li>
            Flip our images randomly on the horizontal using{" "}
            <code>transforms.RandomHorizontalFlip()</code> (this could be
            considered a form of data augmentation because it will artificially
            change our image data).
          </li>
          <li>
            Turn our images from a PIL image to a PyTorch tensor using{" "}
            <code>transforms.ToTensor()</code>.
          </li>
        </ul>

        <p>
          We can compile all of these steps using{" "}
          <code>torchvision.transforms.Compose()</code>.
        </p>
        <CodeIOBlock
          inputCode={`# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])`}
        />

        <p>
          Now we've got a composition of transforms, let's write a function to
          try them out on various images.
        </p>
        <CodeIOBlock
          inputCode={`def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list, 
                        transform=data_transform, 
                        n=3)`}
        />

        <img src={food} className="centered-image" />
        <p>Nice!</p>

        <p>
          We've now got a way to convert our images to tensors using{" "}
          <code>torchvision.transforms</code>.
        </p>

        <p>
          We can also manipulate their size and orientation if needed (some
          models prefer images of different sizes and shapes).
        </p>

        <p>
          Generally, the larger the shape of the image, the more information a
          model can recover.
        </p>

        <p>
          For example, an image of size <code>[256, 256, 3]</code> will have 16x
          more pixels than an image of size <code>[64, 64, 3]</code> (
          <code>(256*256*3)/(64*64*3)=16</code>).
        </p>

        <p>
          However, the tradeoff is that more pixels require more computations.
        </p>

        <div className="resource">
          Question:
          <p>
            Try commenting out one of the transforms in data_transform and
            running the plotting function plot_transformed_images() again, what
            happens?{" "}
          </p>
        </div>

        <h2>4. Option 1: Loading Image Data Using ImageFolder</h2>
        <p>
          Alright, time to turn our image data into a Dataset capable of being
          used with PyTorch.
        </p>

        <p>
          Since our data is in standard image classification format, we can use
          the class <code>torchvision.datasets.ImageFolder</code>.
        </p>

        <p>
          We can pass it the file path of a target image directory as well as a
          series of transforms we'd like to perform on our images.
        </p>

        <p>
          Let's test it out on our data folders <code>train_dir</code> and{" "}
          <code>test_dir</code>, passing in{" "}
          <code>transform=data_transform</code> to turn our images into tensors.
        </p>
        <CodeIOBlock
          inputCode={`# Use ImageFolder to create dataset(s)
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")`}
          outputCode={`Train data:
Dataset ImageFolder
    Number of datapoints: 225
    Root location: data/pizza_steak_sushi/train
    StandardTransform
Transform: Compose(
               Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
               RandomHorizontalFlip(p=0.5)
               ToTensor()
           )
Test data:
Dataset ImageFolder
    Number of datapoints: 75
    Root location: data/pizza_steak_sushi/test
    StandardTransform
Transform: Compose(
               Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
               RandomHorizontalFlip(p=0.5)
               ToTensor()
           )`}
        />
        <p>Beautiful!</p>

        <p>It looks like PyTorch has registered our Datasets.</p>

        <p>
          Let's inspect them by checking out the <code>classes</code> and{" "}
          <code>class_to_idx</code> attributes as well as the lengths of our
          training and test sets.
        </p>

        <CodeIOBlock
          inputCode={`# Get class names as a list
class_names = train_data.classes
class_names`}
          outputCode={`['pizza', 'steak', 'sushi']`}
        />

        <CodeIOBlock
          inputCode={`# Can also get class names as a dict
class_dict = train_data.class_to_idx
class_dict`}
          outputCode={`{'pizza': 0, 'steak': 1, 'sushi': 2}`}
        />

        <CodeIOBlock
          inputCode={`# Check the lengths
len(train_data), len(test_data)`}
          outputCode={`(225, 75)`}
        />

        <p>
          Nice! Looks like we'll be able to use these to reference for later.
        </p>

        <p>How about our images and labels?</p>

        <p>How do they look?</p>

        <p>
          We can index on our <code>train_data</code> and <code>test_data</code>{" "}
          Datasets to find samples and their target labels.
        </p>
        <CodeIOBlock
          inputCode={`img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")`}
          outputCode={`Image tensor:
tensor([[[0.1137, 0.1020, 0.0980,  ..., 0.1255, 0.1216, 0.1176],
         [0.1059, 0.0980, 0.0980,  ..., 0.1294, 0.1294, 0.1294],
         [0.1020, 0.0980, 0.0941,  ..., 0.1333, 0.1333, 0.1333],
         ...,
         [0.1098, 0.1098, 0.1255,  ..., 0.1686, 0.1647, 0.1686],
         [0.0863, 0.0941, 0.1098,  ..., 0.1686, 0.1647, 0.1686],
         [0.0863, 0.0863, 0.0980,  ..., 0.1686, 0.1647, 0.1647]],

        [[0.0745, 0.0706, 0.0745,  ..., 0.0588, 0.0588, 0.0588],
         [0.0706, 0.0706, 0.0745,  ..., 0.0627, 0.0627, 0.0627],
         [0.0706, 0.0745, 0.0745,  ..., 0.0706, 0.0706, 0.0706],
         ...,
         [0.1255, 0.1333, 0.1373,  ..., 0.2510, 0.2392, 0.2392],
         [0.1098, 0.1176, 0.1255,  ..., 0.2510, 0.2392, 0.2314],
         [0.1020, 0.1059, 0.1137,  ..., 0.2431, 0.2353, 0.2275]],

        [[0.0941, 0.0902, 0.0902,  ..., 0.0196, 0.0196, 0.0196],
         [0.0902, 0.0863, 0.0902,  ..., 0.0196, 0.0157, 0.0196],
         [0.0902, 0.0902, 0.0902,  ..., 0.0157, 0.0157, 0.0196],
         ...,
         [0.1294, 0.1333, 0.1490,  ..., 0.1961, 0.1882, 0.1804],
         [0.1098, 0.1137, 0.1255,  ..., 0.1922, 0.1843, 0.1804],
         [0.1059, 0.1020, 0.1059,  ..., 0.1843, 0.1804, 0.1765]]])
Image shape: torch.Size([3, 64, 64])
Image datatype: torch.float32
Image label: 0
Label datatype: <class 'int'>`}
        />
        <p>
          Our images are now in the form of a tensor (with shape{" "}
          <code>[3, 64, 64]</code>) and the labels are in the form of an integer
          relating to a specific class (as referenced by the{" "}
          <code>class_to_idx</code> attribute).
        </p>

        <p>How about we plot a single image tensor using matplotlib?</p>

        <p>
          We'll first have to permute (rearrange the order of its dimensions) so
          it's compatible.
        </p>

        <p>
          Right now our image dimensions are in the format CHW (color channels,
          height, width) but matplotlib prefers HWC (height, width, color
          channels).
        </p>
        <CodeIOBlock
          inputCode={`# Rearrange the order of dimensions
img_permute = img.permute(1, 2, 0)

# Print out different shapes (before and after permute)
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

# Plot the image
plt.figure(figsize=(10, 7))
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title(class_names[label], fontsize=14);`}
          outputCode={`Original shape: torch.Size([3, 64, 64]) -> [color_channels, height, width]
Image permute shape: torch.Size([64, 64, 3]) -> [height, width, color_channels]`}
        />

        <img src={bigpizza} className="centered-image" />
        <p>Notice the image is now more pixelated (less quality).</p>

        <p>This is due to it being resized from 512x512 to 64x64 pixels.</p>

        <p>
          The intuition here is that if you think the image is harder to
          recognize what's going on, chances are a model will find it harder to
          understand too.
        </p>

        <h2>4.1 Turn loaded images into DataLoader's</h2>
        <p>
          We've got our images as PyTorch Datasets, but now let's turn them into
          DataLoaders.
        </p>

        <p>
          We'll do so using <code>torch.utils.data.DataLoader</code>.
        </p>

        <p>
          Turning our Datasets into DataLoaders makes them iterable, so a model
          can learn the relationships between samples and targets (features and
          labels).
        </p>

        <p>
          To keep things simple, we'll use <code>batch_size=1</code> and{" "}
          <code>num_workers=1</code>.
        </p>

        <p>
          What's <code>num_workers</code>?
        </p>

        <p>Good question.</p>

        <p>
          It defines how many subprocesses will be created to load your data.
        </p>

        <p>
          Think of it like this: the higher <code>num_workers</code> is set to,
          the more compute power PyTorch will use to load your data.
        </p>

        <p>
          Personally, I usually set it to the total number of CPUs on my machine
          via Python's <code>os.cpu_count()</code>.
        </p>

        <p>
          This ensures the DataLoader recruits as many cores as possible to load
          data.
        </p>
        <div className="note">
          <strong>Note:</strong>There are more parameters you can get familiar
          with using torch.utils.data.DataLoader in the PyTorch documentation.
        </div>
        <CodeIOBlock
          inputCode={`# Turn train and test Datasets into DataLoaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data

train_dataloader, test_dataloader`}
          outputCode={`(<torch.utils.data.dataloader.DataLoader at 0x7f53c0b9dca0>,
 <torch.utils.data.dataloader.DataLoader at 0x7f53c0b9de50>)`}
        />

        <p>Awesome!</p>
        <p>
          We've now got our training and testing DataLoaders ready to go for
          training a model.
        </p>

        <CodeIOBlock
          inputCode={`img, label = next(iter(train_dataloader))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")`}
          outputCode={`Image shape: torch.Size([1, 3, 64, 64]) -> [batch_size, color_channels, height, width]
Label shape: torch.Size([1])`}
        />
        <p>
          We could now use these DataLoaders with a training and testing loop to
          train a model.
        </p>

        <p>
          But before we do, let's look at another option to load images (or
          almost any other kind of data).
        </p>

        <h2>5. Option 2: Loading Image Data with a Custom Dataset</h2>

        <p>
          What if a pre-built Dataset creator like{" "}
          <code>torchvision.datasets.ImageFolder()</code> didn't exist?
        </p>

        <p>Or one for your specific problem didn't exist?</p>

        <p>Well, you could build your own.</p>

        <p>
          But wait, what are the pros and cons of creating your own custom way
          to load Datasets?
        </p>
        <table>
          <thead>
            <tr>
              <th>Pros of creating a custom Dataset</th>
              <th>Cons of creating a custom Dataset</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Can create a Dataset out of almost anything.</td>
              <td>
                Even though you could create a Dataset out of almost anything,
                it doesn't mean it will work.
              </td>
            </tr>
            <tr>
              <td>Not limited to PyTorch pre-built Dataset functions.</td>
              <td>
                Using a custom Dataset often results in writing more code, which
                could be prone to errors or performance issues.
              </td>
            </tr>
          </tbody>
        </table>
        <p>
          To see this in action, let's work towards replicating{" "}
          <code>torchvision.datasets.ImageFolder()</code> by subclassing{" "}
          <code>torch.utils.data.Dataset</code> (the base class for all
          Dataset's in PyTorch).
        </p>

        <p>We'll start by importing the modules we need:</p>
        <ul>
          <li>
            Python's <code>os</code> for dealing with directories (our data is
            stored in directories).
          </li>
          <li>
            Python's <code>pathlib</code> for dealing with filepaths (each of
            our images has a unique filepath).
          </li>
          <li>
            <code>torch</code> for all things PyTorch.
          </li>
          <li>
            PIL's <code>Image</code> class for loading images.
          </li>
          <li>
            <code>torch.utils.data.Dataset</code> to subclass and create our own
            custom Dataset.
          </li>
          <li>
            <code>torchvision.transforms</code> to turn our images into tensors.
          </li>
          <li>
            Various types from Python's <code>typing</code> module to add type
            hints to our code.
          </li>
        </ul>

        <div className="note">
          <strong>Note:</strong>
          You can customize the following steps for your own dataset. The
          premise remains: write code to load your data in the format you'd like
          it.
        </div>

        <CodeIOBlock
          inputCode={`import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List`}
        />

        <p>
          Remember how our instances of torchvision.datasets.ImageFolder()
          allowed us to use the classes and class_to_idx attributes?
        </p>

        <CodeIOBlock
          inputCode={`# Instance of torchvision.datasets.ImageFolder()
train_data.classes, train_data.class_to_idx`}
          outputCode={`(['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2})`}
        />

        <h2>5.1 Creating a helper function to get class names</h2>

        <p>
          Let's write a helper function capable of creating a list of class
          names and a dictionary of class names and their indexes given a
          directory path.
        </p>

        <p>To do so, we'll:</p>
        <ul>
          <li>
            Get the class names using <code>os.scandir()</code> to traverse a
            target directory (ideally the directory is in standard image
            classification format).
          </li>
          <li>
            Raise an error if the class names aren't found (if this happens,
            there might be something wrong with the directory structure).
          </li>
          <li>
            Turn the class names into a dictionary of numerical labels, one for
            each class.
          </li>
        </ul>
        <p>
          Let's see a small example of step 1 before we write the full function.
        </p>

        <CodeIOBlock
          inputCode={`# Setup path for target directory
target_directory = train_dir
print(f"Target directory: {target_directory}")

# Get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path / "train"))])
print(f"Class names found: {class_names_found}")`}
          outputCode={`Target directory: data/pizza_steak_sushi/train
Class names found: ['pizza', 'steak', 'sushi']`}
        />

        <p>Excellent! How about we turn it into a full function?</p>

        <CodeIOBlock
          inputCode={`# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx`}
        />

        <p>Looking good! Now let's test out our find_classes() function.</p>
        <CodeIOBlock
          inputCode={`find_classes(train_dir)`}
          outputCode={`(['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2})`}
        />

        <p>Woohoo! Looking good!</p>
        <h2>5.2 Create a custom Dataset to replicate ImageFolder</h2>
        <p>Now we're ready to build our own custom Dataset.</p>

        <p>
          We'll build one to replicate the functionality of{" "}
          <code>torchvision.datasets.ImageFolder()</code>.
        </p>

        <p>
          This will be good practice, plus, it'll reveal a few of the required
          steps to make your own custom Dataset.
        </p>

        <p>It'll be a fair bit of code... but nothing we can't handle!</p>

        <p>Let's break it down:</p>
        <ul>
          <li>
            Subclass <code>torch.utils.data.Dataset</code>.
          </li>
          <li>
            Initialize our subclass with a <code>targ_dir</code> parameter (the
            target data directory) and <code>transform</code> parameter (so we
            have the option to transform our data if needed).
          </li>
          <li>
            Create several attributes for paths (the paths of our target
            images), transform (the transforms we might like to use, this can be
            None), classes and class_to_idx (from our{" "}
            <code>find_classes()</code> function).
          </li>
          <li>
            Create a function to load images from file and return them, this
            could be using PIL or <code>torchvision.io</code> (for input/output
            of vision data).
          </li>
          <li>
            Overwrite the <code>__len__</code> method of{" "}
            <code>torch.utils.data.Dataset</code> to return the number of
            samples in the Dataset, this is recommended but not required. This
            is so you can call <code>len(Dataset)</code>.
          </li>
          <li>
            Overwrite the <code>__getitem__</code> method of{" "}
            <code>torch.utils.data.Dataset</code> to return a single sample from
            the Dataset, this is required.
          </li>
        </ul>

        <p>Let's do it!</p>
        <CodeIOBlock
          inputCode={`# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)`}
        />
        <p>Woah! That was a lot of code just to load images.</p>

        <p>
          This is one of the downsides of creating your own custom Datasets.
        </p>

        <p>
          But the good part is — once it's done, you can reuse it. Just move the
          code into a file like <code>data_loader.py</code> and use it in future
          projects.
        </p>

        <p>
          Now, before we test our new <code>ImageFolderCustom</code> class,
          let's set up some transforms to prepare our images.
        </p>

        <CodeIOBlock
          inputCode={`# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])`}
        />
        <p>Now comes the exciting part!</p>

        <p>
          Let’s turn our training images (from <code>train_dir</code>) and
          testing images (from <code>test_dir</code>) into Datasets using our
          custom <code>ImageFolderCustom</code> class.
        </p>
        <CodeIOBlock
          inputCode={`train_data_custom = ImageFolderCustom(targ_dir=train_dir, 
                                      transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, 
                                     transform=test_transforms)
train_data_custom, test_data_custom`}
          outputCode={`(<__main__.ImageFolderCustom at 0x7f5461f70c70>,
 <__main__.ImageFolderCustom at 0x7f5461f70c40>)`}
        />
        <p>Great! No errors — that’s a good sign.</p>

        <p>
          Now let’s confirm everything works by using <code>len()</code> on our
          new Datasets and checking the <code>classes</code> and{" "}
          <code>class_to_idx</code> attributes.
        </p>

        <CodeIOBlock
          inputCode={`len(train_data_custom), len(test_data_custom)`}
          outputCode={`(225, 75)`}
        />

        <CodeIOBlock
          inputCode={`train_data_custom.classes`}
          outputCode={`['pizza', 'steak', 'sushi']`}
        />

        <CodeIOBlock
          inputCode={`train_data_custom.class_to_idx`}
          outputCode={`{'pizza': 0, 'steak': 1, 'sushi': 2}`}
        />

        <code>
          len(test_data_custom) == len(test_data) and len(test_data_custom) ==
          len(test_data)
        </code>
        <p>Yes!!!</p>

        <p>It looks like our custom Dataset is working perfectly.</p>

        <p>
          As a final check, we could compare it with the Dataset created using{" "}
          <code>torchvision.datasets.ImageFolder()</code> to make sure
          everything matches.
        </p>

        <CodeIOBlock
          inputCode={`# Check for equality amongst our custom Dataset and ImageFolder Dataset
print((len(train_data_custom) == len(train_data)) & (len(test_data_custom) == len(test_data)))
print(train_data_custom.classes == train_data.classes)
print(train_data_custom.class_to_idx == train_data.class_to_idx)`}
          outputCode={`True
True
True`}
        />
        <p>Ho ho!</p>

        <p>
          Look at us go — three <code>True</code>'s in a row!
        </p>

        <p>You really can't get much better than that.</p>

        <p>
          Now let’s step it up and plot some random images to test if our{" "}
          <code>__getitem__</code> method is working correctly.
        </p>

        <h2>5.3 Create a function to display random images</h2>
        <p>You know what time it is!</p>

        <p>
          Time to become a data explorer and <strong>visualize</strong>,{" "}
          <strong>visualize</strong>, <strong>visualize</strong>!
        </p>

        <p>
          We’ll make a helper function called{" "}
          <code>display_random_images()</code> to show images from our Dataset.
        </p>

        <p>This function will:</p>
        <ul>
          <li>
            Take in a Dataset, the class names, number of images to show (
            <code>n</code>), and a random seed.
          </li>
          <li>
            Limit <code>n</code> to a maximum of 10.
          </li>
          <li>
            Use the seed to make sure we get the same images each time (if seed
            is set).
          </li>
          <li>
            Pick <code>n</code> random sample indexes using{" "}
            <code>random.sample()</code>.
          </li>
          <li>Set up a matplotlib plot.</li>
          <li>Loop through the samples and plot them using matplotlib.</li>
          <li>
            Ensure each image is in HWC format (height, width, color channels)
            for proper display.
          </li>
        </ul>

        <CodeIOBlock
          inputCode={`# 1. Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)`}
        />
        <p>What a great-looking function!</p>

        <p>
          Let’s give it a test using the Dataset we made earlier with{" "}
          <code>torchvision.datasets.ImageFolder()</code>.
        </p>

        <CodeIOBlock
          inputCode={`# Display random images from ImageFolder created Dataset
display_random_images(train_data, 
                      n=5, 
                      classes=class_names,
                      seed=None)`}
        />

        <img src={foods} className="centered-image" />
        <p>Nice!!!</p>

        <p>
          Looks like our <code>ImageFolderCustom</code> is working just the way
          we want it to.
        </p>

        <h2>Turn custom loaded images into DataLoader's</h2>
        <p>
          We've got a way to turn our raw images into <code>Dataset</code>'s
          (features mapped to labels or X's mapped to y's) using our{" "}
          <code>ImageFolderCustom</code> class.
        </p>

        <p>
          Now how can we turn these custom <code>Dataset</code>'s into{" "}
          <code>DataLoader</code>'s?
        </p>

        <p>
          If you guessed by using <code>torch.utils.data.DataLoader()</code>,
          you're right!
        </p>

        <p>
          Since our custom <code>Dataset</code> is a subclass of{" "}
          <code>torch.utils.data.Dataset</code>, we can plug it directly into{" "}
          <code>DataLoader()</code>.
        </p>

        <p>
          We'll follow similar steps as before, but this time with our own
          custom-built <code>Dataset</code>'s.
        </p>

        <CodeIOBlock
          inputCode={`# Turn train and test custom Dataset's into DataLoader's
from torch.utils.data import DataLoader
train_dataloader_custom = DataLoader(dataset=train_data_custom, # use custom created train Dataset
                                     batch_size=1, # how many samples per batch?
                                     num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

test_dataloader_custom = DataLoader(dataset=test_data_custom, # use custom created test Dataset
                                    batch_size=1, 
                                    num_workers=0, 
                                    shuffle=False) # don't usually need to shuffle testing data

train_dataloader_custom, test_dataloader_custom`}
          outputCode={`(<torch.utils.data.dataloader.DataLoader at 0x7f5460ab8400>,
 <torch.utils.data.dataloader.DataLoader at 0x7f5460ab8490>)`}
        />
        <p>Do the shapes of the samples look the same?</p>

        <CodeIOBlock
          inputCode={`# Get image and label from custom DataLoader
img_custom, label_custom = next(iter(train_dataloader_custom))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label_custom.shape}")`}
          outputCode={`Image shape: torch.Size([1, 3, 64, 64]) -> [batch_size, color_channels, height, width]
Label shape: torch.Size([1])`}
        />

        <p>They sure do!</p>

        <p>
          Now let’s explore some other ways we can transform our image data.
        </p>

        <h2>6. Other forms of transforms (data augmentation)</h2>
        <p>
          We've seen a couple of transforms on our data already, but there are
          many more.
        </p>

        <p>
          You can check them out in the{" "}
          <a
            href="https://pytorch.org/vision/stable/transforms.html"
            target="_blank"
          >
            torchvision.transforms documentation
          </a>
          .
        </p>

        <p>
          Transforms alter your images in different ways. This could mean
          converting images into tensors (as we’ve seen), cropping, randomly
          erasing portions, or rotating them.
        </p>

        <p>These transformations are often part of data augmentation.</p>

        <p>
          Data augmentation increases the diversity of your training set by
          artificially altering the data. This helps the model generalize
          better, meaning it can make more robust predictions on unseen data.
        </p>

        <p>
          For more examples of data augmentation using torchvision.transforms,
          check out PyTorch’s{" "}
          <a
            href="https://pytorch.org/vision/stable/transforms.html"
            target="_blank"
          >
            Illustration of Transforms
          </a>
          .
        </p>

        <p>Now, let’s try one out ourselves.</p>

        <p>
          In machine learning, we harness randomness, and research shows that
          random transforms (like transforms.RandAugment() and
          transforms.TrivialAugmentWide()) often perform better than hand-picked
          ones.
        </p>

        <p>
          The concept behind TrivialAugment is simple: randomly pick a set of
          transforms and apply them at a random magnitude (higher magnitude
          means more intense effects). The PyTorch team used TrivialAugment to
          train their state-of-the-art vision models.
        </p>

        <img src={graph} className="centered-image" />
        <p>
          TrivialAugment was one of the key components in a recent upgrade to
          various state-of-the-art PyTorch vision models.
        </p>

        <p>Now, how about testing it out on some of our own images?</p>

        <p>
          The key parameter to focus on in{" "}
          <code>transforms.TrivialAugmentWide()</code> is{" "}
          <code>num_magnitude_bins=31</code>.
        </p>

        <p>
          This parameter defines how much of a range an intensity value will be
          picked to apply a certain transform. A value of 0 means no range,
          while 31 means the maximum range, giving the highest chance for the
          most intense transformations.
        </p>

        <p>
          We can easily incorporate <code>transforms.TrivialAugmentWide()</code>{" "}
          into <code>transforms.Compose()</code>.
        </p>

        <CodeIOBlock
          inputCode={`from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense 
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
])

# Don't need to perform augmentation on the test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])`}
        />

        <div className="note">
          <strong>Note:</strong>You usually don't perform data augmentation on
          the test set. The idea of data augmentation is to to artificially
          increase the diversity of the training set to better predict on the
          testing set. However, you do need to make sure your test set images
          are transformed to tensors. We size the test images to the same size
          as our training images too, however, inference can be done on
          different size images if necessary (though this may alter
          performance).
        </div>
        <p>
          Beautiful! Now, we've got our training transform (with data
          augmentation) and test transform (without data augmentation).
        </p>

        <p>Let's test out our data augmentation!</p>
        <CodeIOBlock
          inputCode={`# Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Plot random images
plot_transformed_images(
    image_paths=image_path_list,
    transform=train_transforms,
    n=3,
    seed=None
)`}
        />

        <img src={items} className="centered-image" />
        <p>
          Try running the cell above a few times and see how the original image
          changes as it goes through the transform.
        </p>

        <h2>7. Model 0: TinyVGG without data augmentation</h2>
        <p>
          Alright, we've seen how to turn our data from images in folders to
          transformed tensors.
        </p>
        <p>
          Now let's construct a computer vision model to see if we can classify
          if an image is of pizza, steak, or sushi.
        </p>
        <p>
          To begin, we'll start with a simple transform, only resizing the
          images to (64, 64) and turning them into tensors.
        </p>

        <h2>7.1 Creating transforms and loading data for Model 0</h2>

        <CodeIOBlock
          inputCode={`# Create simple transform
simple_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])`}
        />
        <p>Excellent, now we've got a simple transform, let's:</p>
        <ol>
          <li>
            Load the data, turning each of our training and test folders first
            into a Dataset with torchvision.datasets.ImageFolder()
          </li>
          <li>Then into a DataLoader using torch.utils.data.DataLoader().</li>
          <li>
            We'll set the batch_size=32 and num_workers to as many CPUs on our
            machine (this will depend on what machine you're using).
          </li>
        </ol>
        <CodeIOBlock
          inputCode={`# 1. Load and transform data
from torchvision import datasets
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

# 2. Turn data into DataLoaders
import os
from torch.utils.data import DataLoader

# Setup batch size and number of workers 
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

# Create DataLoader's
train_dataloader_simple = DataLoader(train_data_simple, 
                                     batch_size=BATCH_SIZE, 
                                     shuffle=True, 
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=NUM_WORKERS)

train_dataloader_simple, test_dataloader_simple`}
          outputCode={`Creating DataLoader's with batch size 32 and 16 workers.
(<torch.utils.data.dataloader.DataLoader at 0x7f5460ad2f70>,
 <torch.utils.data.dataloader.DataLoader at 0x7f5460ad23d0>)`}
        />

        <p>DataLoader's created! Let's build a model.</p>

        <h2>7.2 Create TinyVGG model class</h2>
        <p>
          In notebook 03, we used the TinyVGG model from the CNN Explainer
          website.
        </p>

        <p>
          Let's recreate the same model, except this time we'll be using color
          images instead of grayscale (in_channels=3 instead of in_channels=1
          for RGB pixels).
        </p>

        <CodeIOBlock
          inputCode={`class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)
model_0`}
          outputCode={`TinyVGG(
  (conv_block_1): Sequential(
    (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=2560, out_features=3, bias=True)
  )
)`}
        />

        <div className="note">
          <strong>Note:</strong> One of the ways to speed up deep learning
          models computing on a GPU is to leverage operator fusion. This means
          in the forward() method in our model above, instead of calling a layer
          block and reassigning x every time, we call each block in succession
          (see the final line of the forward() method in the model above for an
          example). This saves the time spent reassigning x (memory heavy) and
          focuses on only computing on x. See Making Deep Learning Go Brrrr From
          First Principles by Horace He for more ways on how to speed up machine
          learning models.
        </div>

        <p>
          Now, let's test the model with a forward pass using a single image.
          We'll create a random image tensor to simulate a batch of one image.
        </p>

        <h2>7.3 Try a forward pass on a single image (to test the model)</h2>

        <p>
          A good way to test a model is by performing a forward pass on a single
          piece of data.
        </p>

        <p>
          It's also a useful method for testing the input and output shapes of
          the different layers.
        </p>

        <p>To perform a forward pass on a single image, follow these steps:</p>

        <ol>
          <li>Get a batch of images and labels from the DataLoader.</li>
          <li>
            Extract a single image from the batch and use{" "}
            <code>unsqueeze()</code> to ensure the image has a batch size of 1
            (so its shape fits the model).
          </li>
          <li>
            Perform inference on the image (making sure to move it to the target
            device).
          </li>
          <li>
            Print out the process and convert the model's raw output logits to
            prediction probabilities using <code>torch.softmax()</code> (since
            we're working with multi-class data), then convert the prediction
            probabilities to prediction labels using <code>torch.argmax()</code>
            .
          </li>
        </ol>

        <CodeIOBlock
          inputCode={`# 1. Get a batch of images and labels from the DataLoader
img_batch, label_batch = next(iter(train_dataloader_simple))

# 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}\n")

# 3. Perform a forward pass on a single image
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(device))
    
# 4. Print out what's happening and convert model logits -> pred probs -> pred label
print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")`}
          outputCode={`Single image shape: torch.Size([1, 3, 64, 64])

Output logits:
tensor([[0.0578, 0.0634, 0.0352]], device='cuda:0')

Output prediction probabilities:
tensor([[0.3352, 0.3371, 0.3277]], device='cuda:0')

Output prediction label:
tensor([1], device='cuda:0')

Actual label:
2`}
        />

        <p>
          Wonderful, it looks like our model is producing the expected output.
        </p>

        <p>
          You can run the cell above multiple times, with each run predicting a
          different image.
        </p>

        <p>You'll likely notice that the predictions are often incorrect.</p>

        <p>
          This is expected because the model hasn't been trained yet and is
          essentially making guesses using random weights.
        </p>

        <h2>
          7.4 Use torchinfo to get an idea of the shapes going through our model
        </h2>

        <p>
          Printing out our model using <code>print(model)</code> gives us an
          overview of what's happening with the model.
        </p>

        <p>
          We can also print the shapes of our data throughout the{" "}
          <code>forward()</code> method.
        </p>

        <p>
          However, a more useful way to gather information from the model is by
          using <code>torchinfo</code>.
        </p>

        <p>
          <code>torchinfo</code> has a <code>summary()</code> method that takes
          a PyTorch model and an <code>input_shape</code>, then returns details
          of what happens as a tensor moves through the model.
        </p>

        <div className="note">
          <strong>Note:</strong>If you're using Google Colab, you'll need to
          install torchinfo.
        </div>

        <CodeIOBlock
          inputCode={`# Install torchinfo if it's not available, import it if it is
try: 
    import torchinfo
except:
    !pip install torchinfo
    import torchinfo
    
from torchinfo import summary
summary(model_0, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size `}
          outputCode={`==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
TinyVGG                                  [1, 3]                    --
├─Sequential: 1-1                        [1, 10, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 10, 64, 64]           280
│    └─ReLU: 2-2                         [1, 10, 64, 64]           --
│    └─Conv2d: 2-3                       [1, 10, 64, 64]           910
│    └─ReLU: 2-4                         [1, 10, 64, 64]           --
│    └─MaxPool2d: 2-5                    [1, 10, 32, 32]           --
├─Sequential: 1-2                        [1, 10, 16, 16]           --
│    └─Conv2d: 2-6                       [1, 10, 32, 32]           910
│    └─ReLU: 2-7                         [1, 10, 32, 32]           --
│    └─Conv2d: 2-8                       [1, 10, 32, 32]           910
│    └─ReLU: 2-9                         [1, 10, 32, 32]           --
│    └─MaxPool2d: 2-10                   [1, 10, 16, 16]           --
├─Sequential: 1-3                        [1, 3]                    --
│    └─Flatten: 2-11                     [1, 2560]                 --
│    └─Linear: 2-12                      [1, 3]                    7,683
==========================================================================================
Total params: 10,693
Trainable params: 10,693
Non-trainable params: 0
Total mult-adds (M): 6.75
==========================================================================================
Input size (MB): 0.05
Forward/backward pass size (MB): 0.82
Params size (MB): 0.04
Estimated Total Size (MB): 0.91
==========================================================================================`}
        />
        <p>Nice!</p>

        <p>
          The output of <code>torchinfo.summary()</code> provides a lot of
          information about our model.
        </p>

        <p>
          It includes the total number of parameters in the model (
          <code>Total params</code>), as well as the estimated total size (in
          MB), which indicates the size of the model.
        </p>

        <p>
          You can also see how the input and output shapes change as data with a
          given <code>input_size</code> moves through the model.
        </p>

        <p>
          Right now, the number of parameters and the total model size are low.
        </p>

        <p>This is because we're starting with a small model.</p>

        <p>If we need to increase its size later, we can.</p>

        <h2>7.5 Create train & test loop function</h2>
        <p>We've got data and a model.</p>

        <p>
          Now, let's create training and test loop functions to train the model
          on the training data and evaluate it on the testing data.
        </p>

        <p>
          To ensure we can reuse these loops, we'll turn them into functions.
        </p>

        <p>Specifically, we will define three functions:</p>

        <ul>
          <li>
            <code>train_step()</code> – takes a model, a DataLoader, a loss
            function, and an optimizer to train the model on the DataLoader.
          </li>
          <li>
            <code>test_step()</code> – takes a model, a DataLoader, and a loss
            function to evaluate the model on the DataLoader.
          </li>
          <li>
            <code>train()</code> – combines the above two steps for a specified
            number of epochs and returns a results dictionary.
          </li>
        </ul>
        <p>
          Let's start by building the <code>train_step()</code> function.
        </p>

        <p>
          Since we're working with batches from the DataLoader, we'll accumulate
          the model's loss and accuracy values during training (by summing them
          up for each batch) and then adjust them at the end before returning
          the results.
        </p>

        <CodeIOBlock
          inputCode={`def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
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

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc`}
        />
        <p>
          Woohoo! The <code>train_step()</code> function is done.
        </p>

        <p>
          Now, let's do the same for the <code>test_step()</code> function.
        </p>

        <p>
          The main difference here is that the <code>test_step()</code> won't
          take an optimizer and therefore won't perform gradient descent.
        </p>

        <p>
          Since we'll be doing inference, we'll ensure to use the{" "}
          <code>torch.inference_mode()</code> context manager when making
          predictions.
        </p>

        <CodeIOBlock
          inputCode={`def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
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
    return test_loss, test_acc`}
        />
        <p>Excellent!</p>

        <h2>
          7.6 Creating a train() function to combine train_step() and
          test_step()
        </h2>
        <p>
          Now we need a way to combine our <code>train_step()</code> and{" "}
          <code>test_step()</code> functions.
        </p>

        <p>
          To do this, we'll package them together in a <code>train()</code>{" "}
          function.
        </p>

        <p>This function will train the model and evaluate it.</p>

        <p>Specifically, it will:</p>

        <ul>
          <li>
            Take in a model, DataLoaders for the training and test sets, an
            optimizer, a loss function, and the number of epochs for each train
            and test step.
          </li>
          <li>
            Create an empty results dictionary to store <code>train_loss</code>,{" "}
            <code>train_acc</code>, <code>test_loss</code>, and{" "}
            <code>test_acc</code> values (we will fill this up during training).
          </li>
          <li>
            Loop through the training and test steps for a specified number of
            epochs.
          </li>
          <li>Print out the progress at the end of each epoch.</li>
          <li>
            Update the results dictionary with the latest metrics for each
            epoch.
          </li>
          <li>Return the filled results.</li>
        </ul>

        <p>
          To track the number of epochs, we can import <code>tqdm</code> from{" "}
          <code>tqdm.auto</code>. <code>tqdm</code> is one of the most popular
          progress bar libraries for Python, and <code>tqdm.auto</code>{" "}
          automatically selects the appropriate progress bar for your
          environment (e.g., Jupyter Notebook vs. Python script).
        </p>
        <CodeIOBlock
          inputCode={`from tqdm.auto import tqdm

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results`}
        />

        <h2>7.7 Train and Evaluate Model 0</h2>
        <p>
          Alright, alright, alright – we've got all the ingredients we need to
          train and evaluate our model.
        </p>

        <p>
          Now, it's time to put our TinyVGG model, DataLoaders, and the{" "}
          <code>train()</code> function together to see if we can build a model
          capable of distinguishing between pizza, steak, and sushi!
        </p>

        <p>
          Let's recreate <code>model_0</code> (we don't have to, but we'll do it
          for completeness) and then call our <code>train()</code> function,
          passing in the necessary parameters.
        </p>

        <p>
          To keep our experiments quick, we'll train the model for 5 epochs
          (though you could increase this if you'd like).
        </p>

        <p>
          For the optimizer and loss function, we'll use{" "}
          <code>torch.nn.CrossEntropyLoss()</code> (since we're working with
          multi-class classification data) and <code>torch.optim.Adam()</code>{" "}
          with a learning rate of <code>1e-3</code>, respectively.
        </p>

        <p>
          To track how long the training takes, we'll import Python's{" "}
          <code>timeit.default_timer()</code> method to measure the training
          time.
        </p>

        <CodeIOBlock
          inputCode={`# Set random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 5

# Recreate an instance of TinyVGG
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_0 
model_0_results = train(model=model_0, 
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")`}
          outputCode={`0%|          | 0/5 [00:00<?, ?it/s]
Epoch: 1 | train_loss: 1.1078 | train_acc: 0.2578 | test_loss: 1.1360 | test_acc: 0.2604
Epoch: 2 | train_loss: 1.0847 | train_acc: 0.4258 | test_loss: 1.1620 | test_acc: 0.1979
Epoch: 3 | train_loss: 1.1157 | train_acc: 0.2930 | test_loss: 1.1697 | test_acc: 0.1979
Epoch: 4 | train_loss: 1.0956 | train_acc: 0.4141 | test_loss: 1.1384 | test_acc: 0.1979
Epoch: 5 | train_loss: 1.0985 | train_acc: 0.2930 | test_loss: 1.1426 | test_acc: 0.1979
Total training time: 4.935 seconds`}
        />

        <p>Hmm...</p>

        <p>It looks like our model performed pretty poorly.</p>

        <p>But that's okay for now; we'll keep persevering.</p>

        <p>What are some ways we could potentially improve it?</p>

        <h2>7.8 Plot the loss curves of Model 0</h2>
        <p>
          From the printouts of our <code>model_0</code> training, it didn’t
          seem to perform very well.
        </p>

        <p>
          But we can further evaluate it by plotting the model's loss curves.
        </p>

        <p>
          Loss curves display the model's results over time and are a great way
          to observe how the model performs on different datasets (e.g.,
          training and test).
        </p>

        <p>
          Let's create a function to plot the values in our{" "}
          <code>model_0_results</code> dictionary.
        </p>
        <CodeIOBlock
          inputCode={`# Check the model_0_results keys
model_0_results.keys()`}
          outputCode={`dict_keys(['train_loss', 'train_acc', 'test_loss', 'test_acc'])`}
        />
        <p>
          We'll need to extract each of these keys and turn them into a plot.
        </p>

        <CodeIOBlock
          inputCode={`def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();`}
        />

        <p>
          Okay, let's test our <code>plot_loss_curves()</code> function out.
        </p>
        <CodeIOBlock inputCode={`plot_loss_curves(model_0_results)`} />

        <img src={plot} className="centered-image" />
        <p>Woah.</p>

        <p>Looks like things are all over the place...</p>

        <p>
          But we kind of knew that because our model's printout results during
          training didn’t show much promise.
        </p>

        <p>
          You could try training the model for longer and see what happens when
          you plot the loss curve over a longer time horizon.
        </p>

        <h2>8. What should an ideal loss curve look like?</h2>
        <p>
          Looking at training and test loss curves is a great way to check if
          your model is overfitting.
        </p>

        <p>
          An overfitting model performs much better (often by a considerable
          margin) on the training set than on the validation/test set.
        </p>

        <p>
          If your training loss is significantly lower than your test loss, your
          model is overfitting.
        </p>

        <p>
          This means it's learning the patterns in the training data too well,
          but those patterns aren't generalizing to the test data.
        </p>

        <p>
          On the other hand, if both your training and test losses are not as
          low as you'd like, the model is considered underfitting.
        </p>

        <p>
          The ideal situation is for the training and test loss curves to
          closely align with each other.
        </p>
        <img src={fitting} className="centered-image" />
        <p>
          <strong>Left:</strong> If your training and test loss curves aren't as
          low as you'd like, this is considered underfitting.
        </p>

        <p>
          <strong>Middle:</strong> When your test/validation loss is higher than
          your training loss, this is considered overfitting.
        </p>

        <p>
          <strong>Right:</strong> The ideal scenario is when your training and
          test loss curves line up over time. This means your model is
          generalizing well.
        </p>

        <p>
          There are more combinations and different things loss curves can show.
          For more on these, check out Google's{" "}
          <em>Interpreting Loss Curves</em> guide.
        </p>

        <h2>8.1 How to deal with overfitting</h2>
        <p>
          Since the main problem with overfitting is that your model is fitting
          the training data too well, you'll want to use techniques to "reign it
          in."
        </p>

        <p>A common technique for preventing overfitting is regularization.</p>

        <p>
          I like to think of this as "making our models more regular," meaning
          they become capable of fitting more kinds of data.
        </p>

        <p>Let’s discuss a few methods to prevent overfitting.</p>
        <table>
          <thead>
            <tr>
              <th>Method to prevent overfitting</th>
              <th>What is it?</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>Get more data</strong>
              </td>
              <td>
                Having more data gives the model more opportunities to learn
                patterns, which may be more generalizable to new examples.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Simplify your model</strong>
              </td>
              <td>
                If the current model is already overfitting the training data,
                it may be too complicated. This means it’s learning the patterns
                of the data too well and isn’t able to generalize well to unseen
                data. One way to simplify a model is to reduce the number of
                layers or reduce the number of hidden units in each layer.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Use data augmentation</strong>
              </td>
              <td>
                Data augmentation manipulates the training data in a way that
                makes it harder for the model to memorize, as it artificially
                adds more variety to the data. If a model is able to learn
                patterns in augmented data, it may be able to generalize better
                to unseen data.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Use transfer learning</strong>
              </td>
              <td>
                Transfer learning involves leveraging the patterns (also called
                pretrained weights) one model has learned to use as the
                foundation for your own task. For example, we could use a
                computer vision model pretrained on a large variety of images
                and then tweak it to be more specialized for food images.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Use dropout layers</strong>
              </td>
              <td>
                Dropout layers randomly remove connections between hidden layers
                in neural networks, effectively simplifying a model and making
                the remaining connections more robust. See{" "}
                <code>torch.nn.Dropout()</code> for more.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Use learning rate decay</strong>
              </td>
              <td>
                The idea here is to slowly decrease the learning rate as the
                model trains. This is akin to reaching for a coin at the back of
                a couch: the closer you get, the smaller your steps. Similarly,
                as you get closer to convergence, you’ll want your weight
                updates to be smaller.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Use early stopping</strong>
              </td>
              <td>
                Early stopping stops model training before it begins to overfit.
                For example, if the model's loss has stopped decreasing for the
                past 10 epochs (this number is arbitrary), you might want to
                stop the training and use the model weights from 10 epochs
                prior, which had the lowest loss.
              </td>
            </tr>
          </tbody>
        </table>
        <p>
          There are more methods for dealing with overfitting, but these are
          some of the main ones.
        </p>

        <p>
          As you start to build more and more deep models, you'll find that deep
          learning models are very good at learning patterns in data. As a
          result, dealing with overfitting becomes one of the primary challenges
          in deep learning.
        </p>
        <h2>8.2 How to deal with underfitting</h2>
        <p>
          When a model is underfitting, it is considered to have poor predictive
          power on both the training and test sets.
        </p>

        <p>
          In essence, an underfitting model will fail to reduce the loss values
          to a desired level.
        </p>

        <p>
          Right now, looking at our current loss curves, I would consider our
          TinyVGG model, model_0, to be underfitting the data.
        </p>

        <p>
          The main idea behind dealing with underfitting is to increase your
          model's predictive power.
        </p>

        <p>There are several ways to do this.</p>
        <table>
          <thead>
            <tr>
              <th>Method to Prevent Underfitting</th>
              <th>What is it?</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>Add more layers/units to your model</strong>
              </td>
              <td>
                If your model is underfitting, it may not have enough capability
                to learn the required patterns, weights, or representations of
                the data to be predictive. One way to add more predictive power
                to your model is to increase the number of hidden layers/units
                within those layers.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Tweak the learning rate</strong>
              </td>
              <td>
                Perhaps your model's learning rate is too high, and it's trying
                to update its weights too much each epoch, preventing it from
                learning anything. In this case, you might lower the learning
                rate and see what happens.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Use transfer learning</strong>
              </td>
              <td>
                Transfer learning can prevent both overfitting and underfitting.
                It involves using the patterns from a previously working model
                and adjusting them to your own problem.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Train for longer</strong>
              </td>
              <td>
                Sometimes, a model just needs more time to learn representations
                of data. If you find that your model isn't learning anything
                during small experiments, leaving it to train for more epochs
                may result in better performance.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Use less regularization</strong>
              </td>
              <td>
                Your model might be underfitting because you're trying to
                prevent overfitting too much. Reducing the use of regularization
                techniques can help your model fit the data better.
              </td>
            </tr>
          </tbody>
        </table>
        <h2>8.3 The balance between overfitting and underfitting</h2>
        <p>
          None of the methods discussed above are silver bullets, meaning they
          don't always work.
        </p>

        <p>
          Preventing overfitting and underfitting is possibly the most active
          area of machine learning research. Everyone wants their models to fit
          better (less underfitting) but not so well that they don't generalize
          well and perform in the real world (less overfitting).
        </p>

        <p>
          There's a fine line between overfitting and underfitting, as too much
          of each can cause the other.
        </p>

        <p>
          Transfer learning is perhaps one of the most powerful techniques for
          dealing with both overfitting and underfitting in your problems.
          Instead of handcrafting different overfitting and underfitting
          techniques, transfer learning allows you to take a pre-trained model
          from a similar problem space (such as one from{" "}
          <a href="https://paperswithcode.com/sota" target="_blank">
            paperswithcode.com/sota
          </a>{" "}
          or{" "}
          <a href="https://huggingface.co/models" target="_blank">
            Hugging Face models
          </a>
          ) and apply it to your own dataset.
        </p>

        <p>We'll explore the power of transfer learning in a later notebook.</p>

        <h2>9. Model 1: TinyVGG with Data Augmentation</h2>
        <p>Time to try out another model!</p>

        <p>
          This time, let's load in the data and use data augmentation to see if
          it improves our results in any way.
        </p>

        <p>
          First, we'll compose a training transform to include{" "}
          <code>transforms.TrivialAugmentWide()</code> along with resizing and
          converting our images into tensors.
        </p>

        <p>
          We'll do the same for a testing transform, except without the data
          augmentation.
        </p>
        <h2>9.1 Create transform with data augmentation</h2>
        <CodeIOBlock
          inputCode={`# Create training transform with TrivialAugment
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor() 
])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])`}
        />
        <p>Wonderful!</p>

        <p>
          Now let's turn our images into Datasets using{" "}
          <code>torchvision.datasets.ImageFolder()</code> and then into
          DataLoaders with <code>torch.utils.data.DataLoader()</code>.
        </p>

        <h2>9.2 Create train and test Dataset's and DataLoader's</h2>
        <p>
          We'll make sure the train Dataset uses the{" "}
          <code>train_transform_trivial_augment</code> and the test Dataset uses
          the <code>test_transform</code>.
        </p>

        <CodeIOBlock
          inputCode={`# Turn image folders into Datasets
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

train_data_augmented, test_data_simple`}
          outputCode={`(Dataset ImageFolder
     Number of datapoints: 225
     Root location: data/pizza_steak_sushi/train
     StandardTransform
 Transform: Compose(
                Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
                TrivialAugmentWide(num_magnitude_bins=31, interpolation=InterpolationMode.NEAREST, fill=None)
                ToTensor()
            ),
 Dataset ImageFolder
     Number of datapoints: 75
     Root location: data/pizza_steak_sushi/test
     StandardTransform
 Transform: Compose(
                Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
                ToTensor()
            ))`}
        />
        <p>
          And we'll create DataLoaders with a <code>batch_size=32</code> and
          with <code>num_workers</code> set to the number of CPUs available on
          our machine (we can get this using Python's{" "}
          <code>os.cpu_count()</code>).
        </p>

        <CodeIOBlock
          inputCode={`# Turn Datasets into DataLoader's
import os
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

torch.manual_seed(42)
train_dataloader_augmented = DataLoader(train_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=NUM_WORKERS)

train_dataloader_augmented, test_dataloader`}
          outputCode={`(<torch.utils.data.dataloader.DataLoader at 0x7f53c6d64040>,
 <torch.utils.data.dataloader.DataLoader at 0x7f53c0b9de50>)`}
        />

        <h2>9.3 Construct and train Model 1</h2>
        <p>Data loaded!</p>

        <p>
          Now, to build our next model, <code>model_1</code>, we can reuse our{" "}
          <code>TinyVGG</code> class from before.
        </p>

        <p>We'll make sure to send it to the target device.</p>

        <CodeIOBlock
          inputCode={`# Create model_1 and send it to the target device
torch.manual_seed(42)
model_1 = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(train_data_augmented.classes)).to(device)
model_1`}
          outputCode={`TinyVGG(
  (conv_block_1): Sequential(
    (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=2560, out_features=3, bias=True)
  )
)`}
        />

        <p>Model ready!</p>

        <p>Time to train!</p>

        <p>
          Since we've already got functions for the training loop (
          <code>train_step()</code>) and testing loop (<code>test_step()</code>)
          and a function to put them together in <code>train()</code>, let's
          reuse those.
        </p>

        <p>
          We'll use the same setup as <code>model_0</code>, with only the{" "}
          <code>train_dataloader</code> parameter varying:
        </p>

        <ul>
          <li>Train for 5 epochs.</li>
          <li>
            Use <code>train_dataloader=train_dataloader_augmented</code> as the
            training data in <code>train()</code>.
          </li>
          <li>
            Use <code>torch.nn.CrossEntropyLoss()</code> as the loss function
            (since we're working with multi-class classification).
          </li>
          <li>
            Use <code>torch.optim.Adam()</code> with <code>lr=0.001</code> as
            the learning rate for the optimizer.
          </li>
        </ul>

        <CodeIOBlock
          inputCode={`# Set random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 5

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_1
model_1_results = train(model=model_1, 
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")`}
          outputCode={`  0%|          | 0/5 [00:00<?, ?it/s]
Epoch: 1 | train_loss: 1.1074 | train_acc: 0.2500 | test_loss: 1.1058 | test_acc: 0.2604
Epoch: 2 | train_loss: 1.0791 | train_acc: 0.4258 | test_loss: 1.1382 | test_acc: 0.2604
Epoch: 3 | train_loss: 1.0803 | train_acc: 0.4258 | test_loss: 1.1685 | test_acc: 0.2604
Epoch: 4 | train_loss: 1.1285 | train_acc: 0.3047 | test_loss: 1.1623 | test_acc: 0.2604
Epoch: 5 | train_loss: 1.0880 | train_acc: 0.4258 | test_loss: 1.1472 | test_acc: 0.2604
Total training time: 4.924 seconds`}
        />

        <p>Hmm...</p>

        <p>It doesn't look like our model performed very well again.</p>

        <p>Let's check out its loss curves.</p>
        <h2>9.4 Plot the loss curves of Model 1</h2>
        <p>
          Since we've got the results of model_1 saved in a results dictionary,
          model_1_results, we can plot them using plot_loss_curves().
        </p>
        <CodeIOBlock inputCode={`plot_loss_curves(model_1_results)`} />

        <img src={lossplot} className="centered-image" />
        <p>Wow...</p>

        <p>These don't look very good either...</p>

        <p>Is our model underfitting or overfitting?</p>

        <p>Or both?</p>

        <p>
          Ideally, we'd like it to have higher accuracy and lower loss, right?
        </p>

        <p>What are some methods you could try to use to achieve these?</p>
        <h2>10. Compare model results</h2>

        <p>
          Even though our models are performing poorly, we can still compare
          them with code.
        </p>

        <p>First, let’s turn our model results into pandas DataFrames.</p>

        <CodeIOBlock
          inputCode={`import pandas as pd
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)
model_0_df`}
          outputCode={`
train_loss	train_acc	test_loss	test_acc
0	1.107833	0.257812	1.136041	0.260417
1	1.084713	0.425781	1.162014	0.197917
2	1.115697	0.292969	1.169704	0.197917
3	1.095564	0.414062	1.138373	0.197917
4	1.098520	0.292969	1.142631	0.197917
`}
        />

        <p>
          And now we can write some plotting code using <code>matplotlib</code>{" "}
          to visualize the results of <code>model_0</code> and{" "}
          <code>model_1</code> together.
        </p>

        <CodeIOBlock
          inputCode={`# Setup a plot 
plt.figure(figsize=(15, 10))

# Get number of epochs
epochs = range(len(model_0_df))

# Plot train loss
plt.subplot(2, 2, 1)
plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot test loss
plt.subplot(2, 2, 2)
plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot train accuracy
plt.subplot(2, 2, 3)
plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.legend()

# Plot test accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.legend();`}
        />
        <img src={graphs} className="centered-image" />

        <p>
          It looks like our models both performed equally poorly and were kind
          of sporadic (the metrics go up and down sharply).
        </p>
        <p>
          If you built <code>model_2</code>, what would you do differently to
          try and improve performance?
        </p>

        <h2>11. Make a prediction on a custom image</h2>

        <p>
          If you've trained a model on a certain dataset, chances are you'd like
          to make a prediction on your own custom data.
        </p>

        <p>
          In our case, since we've trained a model on <code>pizza</code>,{" "}
          <code>steak</code> and <code>sushi</code> images, how could we use our
          model to make a prediction on one of our own images?
        </p>

        <p>
          To do so, we can load an image and then preprocess it in a way that
          matches the type of data our model was trained on.
        </p>

        <p>
          In other words, we'll have to convert our own custom image to a{" "}
          <code>tensor</code> and make sure it's in the right datatype before
          passing it to our model.
        </p>

        <p>Let's start by downloading a custom image.</p>

        <p>
          Since our model predicts whether an image contains <code>pizza</code>,{" "}
          <code>steak</code> or <code>sushi</code>, let's download a photo of my
          Dad giving two thumbs up to a big pizza from the Learn PyTorch for
          Deep Learning GitHub.
        </p>

        <p>
          We download the image using Python's <code>requests</code> module.
        </p>

        <div className="note">
          <strong>Note:</strong>
          If you're using Google Colab, you can also upload an image to the
          current session by going to the left hand side menu -) Files -) Upload
          to session storage. Beware though, this image will delete when your
          Google Colab session ends.
        </div>

        <CodeIOBlock
          inputCode={`# Download custom image
import requests

# Setup custom image path
custom_image_path = data_path / "04-pizza-dad.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")`}
          outputCode={`
data/04-pizza-dad.jpeg already exists, skipping download.`}
        />

        <h2>11.1 Loading in a custom image with PyTorch</h2>

        <p>Excellent!</p>

        <p>
          Looks like we've got a custom image downloaded and ready to go at{" "}
          <code>data/04-pizza-dad.jpeg</code>.
        </p>

        <p>Time to load it in.</p>

        <p>
          PyTorch's <code>torchvision</code> has several input and output ("IO"
          or "io" for short) methods for reading and writing images and video in{" "}
          <code>torchvision.io</code>.
        </p>

        <p>
          Since we want to load in an image, we'll use{" "}
          <code>torchvision.io.read_image()</code>.
        </p>

        <p>
          This method will read a JPEG or PNG image and turn it into a 3
          dimensional RGB or grayscale <code>torch.Tensor</code> with values of
          datatype <code>uint8</code> in range <code>[0, 255]</code>.
        </p>

        <p>Let's try it out.</p>

        <CodeIOBlock
          inputCode={`import torchvision

# Read in custom image
custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))

# Print out image data
print(f"Custom image tensor:\n{custom_image_uint8}\n")
print(f"Custom image shape: {custom_image_uint8.shape}\n")
print(f"Custom image dtype: {custom_image_uint8.dtype}")`}
          outputCode={`Custom image tensor:
tensor([[[154, 173, 181,  ...,  21,  18,  14],
         [146, 165, 181,  ...,  21,  18,  15],
         [124, 146, 172,  ...,  18,  17,  15],
         ...,
         [ 72,  59,  45,  ..., 152, 150, 148],
         [ 64,  55,  41,  ..., 150, 147, 144],
         [ 64,  60,  46,  ..., 149, 146, 143]],

        [[171, 190, 193,  ...,  22,  19,  15],
         [163, 182, 193,  ...,  22,  19,  16],
         [141, 163, 184,  ...,  19,  18,  16],
         ...,
         [ 55,  42,  28,  ..., 107, 104, 103],
         [ 47,  38,  24,  ..., 108, 104, 102],
         [ 47,  43,  29,  ..., 107, 104, 101]],

        [[119, 138, 147,  ...,  17,  14,  10],
         [111, 130, 145,  ...,  17,  14,  11],
         [ 87, 111, 136,  ...,  14,  13,  11],
         ...,
         [ 35,  22,   8,  ...,  52,  52,  48],
         [ 27,  18,   4,  ...,  50,  49,  44],
         [ 27,  23,   9,  ...,  49,  46,  43]]], dtype=torch.uint8)

Custom image shape: torch.Size([3, 4032, 3024])

Custom image dtype: torch.uint8`}
        />
        <p>
          Nice! Looks like our image is in <code>tensor</code> format, however,
          is this image format compatible with our model?
        </p>

        <p>
          Our <code>custom_image</code> tensor is of datatype{" "}
          <code>torch.uint8</code> and its values are between{" "}
          <code>[0, 255]</code>.
        </p>

        <p>
          But our model takes image tensors of datatype{" "}
          <code>torch.float32</code> and with values between <code>[0, 1]</code>
          .
        </p>

        <p>
          So before we use our custom image with our model, we'll need to
          convert it to the same format as the data our model is trained on.
        </p>

        <p>If we don't do this, our model will error.</p>
        <CodeIOBlock
          inputCode={`# Try to make a prediction on image in uint8 format (this will error)
model_1.eval()
with torch.inference_mode():
    model_1(custom_image_uint8.to(device))`}
        />

        <img src={error} className="centered-image" />
        <p>What now?</p>

        <p>It looks like we're getting a shape error.</p>

        <p>Why might this be?</p>

        <p>
          We converted our custom image to be the same size as the images our
          model was trained on...
        </p>

        <p>Oh wait...</p>

        <p>There's one dimension we forgot about.</p>

        <p>
          The <code>batch size</code>.
        </p>

        <p>
          Our model expects image tensors with a <code>batch size</code>{" "}
          dimension at the start (<code>NCHW</code> where <code>N</code> is the
          batch size).
        </p>

        <p>
          Except our <code>custom image</code> is currently only{" "}
          <code>CHW</code>.
        </p>

        <p>
          We can add a <code>batch size</code> dimension using{" "}
          <code>torch.unsqueeze(dim=0)</code> to add an extra dimension to our
          image and finally make a prediction.
        </p>

        <p>
          Essentially we'll be telling our model to predict on a single image
          (an image with a <code>batch_size</code> of 1).
        </p>

        <CodeIOBlock
          inputCode={`model_1.eval()
with torch.inference_mode():
    # Add an extra dimension to image
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    
    # Print out different shapes
    print(f"Custom image transformed shape: {custom_image_transformed.shape}")
    print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")
    
    # Make a prediction on image with an extra dimension
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))`}
          outputCode={`Custom image transformed shape: torch.Size([3, 64, 64])
Unsqueezed custom image shape: torch.Size([1, 3, 64, 64])`}
        />

        <p>Yes!!! It looks like it worked!</p>

        <div className="note">
          <strong>Note:</strong>
          What we've just gone through are three of the classical and most
          common deep learning and PyTorch issues: Wrong datatypes - our model
          expects torch.float32 where our original custom image was uint8. Wrong
          device - our model was on the target device (in our case, the GPU)
          whereas our target data hadn't been moved to the target device yet.
          Wrong shapes - our model expected an input image of shape [N, C, H, W]
          or [batch_size, color_channels, height, width] whereas our custom
          image tensor was of shape [color_channels, height, width]. Keep in
          mind, these errors aren't just for predicting on custom images. They
          will be present with almost every kind of data type (text, audio,
          structured data) and problem you work with.
        </div>

        <p>Now let's take a look at our model's predictions.</p>

        <CodeIOBlock
          inputCode={`custom_image_pred`}
          outputCode={`tensor([[ 0.1172,  0.0160, -0.1425]], device='cuda:0')`}
        />

        <p>
          Alright, these are still in <code>logit</code> form (the raw outputs
          of a model are called <code>logits</code>).
        </p>

        <p>
          Let's convert them from <code>logits</code> →{" "}
          <code>prediction probabilities</code> → <code>prediction labels</code>
          .
        </p>
        <CodeIOBlock
          inputCode={`# Print out prediction logits
print(f"Prediction logits: {custom_image_pred}")

# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(f"Prediction probabilities: {custom_image_pred_probs}")

# Convert prediction probabilities -> prediction labels
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(f"Prediction label: {custom_image_pred_label}")`}
          outputCode={`Prediction logits: tensor([[ 0.1172,  0.0160, -0.1425]], device='cuda:0')
Prediction probabilities: tensor([[0.3738, 0.3378, 0.2883]], device='cuda:0')
Prediction label: tensor([0], device='cuda:0')`}
        />

        <p>Alright!</p>

        <p>Looking good.</p>

        <p>
          But of course our <code>prediction label</code> is still in
          index/tensor form.
        </p>

        <p>
          We can convert it to a <code>string class name prediction</code> by
          indexing on the <code>class_names</code> list.
        </p>

        <CodeIOBlock
          inputCode={`# Find the predicted label
custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
custom_image_pred_class`}
          outputCode={`'pizza'`}
        />
        <p>Wow.</p>

        <p>
          It looks like the <code>model</code> gets the <code>prediction</code>{" "}
          right, even though it was performing poorly based on our{" "}
          <code>evaluation metrics</code>.
        </p>
        <div className="note">
          <strong>Note:</strong> The model in its current form will predict
          "pizza", "steak" or "sushi" no matter what image it's given. If you
          wanted your model to predict on a different class, you'd have to train
          it to do so.
        </div>
        <p>
          But if we check the <code>custom_image_pred_probs</code>, we'll notice
          that the <code>model</code> gives almost equal weight (the values are
          similar) to every <code>class</code>.
        </p>

        <CodeIOBlock
          inputCode={`# The values of the prediction probabilities are quite similar
custom_image_pred_probs`}
          outputCode={`tensor([[0.3738, 0.3378, 0.2883]], device='cuda:0')`}
        />

        <p>
          Having prediction probabilities this similar could mean a couple of
          things:
        </p>

        <ul>
          <li>
            The <code>model</code> is trying to predict all three{" "}
            <code>classes</code> at the same time (there may be an{" "}
            <code>image</code> containing <code>pizza</code>, <code>steak</code>{" "}
            and <code>sushi</code>).
          </li>
          <li>
            The <code>model</code> doesn't really know what it wants to predict
            and is in turn just assigning similar values to each of the{" "}
            <code>classes</code>.
          </li>
        </ul>

        <p>
          Our case is number 2, since our <code>model</code> is poorly trained,
          it is basically guessing the <code>prediction</code>.
        </p>

        <h2>
          11.3 Putting custom image prediction together: building a function
        </h2>

        <p>
          Doing all of the above steps every time you'd like to make a{" "}
          <code>prediction</code> on a custom <code>image</code> would quickly
          become tedious.
        </p>

        <p>
          So let's put them all together in a function we can easily use over
          and over again.
        </p>

        <p>
          Specifically, let's make a <code>function</code> that:
        </p>

        <ul>
          <li>
            Takes in a target <code>image path</code> and converts to the right
            datatype for our <code>model</code> (<code>torch.float32</code>).
          </li>
          <li>
            Makes sure the target <code>image</code> pixel values are in the
            range <code>[0, 1]</code>.
          </li>
          <li>
            Transforms the target <code>image</code> if necessary.
          </li>
          <li>
            Makes sure the <code>model</code> is on the target{" "}
            <code>device</code>.
          </li>
          <li>
            Makes a <code>prediction</code> on the target <code>image</code>{" "}
            with a trained <code>model</code> (ensuring the <code>image</code>{" "}
            is the right size and on the same <code>device</code> as the{" "}
            <code>model</code>).
          </li>
          <li>
            Converts the <code>model</code>'s output <code>logits</code> to{" "}
            <code>prediction probabilities</code>.
          </li>
          <li>
            Converts the <code>prediction probabilities</code> to{" "}
            <code>prediction labels</code>.
          </li>
          <li>
            Plots the target <code>image</code> alongside the{" "}
            <code>model prediction</code> and{" "}
            <code>prediction probability</code>.
          </li>
        </ul>

        <p>A fair few steps but we've got this!</p>

        <CodeIOBlock
          inputCode={`def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 
    
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # 4. Make sure the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False);`}
        />

        <p>What a nice looking function, let's test it out.</p>

        <CodeIOBlock
          inputCode={`# Pred on our custom image
pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)`}
        />

        <img src={man} className="centered-image" />

        <p>Two thumbs up again!</p>

        <p>
          Looks like our <code>model</code> got the <code>prediction</code>{" "}
          right just by guessing.
        </p>

        <p>
          This won't always be the case with other <code>images</code> though...
        </p>

        <p>
          The <code>image</code> is pixelated too because we resized it to{" "}
          <code>[64, 64]</code> using <code>custom_image_transform</code>.
        </p>

        <h2>Main takeaways</h2>
        <p>We've covered a fair bit in this module.</p>

        <ul>
          <li>
            PyTorch has many in-built functions to deal with all kinds of data,
            from vision to text to audio to recommendation systems.
          </li>
          <li>
            If PyTorch's built-in data loading functions don't suit your
            requirements, you can write code to create your own custom datasets
            by subclassing <code>torch.utils.data.Dataset</code>.
          </li>
          <li>
            <code>torch.utils.data.DataLoader</code>'s in PyTorch help turn your
            Dataset's into iterables that can be used when training and testing
            a model.
          </li>
          <li>
            A lot of machine learning is dealing with the balance between
            overfitting and underfitting (we discussed different methods for
            each above, so a good exercise would be to research more and write
            code to try out the different techniques).
          </li>
          <li>
            Predicting on your own custom data with a trained model is possible,
            as long as you format the data into a similar format to what the
            model was trained on. Make sure you take care of the three big
            PyTorch and deep learning errors:
            <ul>
              <li>
                Wrong datatypes - Your model expected <code>torch.float32</code>{" "}
                when your data is <code>torch.uint8</code>.
              </li>
              <li>
                Wrong data shapes - Your model expected{" "}
                <code>[batch_size, color_channels, height, width]</code> when
                your data is <code>[color_channels, height, width]</code>.
              </li>
              <li>
                Wrong devices - Your model is on the GPU but your data is on the
                CPU.
              </li>
            </ul>
          </li>
        </ul>

        <h1>The End</h1>
      </section>
    </div>
  );
};

export default customDatasets;
