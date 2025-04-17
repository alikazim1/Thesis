import React from "react";
import { FaBrain, FaCogs, FaCube, FaRocket } from "react-icons/fa";
import img from "./Img/img.png";
import learn from "./Img/learn.png";
import head from "./Img/head.png";
import CodeIOBlock from "./CodeIOBlock.jsx";
import sushi from "./Img/sushi.png";
import prin from "./Img/print.png";
import happy from "./Img/happy.png";
import info from "./Img/info.png";
import graph from "./Img/graph.png";
import layer from "./Img/layer.png";
import writing from "./Img/writing.png";
import input from "./Img/input.png";
import huggingface from "./Img/huggingface.png";

const transferLearning = () => {
  return (
    <div className="content">
      <h1 className="page-title">07. Transfer Learning</h1>

      <section>
        <p>
          We've built a few models by hand so far, but their performance has
          been poor.
        </p>
        <p>
          You might be wondering, is there a well-performing model already
          available for our problem?
        </p>
        <p>In the world of deep learning, the answer is often yes.</p>
        <p>
          We'll explore how to leverage a powerful technique called{" "}
          <strong>transfer learning</strong> to improve our models.
        </p>

        <h2>What is transfer learning?</h2>
        <p>
          Transfer learning lets us use the patterns (also called weights) that
          another model has learned from a different task and apply them to our
          own problem.
        </p>
        <p>
          For example, we can use the patterns a computer vision model has
          learned from datasets like ImageNet (which contains millions of
          images) to enhance our FoodVision Mini model.
        </p>
        <p>
          Or we might take the patterns from a language model (trained on large
          amounts of text) and use them as the foundation for a model that
          classifies different text samples.
        </p>
        <p>
          The main idea stays the same: take a high-performing model and apply
          it to your own problem.
        </p>

        <img src={img} className="centered-image" />

        <p>
          Here’s an example of transfer learning in action for both computer
          vision and natural language processing (NLP).
        </p>
        <p>
          In computer vision, a model might learn patterns from millions of
          images in <code>ImageNet</code>, then use those patterns to solve a
          different task.
        </p>
        <p>
          For NLP, a language model could learn the structure of language by
          reading all of <code>Wikipedia</code> (and more), then apply that
          knowledge to another problem.
        </p>

        <h2>Why use transfer learning?</h2>
        <p>There are two main benefits of using transfer learning:</p>
        <ul>
          <li>
            You can use an existing model (often a neural network) that’s
            already proven effective on similar problems.
          </li>
          <li>
            You can take advantage of a model that has already learned useful
            patterns from data similar to yours, often leading to great results
            with less custom data.
          </li>
        </ul>

        <img src={learn} className="centered-image" />

        <p>
          We'll put these ideas to the test with our FoodVision Mini project.
          We'll use a computer vision model pretrained on ImageNet and try to
          apply its learned features to classify images of pizza, steak, and
          sushi.
        </p>

        <p>
          Both research and real-world experience support using transfer
          learning.
        </p>

        <p>
          In fact, a recent machine learning research paper suggests that
          practitioners should use transfer learning whenever possible.
        </p>

        <img src={writing} className="centered-image" />

        <p>
          A study comparing the effectiveness of training from scratch versus
          using transfer learning from a practitioner's perspective found that
          transfer learning was significantly more beneficial in terms of cost
          and time. This conclusion comes from the paper{" "}
          <em>
            How to Train Your ViT? Data, Augmentation, and Regularization in
            Vision Transformers
          </em>{" "}
          (section 6, conclusion).
        </p>

        <p>
          Additionally, Jeremy Howard, the founder of fastai, is a strong
          advocate for transfer learning.
        </p>

        <p>
          The things that really make a difference (
          <strong>
            <a href="https://www.fast.ai/">transfer learning</a>
          </strong>
          ), if we can do better at transfer learning, it’s this world-changing
          thing. Suddenly, lots more people can do world-class work with fewer
          resources and less data. — Jeremy Howard on the Lex Fridman Podcast
        </p>

        <h2>Where to find pretrained models</h2>

        <p>The world of deep learning is an amazing place.</p>

        <p>So amazing that many people around the world share their work.</p>

        <p>
          Often, code and pretrained models for the latest state-of-the-art
          research is released within a few days of publishing.
        </p>

        <p>
          And there are several places you can find pretrained models to use for
          your own problems.
        </p>
        <table>
          <tr>
            <th>Location</th>
            <th>What's there?</th>
            <th>Link(s)</th>
          </tr>
          <tr>
            <td>PyTorch domain libraries</td>
            <td>
              Each of the PyTorch domain libraries (torchvision, torchtext) come
              with pretrained models of some form. The models there work right
              within PyTorch.
            </td>
            <td>
              <a href="https://pytorch.org/vision/stable/models.html">
                torchvision.models
              </a>
              ,{" "}
              <a href="https://pytorch.org/text/stable/models.html">
                torchtext.models
              </a>
              ,{" "}
              <a href="https://pytorch.org/audio/stable/models.html">
                torchaudio.models
              </a>
              ,{" "}
              <a href="https://pytorch.org/recipies/stable/models.html">
                torchrec.models
              </a>
            </td>
          </tr>
          <tr>
            <td>HuggingFace Hub</td>
            <td>
              A series of pretrained models on many different domains (vision,
              text, audio and more) from organizations around the world. There's
              plenty of different datasets too.
            </td>
            <td>
              <a href="https://huggingface.co/models">huggingface.co/models</a>,{" "}
              <a href="https://huggingface.co/datasets">
                huggingface.co/datasets
              </a>
            </td>
          </tr>
          <tr>
            <td>timm (PyTorch Image Models) library</td>
            <td>
              Almost all of the latest and greatest computer vision models in
              PyTorch code as well as plenty of other helpful computer vision
              features.
            </td>
            <td>
              <a href="https://github.com/rwightman/pytorch-image-models">
                GitHub - timm
              </a>
            </td>
          </tr>
          <tr>
            <td>Paperswithcode</td>
            <td>
              A collection of the latest state-of-the-art machine learning
              papers with code implementations attached. You can also find
              benchmarks here of model performance on different tasks.
            </td>
            <td>
              <a href="https://paperswithcode.com/">paperswithcode.com</a>
            </td>
          </tr>
        </table>

        <img src={huggingface} className="centered-image" />
        <p>
          With access to such high-quality resources as above, it should be
          common practice at the start of every deep learning problem you take
          on to ask,{" "}
          <strong>"Does a pretrained model exist for my problem?"</strong>
        </p>

        <h2>What we're going to cover</h2>
        <p>
          We're going to take a pretrained model from{" "}
          <code>torchvision.models</code> and customise it to work on (and
          hopefully improve) our FoodVision Mini problem.
        </p>

        <table>
          <thead>
            <tr>
              <th>Topic</th>
              <th>Contents</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0. Getting setup</td>
              <td>
                We've written a fair bit of useful code over the past few
                sections, let's download it and make sure we can use it again.
              </td>
            </tr>
            <tr>
              <td>1. Get data</td>
              <td>
                Let's get the pizza, steak and sushi image classification
                dataset we've been using to try and improve our model's results.
              </td>
            </tr>
            <tr>
              <td>2. Create Datasets and DataLoaders</td>
              <td>
                We'll use the <code>data_setup.py</code> script we wrote in
                chapter 05. PyTorch Going Modular to setup our DataLoaders.
              </td>
            </tr>
            <tr>
              <td>3. Get and customise a pretrained model</td>
              <td>
                Here we'll download a pretrained model from{" "}
                <code>torchvision.models</code> and customise it to our own
                problem.
              </td>
            </tr>
            <tr>
              <td>4. Train model</td>
              <td>
                Let's see how the new pretrained model goes on our pizza, steak,
                sushi dataset. We'll use the training functions we created in
                the previous chapter.
              </td>
            </tr>
            <tr>
              <td>5. Evaluate the model by plotting loss curves</td>
              <td>
                How did our first transfer learning model go? Did it overfit or
                underfit?
              </td>
            </tr>
            <tr>
              <td>6. Make predictions on images from the test set</td>
              <td>
                It's one thing to check out a model's evaluation metrics but
                it's another thing to view its predictions on test samples,
                let's visualize, visualize, visualize!
              </td>
            </tr>
          </tbody>
        </table>

        <h2>0. Getting setup</h2>

        <p>
          Let's get started by importing and downloading the required modules
          for this section.
        </p>

        <p>
          To save us from writing extra code, we'll be leveraging some of the
          Python scripts (such as <code>data_setup.py</code> and{" "}
          <code>engine.py</code>) we created in the previous section, 05.
          PyTorch Going Modular.
        </p>

        <p>
          Specifically, we’ll download the <code>going_modular</code> directory
          from the{" "}
          <a href="https://github.com/pytorch-deep-learning">
            pytorch-deep-learning repository
          </a>{" "}
          (if we don’t already have it).
        </p>

        <p>
          We'll also install the <code>torchinfo</code> package if it’s not
          available.
        </p>

        <p>
          <code>torchinfo</code> will help later on to provide a visual
          representation of our model.
        </p>

        <div className="note">
          <strong>Note:</strong>As of June 2022, this notebook uses the nightly
          versions of torch and torchvision as torchvision v0.13+ is required
          for using the updated multi-weights API. You can install these using
          the command below.
        </div>

        <CodeIOBlock
          inputCode={`# For this notebook to run with updated APIs, we need torch 1.12+ and torchvision 0.13+
try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
    assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision versions not as required, installing nightly versions.")
    !pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")`}
          outputCode={`
torch version: 1.13.0.dev20220620+cu113
torchvision version: 0.14.0.dev20220620+cu113`}
        />

        <CodeIOBlock
          inputCode={`# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

# Try to get torchinfo, install it if it doesn't work
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    !pip install -q torchinfo
    from torchinfo import summary

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
    from going_modular.going_modular import data_setup, engine
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    !git clone https://github.com/mrdbourke/pytorch-deep-learning
    !mv pytorch-deep-learning/going_modular .
    !rm -rf pytorch-deep-learning
    from going_modular.going_modular import data_setup, engine`}
        />

        <p>Now let's setup device agnostic code.</p>

        <div className="note">
          <strong>Note:</strong>
          <p>
            If you're using Google Colab and don't have a GPU turned on yet,
            it's time to enable one. Go to{" "}
            <code>
              Runtime -) Change runtime type -) Hardware accelerator -) GPU
            </code>
            .
          </p>
        </div>
        <CodeIOBlock
          inputCode={`# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device`}
          outputCode={`'cuda'`}
        />

        <h2>1. Get data</h2>

        <p>Before we can start using transfer learning, we need a dataset.</p>
        <p>
          To see how transfer learning compares to our previous attempts at
          model building, we'll download the same dataset we've been using for
          FoodVision Mini.
        </p>
        <p>
          Let's write some code to download the{" "}
          <code>pizza_steak_sushi.zip</code> dataset from the course GitHub and
          then unzip it.
        </p>
        <p>
          We can also ensure that if we've already got the data, it won't be
          redownloaded.
        </p>

        <CodeIOBlock
          inputCode={`import os
import zipfile

from pathlib import Path

import requests

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

    # Remove .zip file
    os.remove(data_path / "pizza_steak_sushi.zip")`}
        />

        <p>data/pizza_steak_sushi directory exists.</p>
        <p>Excellent!</p>
        <p>
          Now that we have the same dataset we've been using previously, which
          contains a series of images of pizza, steak, and sushi in standard
          image classification format, let's proceed.
        </p>
        <p>Next, we'll create paths to our training and test directories.</p>

        <CodeIOBlock
          inputCode={`# Setup Dirs
train_dir = image_path / "train"
test_dir = image_path / "test"`}
        />

        <h2>2. Create Datasets and DataLoaders</h2>
        <p>
          Since we've downloaded the going_modular directory, we can now use the{" "}
          <code>data_setup.py</code> script we created in section 05. PyTorch
          Going Modular to prepare and set up our DataLoaders.
        </p>
        <p>
          However, since we'll be using a pretrained model from{" "}
          <code>torchvision.models</code>, there's a specific transform we need
          to apply to our images first.
        </p>
        <h2>
          2.1 Creating a transform for torchvision.models (manual creation)
        </h2>

        <div className="note">
          <strong>Note:</strong>As of torchvision v0.13+, there's an update to
          how data transforms can be created using torchvision.models. I've
          called the previous method "manual creation" and the new method "auto
          creation". This notebook showcases both.
        </div>

        <p>
          When using a pretrained model, it's important that your custom data
          going into the model is prepared in the same way as the original
          training data that went into the model.
        </p>

        <p>
          Prior to torchvision v0.13+, to create a transform for a pretrained
          model in <code>torchvision.models</code>, the documentation stated:
        </p>

        <p>
          All pre-trained models expect input images normalized in the same way,
          i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where
          H and W are expected to be at least 224.
        </p>

        <p>
          The images have to be loaded into a range of [0, 1] and then
          normalized using <code>mean = [0.485, 0.456, 0.406]</code> and{" "}
          <code>std = [0.229, 0.224, 0.225]</code>.
        </p>

        <p>You can use the following transform to normalize:</p>

        <pre>
          <code>
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
          </code>
        </pre>

        <p>
          The good news is, we can achieve the above transformations with a
          combination of:
        </p>
        <p>
          <strong>Transform number</strong> <strong>Transform required</strong>{" "}
          <strong>Code to perform transform</strong>
        </p>

        <ul>
          <li>
            <strong>1</strong> Mini-batches of size [batch_size, 3, height,
            width] where height and width are at least 224x224.{" "}
            <code>torchvision.transforms.Resize()</code> to resize images into
            [3, 224, 224] and <code>torch.utils.data.DataLoader()</code> to
            create batches of images.
          </li>
          <li>
            <strong>2</strong> Values between 0 & 1.{" "}
            <code>torchvision.transforms.ToTensor()</code>
          </li>
          <li>
            <strong>3</strong> A mean of [0.485, 0.456, 0.406] (values across
            each colour channel).{" "}
            <code>torchvision.transforms.Normalize(mean=...)</code> to adjust
            the mean of our images.
          </li>
          <li>
            <strong>4</strong> A standard deviation of [0.229, 0.224, 0.225]
            (values across each colour channel).{" "}
            <code>torchvision.transforms.Normalize(std=...)</code> to adjust the
            standard deviation of our images.
          </li>
        </ul>

        <div className="note">
          <strong>Note:</strong>some pretrained models from torchvision.models
          in different sizes to [3, 224, 224], for example, some might take them
          in [3, 240, 240]. For specific input image sizes, see the
          documentation.
        </div>

        <p>
          Let's compose a series of torchvision.transforms to perform the above
          steps.
        </p>

        <CodeIOBlock
          inputCode={`# Create a transforms pipeline manually (required for torchvision < 0.13)
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])`}
        />

        <p>Wonderful!</p>

        <p>
          Now that we have a manually created series of transforms to prepare
          our images, let's create training and testing DataLoaders.
        </p>

        <p>
          We can create these using the <code>create_dataloaders</code> function
          from the <code>data_setup.py</code> script we created in{" "}
          <a href="https://github.com/pytorch-deep-learning">
            05. PyTorch Going Modular Part 2
          </a>
          .
        </p>

        <p>
          We'll set <code>batch_size=32</code> so our model processes
          mini-batches of 32 samples at a time.
        </p>

        <p>
          Additionally, we can transform our images using the transform pipeline
          we created above by setting <code>transform=manual_transforms</code>.
        </p>

        <div className="note">
          <strong>Note:</strong> I've included this manual creation of
          transforms in this notebook because you may come across resources that
          use this style. It's also important to note that because these
          transforms are manually created, they're also infinitely customizable.
          So if you wanted to include data augmentation techniques in your
          transforms pipeline, you could.
        </div>

        <CodeIOBlock
          inputCode={`# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=manual_transforms, # resize, convert images to between 0 & 1 and normalize them
                                                                               batch_size=32) # set mini-batch size to 32

train_dataloader, test_dataloader, class_names`}
          outputCode={`(<torch.utils.data.dataloader.DataLoader at 0x7fa9429a3a60>,
 <torch.utils.data.dataloader.DataLoader at 0x7fa9429a37c0>,
 ['pizza', 'steak', 'sushi'])`}
        />

        <h2>2.2 Creating a transform for torchvision.models (auto creation)</h2>

        <p>
          As previously mentioned, when using a pretrained model, it's crucial
          that your custom data is prepared in the same way as the original
          training data that was used to train the model.
        </p>

        <p>
          Above, we saw how to manually create a transform for a pretrained
          model.
        </p>

        <p>
          However, as of torchvision v0.13+, an automatic transform creation
          feature has been added.
        </p>

        <p>
          When setting up a model from <code>torchvision.models</code> and
          selecting the pretrained model weights you'd like to use, for example,
          if we want to use:
        </p>

        <pre>
          <code>
            weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
          </code>
        </pre>

        <p>Where,</p>

        <p>
          <code>EfficientNet_B0_Weights</code> refers to the model architecture
          weights we'd like to use (there are many different model architecture
          options available in <code>torchvision.models</code>).
        </p>

        <p>
          <strong>DEFAULT</strong> means the best available weights (the best
          performance on ImageNet).
        </p>

        <p>
          <strong>Note:</strong> Depending on the model architecture you choose,
          you may also see other options such as <code>IMAGENET_V1</code> and{" "}
          <code>IMAGENET_V2</code>, where generally the higher version number
          provides better performance. However, if you're aiming for the best
          available, <strong>DEFAULT</strong> is the easiest option. See the{" "}
          <a
            href="https://pytorch.org/vision/stable/models.html"
            target="_blank"
          >
            torchvision.models documentation
          </a>{" "}
          for more.
        </p>

        <CodeIOBlock
          inputCode={`# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
weights`}
          outputCode={`EfficientNet_B0_Weights.IMAGENET1K_V1`}
        />

        <p>
          And now, to access the transforms associated with our weights, we can
          use the <code>transforms()</code> method.
        </p>

        <p>
          This essentially means "get the data transforms that were used to
          train the <code>EfficientNet_B0_Weights</code> on ImageNet".
        </p>

        <CodeIOBlock
          inputCode={`# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()
auto_transforms`}
          outputCode={`ImageClassification(
    crop_size=[224]
    resize_size=[256]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    interpolation=InterpolationMode.BICUBIC
)`}
        />

        <p>
          Notice how <code>auto_transforms</code> is very similar to{" "}
          <code>manual_transforms</code>, with the only difference being that{" "}
          <code>auto_transforms</code> came with the model architecture we
          chose, whereas we had to create <code>manual_transforms</code> by
          hand.
        </p>

        <p>
          The benefit of automatically creating a transform through{" "}
          <code>weights.transforms()</code> is that you ensure you're using the
          same data transformation as the pretrained model used when it was
          trained.
        </p>

        <p>
          However, the tradeoff of using automatically created transforms is a
          lack of customization.
        </p>

        <p>
          We can use <code>auto_transforms</code> to create DataLoaders with{" "}
          <code>create_dataloaders()</code> just as before.
        </p>

        <CodeIOBlock
          inputCode={`# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
                                                                               batch_size=32) # set mini-batch size to 32

train_dataloader, test_dataloader, class_names`}
          outputCode={`(<torch.utils.data.dataloader.DataLoader at 0x7fa942951460>,
 <torch.utils.data.dataloader.DataLoader at 0x7fa942951550>,
 ['pizza', 'steak', 'sushi'])`}
        />

        <h2>3. Getting a pretrained model</h2>
        <p>Alright, here comes the fun part!</p>

        <p>
          Over the past few notebooks, we've been building PyTorch neural
          networks from scratch. While that's a valuable skill to have, our
          models haven't been performing as well as we'd like.
        </p>

        <p>That's where transfer learning comes in.</p>

        <p>
          The whole idea of transfer learning is to take an already
          well-performing model on a problem-space similar to yours and then
          customize it to your use case.
        </p>

        <p>
          Since we're working on a computer vision problem (image classification
          with FoodVision Mini), we can find pretrained classification models in{" "}
          <code>torchvision.models</code>.
        </p>

        <p>
          Exploring the documentation, you'll find plenty of common computer
          vision architecture backbones such as:
        </p>

        <table>
          <tr>
            <th>Architecture Backbone</th>
            <th>Code</th>
          </tr>
          <tr>
            <td>ResNet's</td>
            <td>
              <code>
                torchvision.models.resnet18(), torchvision.models.resnet50()
              </code>
              ...
            </td>
          </tr>
          <tr>
            <td>VGG (similar to what we used for TinyVGG)</td>
            <td>
              <code>torchvision.models.vgg16()</code>
            </td>
          </tr>
          <tr>
            <td>EfficientNet's</td>
            <td>
              <code>
                torchvision.models.efficientnet_b0(),
                torchvision.models.efficientnet_b1()
              </code>
              ...
            </td>
          </tr>
          <tr>
            <td>VisionTransformer (ViT's)</td>
            <td>
              <code>
                torchvision.models.vit_b_16(), torchvision.models.vit_b_32()
              </code>
              ...
            </td>
          </tr>
          <tr>
            <td>ConvNeXt</td>
            <td>
              <code>
                torchvision.models.convnext_tiny(),
                torchvision.models.convnext_small()
              </code>
              ...
            </td>
          </tr>
          <tr>
            <td>More available in torchvision.models</td>
            <td>
              <code>torchvision.models...</code>
            </td>
          </tr>
        </table>

        <h2>3.1 Which pretrained model should you use?</h2>

        <p>It depends on your problem and the device you're working with.</p>

        <p>
          Generally, the higher the number in the model name (e.g.{" "}
          <code>efficientnet_b0()</code> -) <code>efficientnet_b1()</code> -){" "}
          <code>efficientnet_b7()</code>) means better performance but a larger
          model.
        </p>

        <p>You might think better performance is always better, right?</p>

        <p>
          That's true, but some better performing models are too big for certain
          devices.
        </p>

        <p>
          For example, say you'd like to run your model on a mobile device.
          You'll have to take into account the limited compute resources on the
          device, so you'd likely be looking for a smaller model.
        </p>

        <p>
          But if you've got unlimited compute power, as The Bitter Lesson
          states, you'd probably take the biggest, most compute-hungry model you
          can.
        </p>

        <p>
          Understanding this performance vs. speed vs. size tradeoff will come
          with time and practice.
        </p>

        <p>
          For me, I've found a nice balance in the <code>efficientnet_bX</code>{" "}
          models.
        </p>

        <p>
          As of May 2022, Nutrify (the machine learning powered app I'm working
          on) is powered by an <code>efficientnet_b0</code>.
        </p>

        <p>
          Comma.ai (a company that makes open-source self-driving car software)
          uses an <code>efficientnet_b2</code> to learn a representation of the
          road.
        </p>

        <div class="note">
          <strong>Note:</strong> Even though we're using{" "}
          <code>efficientnet_bX</code>, it's important not to get too attached
          to any one architecture, as they are always changing as new research
          gets released. Best to experiment, experiment, experiment, and see
          what works for your problem.
        </div>

        <h2>3.2 Setting up a pretrained model</h2>
        <p>
          The pretrained model we're going to be using is{" "}
          <code>torchvision.models.efficientnet_b0()</code>.
        </p>

        <p>
          The architecture is from the paper{" "}
          <a href="https://arxiv.org/abs/1905.11946" target="_blank">
            <strong>
              EfficientNet: Rethinking Model Scaling for Convolutional Neural
              Networks
            </strong>
          </a>
          .
        </p>

        <img src={input} className="centered-image" />

        <p>
          Example of what we're going to create: a pretrained{" "}
          <code>EfficientNet_B0</code> model from{" "}
          <code>torchvision.models</code> with the output layer adjusted for our
          use case of classifying pizza, steak, and sushi images.
        </p>

        <p>
          We can set up the <code>EfficientNet_B0</code> pretrained ImageNet
          weights using the same code as we used to create the transforms.
        </p>

        <pre>
          <code>
            weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT #
            .DEFAULT = best available weights for ImageNet
          </code>
        </pre>

        <p>
          This means the model has already been trained on millions of images
          and has a strong base representation of image data.
        </p>

        <p>
          The PyTorch version of this pretrained model is capable of achieving
          around 77.7% accuracy across ImageNet's 1000 classes.
        </p>

        <p>We'll also send it to the target device.</p>

        <CodeIOBlock
          inputCode={`# OLD: Setup the model with pretrained weights and send it to the target device (this was prior to torchvision v0.13)
# model = torchvision.models.efficientnet_b0(pretrained=True).to(device) # OLD method (with pretrained=True)

# NEW: Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

#model # uncomment to output (it's very long)`}
        />

        <div className="note">
          <strong>Note:</strong>
          <p>
            In previous versions of <code>torchvision</code>, you would create a
            pretrained model with code like:
          </p>

          <pre>
            <code>
              model =
              torchvision.models.efficientnet_b0(pretrained=True).to(device)
            </code>
          </pre>

          <p>
            However, running this code using <code>torchvision v0.13+</code>{" "}
            will result in errors such as the following:
          </p>

          <pre>
            <code>
              UserWarning: The parameter 'pretrained' is deprecated since 0.13
              and will be removed in 0.15, please use 'weights' instead.
            </code>
          </pre>

          <p>And...</p>

          <pre>
            <code>
              UserWarning: Arguments other than a weight enum or None for
              weights are deprecated since 0.13 and will be removed in 0.15. The
              current behavior is equivalent to passing
              weights=EfficientNet_B0_Weights.IMAGENET1K_V1. You can also use
              weights=EfficientNet_B0_Weights.DEFAULT to get the most up-to-date
              weights.
            </code>
          </pre>
        </div>

        <p>If we print the model, we get something similar to the following:</p>

        <img src={prin} className="centered-image" />

        <p>Lots and lots and lots of layers.</p>

        <p>
          This is one of the benefits of transfer learning: taking an existing
          model, crafted by some of the best engineers in the world, and
          applying it to your own problem.
        </p>

        <p>
          Our <code>efficientnet_b0</code> comes in three main parts:
        </p>

        <ul>
          <li>
            <strong>features</strong> - A collection of convolutional layers and
            other various activation layers to learn a base representation of
            vision data (this base representation/collection of layers is often
            referred to as features or feature extractor, "the base layers of
            the model learn the different features of images").
          </li>
          <li>
            <strong>avgpool</strong> - Takes the average of the output of the
            features layer(s) and turns it into a feature vector.
          </li>
          <li>
            <strong>classifier</strong> - Turns the feature vector into a vector
            with the same dimensionality as the number of required output
            classes (since <code>efficientnet_b0</code> is pretrained on
            ImageNet and because ImageNet has 1000 classes,{" "}
            <code>out_features=1000</code> is the default).
          </li>
        </ul>

        <h2>3.3 Getting a summary of our model with torchinfo.summary()</h2>

        <p>
          To learn more about our model, let's use <code>torchinfo</code>'s{" "}
          <code>summary()</code> method.
        </p>

        <p>To do so, we'll pass in:</p>

        <ul>
          <li>
            <strong>model</strong> - the model we'd like to get a summary of.
          </li>
          <li>
            <strong>input_size</strong> - the shape of the data we'd like to
            pass to our model. For the case of <code>efficientnet_b0</code>, the
            input size is <code>(batch_size, 3, 224, 224)</code>, though other
            variants of <code>efficientnet_bX</code> have different input sizes.
          </li>
        </ul>

        <p>
          <strong>Note:</strong> Many modern models can handle input images of
          varying sizes thanks to <code>torch.nn.AdaptiveAvgPool2d()</code>,
          this layer adaptively adjusts the <code>output_size</code> of a given
          input as required. You can try this out by passing different size
          input images to <code>summary()</code> or your models.
        </p>

        <ul>
          <li>
            <strong>col_names</strong> - the various information columns we'd
            like to see about our model.
          </li>
          <li>
            <strong>col_width</strong> - how wide the columns should be for the
            summary.
          </li>
          <li>
            <strong>row_settings</strong> - what features to show in a row.
          </li>
        </ul>

        <CodeIOBlock
          inputCode={`# Print a summary using torchinfo (uncomment for actual output)
summary(model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
) `}
          outputCode={`============================================================================================================================================
Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
============================================================================================================================================
EfficientNet (EfficientNet)                                  [32, 3, 224, 224]    [32, 1000]           --                   True
├─Sequential (features)                                      [32, 3, 224, 224]    [32, 1280, 7, 7]     --                   True
│    └─Conv2dNormActivation (0)                              [32, 3, 224, 224]    [32, 32, 112, 112]   --                   True
│    │    └─Conv2d (0)                                       [32, 3, 224, 224]    [32, 32, 112, 112]   864                  True
│    │    └─BatchNorm2d (1)                                  [32, 32, 112, 112]   [32, 32, 112, 112]   64                   True
│    │    └─SiLU (2)                                         [32, 32, 112, 112]   [32, 32, 112, 112]   --                   --
│    └─Sequential (1)                                        [32, 32, 112, 112]   [32, 16, 112, 112]   --                   True
│    │    └─MBConv (0)                                       [32, 32, 112, 112]   [32, 16, 112, 112]   1,448                True
│    └─Sequential (2)                                        [32, 16, 112, 112]   [32, 24, 56, 56]     --                   True
│    │    └─MBConv (0)                                       [32, 16, 112, 112]   [32, 24, 56, 56]     6,004                True
│    │    └─MBConv (1)                                       [32, 24, 56, 56]     [32, 24, 56, 56]     10,710               True
│    └─Sequential (3)                                        [32, 24, 56, 56]     [32, 40, 28, 28]     --                   True
│    │    └─MBConv (0)                                       [32, 24, 56, 56]     [32, 40, 28, 28]     15,350               True
│    │    └─MBConv (1)                                       [32, 40, 28, 28]     [32, 40, 28, 28]     31,290               True
│    └─Sequential (4)                                        [32, 40, 28, 28]     [32, 80, 14, 14]     --                   True
│    │    └─MBConv (0)                                       [32, 40, 28, 28]     [32, 80, 14, 14]     37,130               True
│    │    └─MBConv (1)                                       [32, 80, 14, 14]     [32, 80, 14, 14]     102,900              True
│    │    └─MBConv (2)                                       [32, 80, 14, 14]     [32, 80, 14, 14]     102,900              True
│    └─Sequential (5)                                        [32, 80, 14, 14]     [32, 112, 14, 14]    --                   True
│    │    └─MBConv (0)                                       [32, 80, 14, 14]     [32, 112, 14, 14]    126,004              True
│    │    └─MBConv (1)                                       [32, 112, 14, 14]    [32, 112, 14, 14]    208,572              True
│    │    └─MBConv (2)                                       [32, 112, 14, 14]    [32, 112, 14, 14]    208,572              True
│    └─Sequential (6)                                        [32, 112, 14, 14]    [32, 192, 7, 7]      --                   True
│    │    └─MBConv (0)                                       [32, 112, 14, 14]    [32, 192, 7, 7]      262,492              True
│    │    └─MBConv (1)                                       [32, 192, 7, 7]      [32, 192, 7, 7]      587,952              True
│    │    └─MBConv (2)                                       [32, 192, 7, 7]      [32, 192, 7, 7]      587,952              True
│    │    └─MBConv (3)                                       [32, 192, 7, 7]      [32, 192, 7, 7]      587,952              True
│    └─Sequential (7)                                        [32, 192, 7, 7]      [32, 320, 7, 7]      --                   True
│    │    └─MBConv (0)                                       [32, 192, 7, 7]      [32, 320, 7, 7]      717,232              True
│    └─Conv2dNormActivation (8)                              [32, 320, 7, 7]      [32, 1280, 7, 7]     --                   True
│    │    └─Conv2d (0)                                       [32, 320, 7, 7]      [32, 1280, 7, 7]     409,600              True
│    │    └─BatchNorm2d (1)                                  [32, 1280, 7, 7]     [32, 1280, 7, 7]     2,560                True
│    │    └─SiLU (2)                                         [32, 1280, 7, 7]     [32, 1280, 7, 7]     --                   --
├─AdaptiveAvgPool2d (avgpool)                                [32, 1280, 7, 7]     [32, 1280, 1, 1]     --                   --
├─Sequential (classifier)                                    [32, 1280]           [32, 1000]           --                   True
│    └─Dropout (0)                                           [32, 1280]           [32, 1280]           --                   --
│    └─Linear (1)                                            [32, 1280]           [32, 1000]           1,281,000            True
============================================================================================================================================
Total params: 5,288,548
Trainable params: 5,288,548
Non-trainable params: 0
Total mult-adds (G): 12.35
============================================================================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 3452.35
Params size (MB): 21.15
Estimated Total Size (MB): 3492.77
============================================================================================================================================`}
        />

        <img src={layer} className="centered-image" />

        <p>Woah!</p>

        <p>Now that's a big model!</p>

        <p>
          From the output of the summary, we can see all of the various input
          and output shape changes as our image data goes through the model.
        </p>

        <p>
          And there are a whole bunch more total parameters (pretrained weights)
          to recognize different patterns in our data.
        </p>

        <p>
          For reference, our model from previous sections,{" "}
          <strong>TinyVGG</strong>, had 8,083 parameters vs. 5,288,548
          parameters for <strong>efficientnet_b0</strong>, an increase of ~654x!
        </p>

        <p>What do you think, will this mean better performance?</p>

        <h2>
          3.4 Freezing the base model and changing the output layer to suit our
          needs
        </h2>

        <p>
          The process of transfer learning typically involves: freezing some
          base layers of a pretrained model (usually the features section) and
          then adjusting the output layers (also known as the head or classifier
          layers) to suit your needs.
        </p>

        <img src={head} className="centered-image" />

        <p>
          You can customize the outputs of a pretrained model by changing the
          output layer(s) to fit your problem. The original{" "}
          <code>torchvision.models.efficientnet_b0()</code> comes with{" "}
          <code>out_features=1000</code> because there are 1000 classes in
          ImageNet, the dataset it was trained on. However, for our problem of
          classifying images of pizza, steak, and sushi, we only need{" "}
          <code>out_features=3</code>.
        </p>

        <p>
          Let's freeze all of the layers/parameters in the features section of
          our <code>efficientnet_b0</code> model.
        </p>

        <div className="note">
          <strong>Note:</strong> To freeze layers means to keep them how they
          are during training. For example, if your model has pretrained layers,
          to freeze them would be to say, "don't change any of the patterns in
          these layers during training, keep them how they are." In essence,
          we'd like to keep the pretrained weights/patterns our model has
          learned from ImageNet as a backbone and then only change the output
          layers.
        </div>

        <p>
          We can freeze all of the layers/parameters in the features section by
          setting the attribute <code>requires_grad=False</code>.
        </p>

        <p>
          For parameters with <code>requires_grad=False</code>, PyTorch doesn't
          track gradient updates, meaning these parameters won't be changed by
          our optimizer during training.
        </p>

        <p>
          In essence, a parameter with <code>requires_grad=False</code> is
          considered "untrainable" or "frozen" in place.
        </p>

        <CodeIOBlock
          inputCode={`# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False`}
        />

        <p>Feature extractor layers frozen!</p>

        <p>
          Let's now adjust the output layer, or the classifier portion of our
          pretrained model, to suit our needs.
        </p>

        <p>
          Currently, our pretrained model has <code>out_features=1000</code>{" "}
          because there are 1000 classes in ImageNet.
        </p>

        <p>However, we only have three classes: pizza, steak, and sushi.</p>

        <p>
          We can change the classifier portion of our model by creating a new
          series of layers.
        </p>

        <p>The current classifier consists of:</p>

        <pre>
          <code>
            (classifier): Sequential( (0): Dropout(p=0.2, inplace=True) (1):
            Linear(in_features=1280, out_features=1000, bias=True)
          </code>
        </pre>

        <p>
          We'll keep the Dropout layer the same using torch.nn.Dropout(p=0.2,
          inplace=True).
        </p>

        <div className="note">
          <strong>Note:</strong>Dropout layers randomly remove connections
          between two neural network layers with a probability of p. For
          example, if p=0.2, 20% of connections between neural network layers
          will be removed at random each pass. This practice is meant to help
          regularize (prevent overfitting) a model by making sure the
          connections that remain learn features to compensate for the removal
          of the other connections (hopefully these remaining features are more
          general).
        </div>

        <p>
          We’ll keep <code>in_features=1280</code> for our Linear output layer,
          but we’ll change the <code>out_features</code> value to the length of
          our <code>class_names</code> (
          <code>len(['pizza', 'steak', 'sushi']) = 3</code>).
        </p>

        <p>
          Our new classifier layer should be on the same device as our model.
        </p>

        <CodeIOBlock
          inputCode={`# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)`}
        />

        <p>Nice!</p>

        <p>
          Output layer updated, let’s get another summary of our model and see
          what’s changed.
        </p>

        <CodeIOBlock
          inputCode={`# # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
summary(model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)`}
          outputCode={`============================================================================================================================================
Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
============================================================================================================================================
EfficientNet (EfficientNet)                                  [32, 3, 224, 224]    [32, 3]              --                   Partial
├─Sequential (features)                                      [32, 3, 224, 224]    [32, 1280, 7, 7]     --                   False
│    └─Conv2dNormActivation (0)                              [32, 3, 224, 224]    [32, 32, 112, 112]   --                   False
│    │    └─Conv2d (0)                                       [32, 3, 224, 224]    [32, 32, 112, 112]   (864)                False
│    │    └─BatchNorm2d (1)                                  [32, 32, 112, 112]   [32, 32, 112, 112]   (64)                 False
│    │    └─SiLU (2)                                         [32, 32, 112, 112]   [32, 32, 112, 112]   --                   --
│    └─Sequential (1)                                        [32, 32, 112, 112]   [32, 16, 112, 112]   --                   False
│    │    └─MBConv (0)                                       [32, 32, 112, 112]   [32, 16, 112, 112]   (1,448)              False
│    └─Sequential (2)                                        [32, 16, 112, 112]   [32, 24, 56, 56]     --                   False
│    │    └─MBConv (0)                                       [32, 16, 112, 112]   [32, 24, 56, 56]     (6,004)              False
│    │    └─MBConv (1)                                       [32, 24, 56, 56]     [32, 24, 56, 56]     (10,710)             False
│    └─Sequential (3)                                        [32, 24, 56, 56]     [32, 40, 28, 28]     --                   False
│    │    └─MBConv (0)                                       [32, 24, 56, 56]     [32, 40, 28, 28]     (15,350)             False
│    │    └─MBConv (1)                                       [32, 40, 28, 28]     [32, 40, 28, 28]     (31,290)             False
│    └─Sequential (4)                                        [32, 40, 28, 28]     [32, 80, 14, 14]     --                   False
│    │    └─MBConv (0)                                       [32, 40, 28, 28]     [32, 80, 14, 14]     (37,130)             False
│    │    └─MBConv (1)                                       [32, 80, 14, 14]     [32, 80, 14, 14]     (102,900)            False
│    │    └─MBConv (2)                                       [32, 80, 14, 14]     [32, 80, 14, 14]     (102,900)            False
│    └─Sequential (5)                                        [32, 80, 14, 14]     [32, 112, 14, 14]    --                   False
│    │    └─MBConv (0)                                       [32, 80, 14, 14]     [32, 112, 14, 14]    (126,004)            False
│    │    └─MBConv (1)                                       [32, 112, 14, 14]    [32, 112, 14, 14]    (208,572)            False
│    │    └─MBConv (2)                                       [32, 112, 14, 14]    [32, 112, 14, 14]    (208,572)            False
│    └─Sequential (6)                                        [32, 112, 14, 14]    [32, 192, 7, 7]      --                   False
│    │    └─MBConv (0)                                       [32, 112, 14, 14]    [32, 192, 7, 7]      (262,492)            False
│    │    └─MBConv (1)                                       [32, 192, 7, 7]      [32, 192, 7, 7]      (587,952)            False
│    │    └─MBConv (2)                                       [32, 192, 7, 7]      [32, 192, 7, 7]      (587,952)            False
│    │    └─MBConv (3)                                       [32, 192, 7, 7]      [32, 192, 7, 7]      (587,952)            False
│    └─Sequential (7)                                        [32, 192, 7, 7]      [32, 320, 7, 7]      --                   False
│    │    └─MBConv (0)                                       [32, 192, 7, 7]      [32, 320, 7, 7]      (717,232)            False
│    └─Conv2dNormActivation (8)                              [32, 320, 7, 7]      [32, 1280, 7, 7]     --                   False
│    │    └─Conv2d (0)                                       [32, 320, 7, 7]      [32, 1280, 7, 7]     (409,600)            False
│    │    └─BatchNorm2d (1)                                  [32, 1280, 7, 7]     [32, 1280, 7, 7]     (2,560)              False
│    │    └─SiLU (2)                                         [32, 1280, 7, 7]     [32, 1280, 7, 7]     --                   --
├─AdaptiveAvgPool2d (avgpool)                                [32, 1280, 7, 7]     [32, 1280, 1, 1]     --                   --
├─Sequential (classifier)                                    [32, 1280]           [32, 3]              --                   True
│    └─Dropout (0)                                           [32, 1280]           [32, 1280]           --                   --
│    └─Linear (1)                                            [32, 1280]           [32, 3]              3,843                True
============================================================================================================================================
Total params: 4,011,391
Trainable params: 3,843
Non-trainable params: 4,007,548
Total mult-adds (G): 12.31
============================================================================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 3452.09
Params size (MB): 16.05
Estimated Total Size (MB): 3487.41
============================================================================================================================================`}
        />

        <img src={info} className="centered-image" />

        <p>Ho, ho! There's a fair few changes here!</p>

        <p>Let's go through them:</p>

        <ul>
          <li>
            <strong>Trainable column</strong> - You'll see that many of the base
            layers (the ones in the features portion) have their Trainable value
            as False. This is because we set their attribute{" "}
            <code>requires_grad=False</code>. Unless we change this, these
            layers won't be updated during future training.
          </li>
          <li>
            <strong>Output shape of classifier</strong> - The classifier portion
            of the model now has an Output Shape value of <code>[32, 3]</code>{" "}
            instead of <code>[32, 1000]</code>. Its Trainable value is also
            True. This means its parameters will be updated during training. In
            essence, we're using the features portion to feed our classifier
            portion a base representation of an image, and then our classifier
            layer is going to learn how to align that base representation with
            our problem.
          </li>
          <li>
            <strong>Less trainable parameters</strong> - Previously there were
            5,288,548 trainable parameters. But since we froze many of the
            layers of the model and only left the classifier as trainable,
            there's now only 3,843 trainable parameters (even less than our
            TinyVGG model). Though there's also 4,007,548 non-trainable
            parameters, these will create a base representation of our input
            images to feed into our classifier layer.
          </li>
        </ul>

        <div className="note">
          <strong>Note:</strong> The more trainable parameters a model has, the
          more compute power/longer it takes to train. Freezing the base layers
          of our model and leaving it with less trainable parameters means our
          model should train quite quickly. This is one huge benefit of transfer
          learning, taking the already learned parameters of a model trained on
          a problem similar to yours and only tweaking the outputs slightly to
          suit your problem.
        </div>

        <h2>4. Train Model</h2>
        <p>
          Now that we've got a pretrained model that's semi-frozen and has a
          customised classifier, how about we see transfer learning in action?
        </p>

        <p>To begin training, let's create a loss function and an optimizer.</p>

        <p>
          Since we're still working with multi-class classification, we'll use{" "}
          <code>nn.CrossEntropyLoss()</code> for the loss function.
        </p>

        <p>
          And we'll stick with <code>torch.optim.Adam()</code> as our optimizer
          with <code>lr=0.001</code>.
        </p>

        <CodeIOBlock
          inputCode={`# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`}
        />

        <p>Wonderful!</p>

        <p>
          To train our model, we can use the <code>train()</code> function we
          defined in the <strong>05. PyTorch Going Modular</strong> section 04.
        </p>

        <p>
          The <code>train()</code> function is in the <code>engine.py</code>{" "}
          script inside the <code>going_modular</code> directory.
        </p>

        <p>Let's see how long it takes to train our model for 5 epochs.</p>

        <p>
          <strong>Note:</strong> We're only going to be training the classifier
          parameters here, as all of the other parameters in our model have been
          frozen.
        </p>

        <CodeIOBlock
          inputCode={`# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")`}
          outputCode={`
  0%|          | 0/5 [00:00<?, ?it/s]
Epoch: 1 | train_loss: 1.0924 | train_acc: 0.3984 | test_loss: 0.9133 | test_acc: 0.5398
Epoch: 2 | train_loss: 0.8717 | train_acc: 0.7773 | test_loss: 0.7912 | test_acc: 0.8153
Epoch: 3 | train_loss: 0.7648 | train_acc: 0.7930 | test_loss: 0.7463 | test_acc: 0.8561
Epoch: 4 | train_loss: 0.7108 | train_acc: 0.7539 | test_loss: 0.6372 | test_acc: 0.8655
Epoch: 5 | train_loss: 0.6254 | train_acc: 0.7852 | test_loss: 0.6260 | test_acc: 0.8561
[INFO] Total training time: 8.977 seconds`}
        />

        <p>Wow!</p>

        <p>
          Our model trained quite fast (~5 seconds on my local machine with a
          NVIDIA TITAN RTX GPU/about 15 seconds on Google Colab with a NVIDIA
          P100 GPU).
        </p>

        <p>
          And it looks like it smashed our previous model results out of the
          park!
        </p>

        <p>
          With an <code>efficientnet_b0</code> backbone, our model achieves
          almost 85%+ accuracy on the test dataset, almost double what we were
          able to achieve with <code>TinyVGG</code>.
        </p>

        <p>Not bad for a model we downloaded with a few lines of code.</p>

        <h2>5. Evaluate model by plotting loss curves</h2>

        <p>Our model looks like it's performing pretty well.</p>

        <p>
          Let's plot its loss curves to see what the training looks like over
          time.
        </p>

        <p>
          We can plot the loss curves using the function{" "}
          <code>plot_loss_curves()</code> we created in 04. PyTorch Custom
          Datasets section 7.8.
        </p>

        <p>
          The function is stored in the <code>helper_functions.py</code> script,
          so we'll try to import it and download the script if we don't have it.
        </p>

        <CodeIOBlock
          inputCode={`# Get the plot_loss_curves() function from helper_functions.py, download the file if we don't have it
try:
    from helper_functions import plot_loss_curves
except:
    print("[INFO] Couldn't find helper_functions.py, downloading...")
    with open("helper_functions.py", "wb") as f:
        import requests
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
        f.write(request.content)
    from helper_functions import plot_loss_curves

# Plot the loss curves of our model
plot_loss_curves(results)`}
        />

        <img src={graph} className="centered-image" />
        <p>Those are some excellent-looking loss curves!</p>

        <p>
          It looks like the loss for both datasets (train and test) is heading
          in the right direction.
        </p>

        <p>The same with the accuracy values, trending upwards.</p>

        <p>
          That goes to show the power of transfer learning. Using a pretrained
          model often leads to pretty good results with a small amount of data
          in less time.
        </p>

        <p>
          I wonder what would happen if you tried to train the model for longer?
          Or if we added more data?
        </p>

        <h2>6. Make predictions on images from the test set</h2>
        <p>
          It looks like our model performs well quantitatively, but how about
          qualitatively?
        </p>

        <p>
          Let's find out by making some predictions with our model on images
          from the test set (these aren't seen during training) and plotting
          them.
        </p>

        <p>
          <strong>Visualize, visualize, visualize!</strong>
        </p>

        <p>
          One thing we'll have to remember is that for our model to make
          predictions on an image, the image has to be in the same format as the
          images our model was trained on.
        </p>

        <p>This means we'll need to make sure our images have:</p>

        <ul>
          <li>
            <strong>Same shape</strong> - If our images are different shapes
            from what our model was trained on, we'll get shape errors.
          </li>
          <li>
            <strong>Same datatype</strong> - If our images are a different
            datatype (e.g. <code>torch.int8</code> vs.{" "}
            <code>torch.float32</code>), we'll get datatype errors.
          </li>
          <li>
            <strong>Same device</strong> - If our images are on a different
            device from our model, we'll get device errors.
          </li>
          <li>
            <strong>Same transformations</strong> - If our model is trained on
            images that have been transformed in a certain way (e.g. normalized
            with a specific mean and standard deviation) and we try and make
            predictions on images transformed in a different way, these
            predictions may be off.
          </li>
        </ul>

        <div className="note">
          <strong>Note:</strong> These requirements go for all kinds of data if
          you're trying to make predictions with a trained model. Data you'd
          like to predict on should be in the same format as your model was
          trained on.
        </div>

        <p>
          To do all of this, we'll create a function{" "}
          <code>pred_and_plot_image()</code> to:
        </p>

        <ul>
          <li>
            Take in a trained model, a list of class names, a filepath to a
            target image, an image size, a transform, and a target device.
          </li>
          <li>
            Open an image with <code>PIL.Image.open()</code>.
          </li>
          <li>
            Create a transform for the image (this will default to the{" "}
            <code>manual_transforms</code> we created above or it could use a
            transform generated from <code>weights.transforms()</code>).
          </li>
          <li>Make sure the model is on the target device.</li>
          <li>
            Turn on model eval mode with <code>model.eval()</code> (this turns
            off layers like <code>nn.Dropout()</code>, so they aren't used for
            inference) and the inference mode context manager.
          </li>
          <li>
            Transform the target image with the transform made in step 3 and add
            an extra batch dimension with <code>torch.unsqueeze(dim=0)</code> so
            our input image has shape [batch_size, color_channels, height,
            width].
          </li>
          <li>
            Make a prediction on the image by passing it to the model ensuring
            it's on the target device.
          </li>
          <li>
            Convert the model's output logits to prediction probabilities with{" "}
            <code>torch.softmax()</code>.
          </li>
          <li>
            Convert model's prediction probabilities to prediction labels with{" "}
            <code>torch.argmax()</code>.
          </li>
          <li>
            Plot the image with <code>matplotlib</code> and set the title to the
            prediction label from step 9 and prediction probability from step 8.
          </li>
        </ul>

        <div className="note">
          <strong>Note:</strong>This is a similar function to 04. PyTorch Custom
          Datasets section 11.3's pred_and_plot_image() with a few tweaked
          steps.
        </div>

        <CodeIOBlock
          inputCode={`from typing import List, Tuple

from PIL import Image

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);`}
        />

        <p>What a good-looking function!</p>

        <p>
          Let's test it out by making predictions on a few random images from
          the test set.
        </p>

        <p>
          We can get a list of all the test image paths using{" "}
          <code>list(Path(test_dir).glob("*/*.jpg"))</code>, the stars in the{" "}
          <code>glob()</code> method say "any file matching this pattern", in
          other words, any file ending in <code>.jpg</code> (all of our images).
        </p>

        <p>
          And then we can randomly sample a number of these using Python's{" "}
          <code>random.sample(population, k)</code> where{" "}
          <code>population</code> is the sequence to sample and <code>k</code>{" "}
          is the number of samples to retrieve.
        </p>

        <CodeIOBlock
          inputCode={`# Get a random list of image paths from test set
import random
num_images_to_plot = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                       k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

# Make predictions on and plot the images
for image_path in test_image_path_sample:
    pred_and_plot_image(model=model, 
                        image_path=image_path,
                        class_names=class_names,
                        # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                        image_size=(224, 224))`}
        />

        <img src={sushi} className="centered-image" />
        <p>Woohoo!</p>

        <p>
          Those predictions look far better than the ones our TinyVGG model was
          previously making.
        </p>

        <h2>6.1 Making predictions on a custom image</h2>
        <p>
          It looks like our model does well qualitatively on data from the test
          set.
        </p>

        <p>But how about on our own custom image?</p>

        <p>That's where the real fun of machine learning is!</p>

        <p>
          Predicting on your own custom data, outside of any training or test
          set.
        </p>

        <p>
          To test our model on a custom image, let's import the old faithful
          pizza-dad.jpeg image (an image of my dad eating pizza).
        </p>

        <p>
          We'll then pass it to the <code>pred_and_plot_image()</code> function
          we created above and see what happens.
        </p>

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
    print(f"{custom_image_path} already exists, skipping download.")

# Predict on custom image
pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=class_names)`}
          outputCode={`data/04-pizza-dad.jpeg already exists, skipping download.`}
        />
        <img src={happy} className="centered-image" />

        <p>Two thumbs up!</p>

        <p>Looks like our model got it right again!</p>

        <p>
          But this time the prediction probability is higher than the one from
          TinyVGG (0.373) in 04. PyTorch Custom Datasets section 11.3.
        </p>

        <p>
          This indicates our <code>efficientnet_b0</code> model is more
          confident in its prediction whereas our <code>TinyVGG</code> model was
          par with just guessing.
        </p>

        <h2>The End</h2>
      </section>
    </div>
  );
};

export default transferLearning;
