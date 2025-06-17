# Traning an Eye for Fashion

This project reproduces the configuration from a PyTorch tutorial, where a simple ML model is created and trained to recognize categories of clothing from the FashionMNIST dataset.

https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

This was chosen as it is a relatively simple project which can be relativley self contained. Still, it is far more complex than the previous examples.

See also: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

## Custom Code

While Forgather is good at assembling objects, the language is not practical for defining logic. For this, we have defined a custom "trainer" class in the project's 'src' directory and we will use Forgather to dynamically import this code, injecting all the required dependencies.

Unlike the previous projects, you will note that the "Modules" section not empty and has a link to the model definition.

## Project Structure

Like the previous example, this project makes use of template inheritance, where there is a common 'project.yaml' file from which all of the configuratioins are derived.

Unlike the previous project, the we make use of the templates library to help automate things. This project was actually more complex to setup than most, as there is not a pre-existing template for Torch-vision projects, so the project template had to fill in the details.


---

