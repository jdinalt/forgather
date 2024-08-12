# Traning  for Fashion

This project reproduces the configuration from a PyTorch tutorial, where a simple ML model is created and trained to recognize categories of clothing from the FashionMNIST dataset.

https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

This was chosen as it is a relatively simple project which can be relativley self contained. Still, it is far more complex than the previous examples.

The model itself does not require any custom code. It's simply a stack of PyTorch layers, chained together with a nn.Sequential. If you would like to know more about the model itself, see:

https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

## Custom Code

While Forgather is good at assembling objects, the language is not practical for defining logic. For this, we have defined a custom "trainer" class in the project's 'src' directory and we will use Forgather to dynamically import this code, injecting all the required dependencies.

Unlike the previous projects, you will note that the "Modules" section not empty and has a link to the Trainer definition.

## Project Structure

Like the previous example, this project makes use of template inheritance, where there is a common 'project.yaml' file from which all of the configuratioins are derived.

The template provides the basic structure, with 'blocks' which may be substituted or extended by child templates. We use this functionaity in the configurations to override various components of the base configuraiton.

This is still a relatively simple project, as it does not reference any external template libraries. We will get to that in the next example.

## Code Generation

Note the output of the code generator. It has detected the inclusion of a dynamic import, thus it has automatically defined a function for importing dynamic modules.

Also note that it knows how to translate some of the rather clunky expreressions from the original YAML file, like calling a method, into relatively clean Python code.

---

