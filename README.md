Building Machine Learning and Deep Learning Models on Google Cloud Platform
============================================================================

This repository contains the chapter code samples for the **upcoming** book *Building Machine Learning and Deep Learning Models on Google Cloud Platform* published by Apress.  This book is a compendium to assist the beginner who knows absolutely nothing about machine learning and deep learning to be able to learn how to build learning models and to do so by leveraging the computational resources of Google Cloud Platform.

![gcp-ml-dl-book](https://ekababisong.org/assets/books/BisongHiRes.jpg)



# Recommended Usage
The recommended way to work with the notebooks in this repository is through the Google Datablab cloud instance.

To do so,
* Create a Datalab instance using the `--no-create-repository` flag. This flag prevents the instance from setting up a Cloud Source repository, which is a private Git repositories hosted on Google Cloud Platform. Use the format:
`datalab create [INSTANCE_NAME] --no-create-repository`
* 
* 

+There are several ways to have this code in `DataLab`, but the best way is to ...
+
+* Create a new datalab instance with the `--no-create-repository` flag so that no Cloud Source repo is setup.
+    * e.g. `datalab --project [PROJECT_NAME] create [INSTANCE_NAME] --no-create-repository`
+    * this will create the instance, and once ready will open a new browser window that drops you inside the default image.
+* Click the ungit icon on the top right - this looks like the git branching icon.
+* Using the "address bar" within ungit, navigate to: `/content/`
+* Now just put the git url you want to clone in the "clone from" section, e.g. the url to this repo.
+* One will then be asked to authenticate, username and password.
+* Then just navigate to the cloned repo and work as normal.
+* N.B. For ungit to recognise the changes, one must save the notebook/file. From here, one can just use the ungit UI to add and commit changes.
+* When finished, using the ungit UI again, one can push these changes to Github.
+
+N.B. the `content` directory mentioned above is at `/mnt/disks/datalab-pd/content` on the instance.
+
+Once finished, there are two ways to stop
+
+1) One can [STOP](https://cloud.google.com/datalab/docs/reference/command-line/stop) the instance. This just stops the instance, but does not delete the disk. This means that one can later re[CONNECT](https://cloud.google.com/datalab/docs/reference/command-line/connect) to the instance, and it is in the state as when it was stopped. This means that the Github connection still exists and one can pull the latest changes in, and make changes. Stopping and connecting can be done from the command line and the console.
+
+2) One can [DELETE](https://cloud.google.com/datalab/docs/reference/command-line/delete) the instance. This stops the instance and deletes the disk. This means one **cannot** reconnect to the instance, and so will have to start the process again. 


First, you will need to install [git](https://git-scm.com/), if you don't have it already.

Next, clone this repository by opening a terminal and typing the following commands:

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/ageron/handson-ml.git
    $ cd handson-ml

If you want to go through chapter 16 on Reinforcement Learning, you will need to [install OpenAI gym](https://gym.openai.com/docs) and its dependencies for Atari simulations.

If you are familiar with Python and you know how to install Python libraries, go ahead and install the libraries listed in `requirements.txt` and jump to the [Starting Jupyter](#starting-jupyter) section. If you need detailed instructions, please read on.

## Python & Required Libraries
Of course, you obviously need Python. Python 2 is already preinstalled on most systems nowadays, and sometimes even Python 3. You can check which version(s) you have by typing the following commands:

    $ python --version   # for Python 2
    $ python3 --version  # for Python 3

Any Python 3 version should be fine, preferably ≥3.5. If you don't have Python 3, I recommend installing it (Python ≥2.6 should work, but it is deprecated so Python 3 is preferable). To do so, you have several options: on Windows or MacOSX, you can just download it from [python.org](https://www.python.org/downloads/). On MacOSX, you can alternatively use [MacPorts](https://www.macports.org/) or [Homebrew](https://brew.sh/). If you are using Python 3.6 on MacOSX, you need to run the following command to install the `certifi` package of certificates because Python 3.6 on MacOSX has no certificates to validate SSL connections (see this [StackOverflow question](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)):

    $ /Applications/Python\ 3.6/Install\ Certificates.command

On Linux, unless you know what you are doing, you should use your system's packaging system. For example, on Debian or Ubuntu, type:

    $ sudo apt-get update
    $ sudo apt-get install python3

Another option is to download and install [Anaconda](https://www.continuum.io/downloads). This is a package that includes both Python and many scientific libraries. You should prefer the Python 3 version.

If you choose to use Anaconda, read the next section, or else jump to the [Using pip](#using-pip) section.

## Using Anaconda
When using Anaconda, you can optionally create an isolated Python environment dedicated to this project. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially different libraries and library versions:

    $ conda create -n mlbook python=3.5 anaconda
    $ source activate mlbook

This creates a fresh Python 3.5 environment called `mlbook` (you can change the name if you want to), and it activates it. This environment contains all the scientific libraries that come with Anaconda. This includes all the libraries we will need (NumPy, Matplotlib, Pandas, Jupyter and a few others), except for TensorFlow, so let's install it:

    $ conda install -n mlbook -c conda-forge tensorflow=1.4.0

This installs TensorFlow 1.4.0 in the `mlbook` environment (fetching it from the `conda-forge` repository). If you chose not to create an `mlbook` environment, then just remove the `-n mlbook` option.

Next, you can optionally install Jupyter extensions. These are useful to have nice tables of contents in the notebooks, but they are not required.

    $ conda install -n mlbook -c conda-forge jupyter_contrib_nbextensions

You are all set! Next, jump to the [Starting Jupyter](#starting-jupyter) section.

## Using pip 
If you are not using Anaconda, you need to install several scientific Python libraries that are necessary for this project, in particular NumPy, Matplotlib, Pandas, Jupyter and TensorFlow (and a few others). For this, you can either use Python's integrated packaging system, pip, or you may prefer to use your system's own packaging system (if available, e.g. on Linux, or on MacOSX when using MacPorts or Homebrew). The advantage of using pip is that it is easy to create multiple isolated Python environments with different libraries and different library versions (e.g. one environment for each project). The advantage of using your system's packaging system is that there is less risk of having conflicts between your Python libraries and your system's other packages. Since I have many projects with different library requirements, I prefer to use pip with isolated environments.

These are the commands you need to type in a terminal if you want to use pip to install the required libraries. Note: in all the following commands, if you chose to use Python 2 rather than Python 3, you must replace `pip3` with `pip`, and `python3` with `python`.

First you need to make sure you have the latest version of pip installed:

    $ pip3 install --user --upgrade pip

The `--user` option will install the latest version of pip only for the current user. If you prefer to install it system wide (i.e. for all users), you must have administrator rights (e.g. use `sudo pip3` instead of `pip3` on Linux), and you should remove the `--user` option. The same is true of the command below that uses the `--user` option.

Next, you can optionally create an isolated environment. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially very different libraries, and different versions:

    $ pip3 install --user --upgrade virtualenv
    $ virtualenv -p `which python3` env

This creates a new directory called `env` in the current directory, containing an isolated Python environment based on Python 3. If you installed multiple versions of Python 3 on your system, you can replace `` `which python3` `` with the path to the Python executable you prefer to use.

Now you must activate this environment. You will need to run this command every time you want to use this environment.

    $ source ./env/bin/activate

Next, use pip to install the required python packages. If you are not using virtualenv, you should add the `--user` option (alternatively you could install the libraries system-wide, but this will probably require administrator rights, e.g. using `sudo pip3` instead of `pip3` on Linux).

    $ pip3 install --upgrade -r requirements.txt

Great! You're all set, you just need to start Jupyter now.

## Starting Jupyter
If you want to use the Jupyter extensions (optional, they are mainly useful to have nice tables of contents), you first need to install them:

    $ jupyter contrib nbextension install --user

Then you can activate an extension, such as the Table of Contents (2) extension:

    $ jupyter nbextension enable toc2/main

Okay! You can now start Jupyter, simply type:

    $ jupyter notebook

This should open up your browser, and you should see Jupyter's tree view, with the contents of the current directory. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree). Click on `index.ipynb` to get started!

Note: you can also visit [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions) to activate and configure Jupyter extensions.

Congrats! You are ready to learn Machine Learning, hands on!

# Contributors
I would like to thank everyone who contributed to this project, either by providing useful feedback, filing issues or submitting Pull Requests. Special thanks go to Steven Bunkley and Ziembla who created the `docker` directory.