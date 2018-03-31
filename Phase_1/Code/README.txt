Getting Started
-------------------------------------------------------------------------------

1. Installing Anaconda
Download Anaconda from the following url
	 https://www.anaconda.com/download/

2. Creating conda environment
Command line method:
Open terminal on mac or command prompt on windows and type the following

     conda create -n env_name python=3.6 anaconda

Install pandas and numpy packages

	conda install --name env_name pandas
	conda install --name env_name numpy

GUI method:
Open the anaconda interface, on the left pane go to environments there select the add environment option. Give the environment a name.
You can see the packages installed in the given environment. Add pandas and numpy by selecting the packages option on the left side of anaconda's interface.

--------------------------------------------------------------------------------
Without anaconda on OS X

Type on terminal:

1. Install brew

	ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

     export PATH=/usr/local/bin:/usr/local/sbin:$PATH

2. Install python
   	   brew install python3

3. install pandas
	   pip install pandas
4. install numpy
	   pip install numpy

--------------------------------------------------------------------------------
Running the programs

1. Storing the datasets
Datasets can be stored anywhere as long as the path changes are made to the program files.

2. Running the programs
Make appropriate changes to the dataset paths in the program.

3. Python path needs to be updated in the shebang, so that bash will recognize it as executable python file.

Open the conda terminal
Activate conda environment

source activate env_name
For task1:
    cd /path/to/pyfile
    
    ./print_actor_vector.py 1698048 TF
    
    ./print_actor_vector.py 1698048 TF-IDF

For task2:
    cd /path/to/pyfile
    
    ./print_genre_vector.py Comedy TF    

    ./print_genre_vector.py Comedy TF-IDF

For task3:
    cd /path/to/pyfile
    
    ./print_ user_vector.py 146 TF
    
    ./print_ user_vector.py 146 TF-IDF

For task4:
    cd /path/to/pyfile
    
    ./differentiatie_genre.py Comedy Drama TF-IDF-DIFF
    
    ./differentiate_genre.py Comedy Drama P-DIFF1
    
    ./differentiate_genre.py Comedy Drama P-DIFF2