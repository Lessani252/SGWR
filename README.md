Note:
-----
In this model, for the geographically weighted matrix a combination of adaptive bisquare and adaptive Gaussian kernel functions are used. For bandwidth optimization, the adaptive bisquare kernel is used. Once the optimal bandwidth is found using adaptive bisquare, it is then used as the number of nearest neighbors (k) for alpha optimization and the final model fitting using gaussian kernel. 

Software:
---------
You can easily install the python packages using (pip install sgwr), and then both the serial and parallel versions of the libraries are installed. For GUI use, please download this setup (SGWR_Model_WINDOWS_0.3_setup.exe) from the above and just follow the installation steps. You should be able to run the GUI in your local computer without using any programming. 

<img width="762" height="526" alt="SGWR" src="https://github.com/user-attachments/assets/318f1c93-5db1-4e3a-b5f1-0c126dc9fdf9" />



More details about how to use the sgwr libraries and GUI are available in this repository: https://github.com/Lessani252/FastSGWR.

Multiscale SGWR:
---------
The multiscale version of this model (M-SGWR) can be found here along with its Github Repo: https://doi.org/10.48550/arXiv.2601.19888.

Description:
------------
In the current version, copy and past 'sgwr' folder in your python environment, if you're using anaconda env, find your env that you work on it and then 'Lib' next past the folder in this folder named 'site-package'. Then you should be able to run the code without any issues using your anaconda prompt. Make sure your env is activated where you work on. Example: (geop-env) C:\Users\unknown>python -m sgwr run -np n -data (directory to your data). In this command 'n' stands for the number of processor. 

Data format should be like this in csv file, as also can be seen in the provided datasets: 
------------------------------------------------------------------------------------------
x-coord   y-coord   (dependent variable) (indipendent variables x1, x2, x3, ..............kn). 
The base code is driven from FastGWR and you can find it via this link (https://github.com/Ziqi-Li/FastGWR). 

