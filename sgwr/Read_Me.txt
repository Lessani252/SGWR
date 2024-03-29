In the current format, copy and past 'sgwr' folder in your python environment, if you're using anaconda env, find your env that you work on it and then 'Lib' next past the folder in this folder named 'site-package'. Then you should be able to run the code without any issues using your anaconda prompt. Make sure your env is activated where you work on. Example: (geop-env) C:\Users\unknown>python -m sgwr run -np n -data (directory to your data). In this command 'n' stands for the number of processor. 

Data format should be like this in csv file: 
xutm   yutm   x-coord   y-coord   (dependent variable) (indipendent variables x1, x2, x3, ..............kn) 

