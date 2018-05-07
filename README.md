# mini-frame
[![Build Status](https://travis-ci.org/jdavidrcamacho/mini-frame.svg?branch=master)](https://travis-ci.org/jdavidrcamacho/mini-frame)

A implementation of the Gaussian process frameworks described on [Rajpaul et al. (2015)](https://academic.oup.com/mnras/article/452/3/2269/1079217) and [Jones et al. (2017)](https://arxiv.org/abs/1711.01318). 

-------------------------

To install just type in the terminal

     python setup.py develop
  
Then you can import it just like a normal python package with
     
     import miniframe
     
-------------------------

On miniframe you can find two main classes to work with the frameworks.

BIGgp allows you to work with Rajpaul et al. (2015) framework, SMALLgp allows you to work with Jones et al. (2017) framework. Check the documentation of the classes to learn a bit more, in the future a "guide" will be written to show how the package works.

     
-------------------------

###### Needed packages
[numpy](http://www.numpy.org/)
[scipy](https://www.scipy.org/)
[emcee](http://dfm.io/emcee/current/)
[matplotlib](https://matplotlib.org/)


<img align="center" width="150" height="170" src="https://musingsonmath.files.wordpress.com/2011/04/gauss_portrait.jpg">
