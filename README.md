# Detecting abnormalities in clinical data with Generative Adversarial Networks

Some of the code for my bachelor's thesis.

## Dependencies
* [Python (Anaconda 3.6.5)](https://anaconda.org/) 
* [NumPy (1.14.3)](http://www.numpy.org/) 
* [TensorFlow (1.8.0)](https://www.tensorflow.org/) 
* [scikit-learn (0.19.2)](http://scikit-learn.org/stable/) 
* [wfdb (2.2.0)](https://github.com/MIT-LCP/wfdb-python) 


## Description
Detection of medical conditions or abnormalities in clinical
data has already been achieved by employing discriminative models, especially
neural networks. For example, neural networks have been trained as binary
classifiers to learn a decision boundary that discriminates between normal
and abnormal ECG recordings. The training process requires normal as well as
abnormal ECG recordings. However, public ECG databases generally consist of
many normal recordings and a few abnormal recordings. This circumstance served
as the main incentive behind my bachelor's thesis, train a generative adversarial
network with only normal data and examine whether it is capable of discriminating between
normal and abnormal data. <br/>
During my research phase, I conducted two separate experiments. In one experiment, I examined
whether generative adversarial networks are capable of detecting cardiac arrhythmia in ECG recordings.
For this, I trained a generative adversarial network with normal ECG recordings. Afterwards
I fed two separate test sets into the trained discriminator, one containing normal ECG
recordings and the other containing arrhythmic ECG recordings. The hypothesis was that,
because the discriminator was trained only on normal recordings, the loss would be larger
for the test set containing arrhythmic recordings. For the other experiment, I repeated the
procedure in order to examine whether generative adversarial networks, when given the blood gas
measurements of a patient, can decide whether or not that patient has respiratory failure.


## Authors

* **Matthias Wright**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details 
