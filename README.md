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
network with only normal data. I ran two separate experiments.
At first I tried to detect respiratory failure on the basis of blood gas measurements.
I trained a generative adversarial network with blood gas measurements from healthy patients.
Afterwards, I fed two separate test sets into the trained discriminator. One containing
blood gas measurements of healthy patients and the other containing blood gas measurements of
patients suffering from respiratory failure. The hope was that because the discriminator was
trained only on healthy data, the loss would be larger for the test set containing the
examples of patients with respiratory failure. It did, but the difference was not significant
from a medical point of view. 
I repeated this approach for the other experiment, but I tried to detect cardiac arrhythmia in
ECG recordings instead. This time the difference between the two test sets was significant from
a medical perspective.


## Authors

* **Matthias Wright**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details 
