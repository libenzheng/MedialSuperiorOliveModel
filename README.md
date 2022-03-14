
# Medial Superior Olive Model
A spiking neuronal network model of medial superior olive circuitry for analyzing spatial hearing perception under various axon myelination patterns.


[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/libenzheng/MedialSuperiorOliveModel/blob/main/LICENSE)
## Authors
- [Ben-Zheng Li](https://github.com/libenzheng) @ [Klug lab](https://www.kluglab.org/)

## Prerequisites


| Package            | Version     | 
| :----------------------- | :---------------- | 
| python | 3.8.12 |
|[brian2](https://briansimulator.org/)             |       	    2.4.2
|[cochlea](https://github.com/mrkrd/cochlea/)             |               2
|cython               |              0.29.25
|matplotlib            |             3.5.0
|numpy                  |            1.21.2
|pandas                  |           1.3.5
|[scikit-learn](https://scikit-learn.org/)             |          1.0.2
|scipy                     |         1.7.3
|[seaborn](https://seaborn.pydata.org/)                    |        0.11.2


## Usage and Example

- Edit and run [configuration.py](https://github.com/libenzheng/MedialSuperiorOliveModel/blob/main/configuration.py) to update simulation settings, including:
    - characteristic frequencies of auditory nerve fibers 
    - frequencies and types of sound stimuli (pure tones and owl calls)
    - number of replica and random permutations
    - parameters of neuron and synapse 
    - myelination properties

- To generate ITD tuning curve in the paper (e.g. Fig.3), run [example.ipynb](https://github.com/libenzheng/MedialSuperiorOliveModel/blob/main/example.ipynb) using more random permutations (n_seed = 20).


## Citation

Ben-Zheng Li, Sio Hang Pun, Mang I Vai, Tim Lei, Achim Klug (2022) Predicting the Influence of Axon Myelination on Sound Localization Precision Using a Spiking Neural Network Model of Auditory Brainstem. Front. Neurosci. 16:840983. doi: 10.3389/fnins.2022.840983





## License

This project is licensed under the MIT license ([LICENSE](https://github.com/libenzheng/MedialSuperiorOliveModel/blob/main/LICENSE)).
