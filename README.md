# SPIJN
## Description
Code for the Sparsity Promoting Iterative Joint NNLS  (SPIJN) algorithm as used for Multi-component Magnetic Resonance Fingerprinting using a joint sparsity constraint.
The algorithm is described in 

Nagtegaal, Martijn, Peter Koken, Thomas Amthor, and Mariya Doneva. “Fast Multi-Component Analysis Using a Joint Sparsity Constraint for MR Fingerprinting.” Magnetic Resonance in Medicine https://doi.org/10.1002/mrm.27947.

## Code provided
SPIJN.py provides the SPIJN function to perform a MC-MRF decomposition and the lsqnonneg fuction to perform a voxel-wise NNLS solve in a similar way.

Example_Simulations.py provides code similar to the simulations as shown in [1].
A comparison between the SPIJN and NNLS algorithm is made with a pre-computed dictionary and numerical phantom, which are saved in data.npz . 

# License Information

The algorithm is provided free of charge for use in research applications. It must not be used for diagnostic applications. The author takes no responsibility of any kind for the accuracy or integrity of the created data sets. No data created using this algorithm should be used to make diagnostic decisions. The source code is provided under the GNU General Public License (https://www.gnu.org/copyleft/gpl.html).

The algorithm, including all of its software components, comes without any warranties. Operation of the software is solely on the user’s own risk. The author takes no responsibility for damage or misfunction of any kind caused by the software.
# Contact information
(c) Martijn Nagtegaal, June 2019

Technical University of Delft, Faculty of Applied Sciences

m.a.nagtegaal@tudelft.nl

![ImPhys Logo](https://d1rkab7tlqy5f1.cloudfront.net/_processed_/6/1/csm_ImPhys-logo_met%20tekst_d076a5cd76.png "Logo Imaging Physics")
