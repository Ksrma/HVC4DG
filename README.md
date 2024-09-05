# The HVC4D-G Algorithm
An $O(n^{3/2}\log n)$ exact algorithm for computing 4-D hypervolume contributions of all the points in an $n$-point set

This is the implementation of our method proposed in:

Jingda Deng, Qingfu Zhang, Jianyong Sun, and Hui Li, "A Fast Exact Algorithm for Computing the Hypervolume Contributions in 4-D Space", IEEE Transactions on Evolutionary Computation, 28(4): 876-890, 2024, DOI: 10.1109/TEVC.2023.3271679.

# HVC4D-G/HVC4D-GS
HVC4DG.cpp: implementation of the basic HVC4D-G algorithm

HVC4DGS.cpp: implementation of an improved version of HVC4D-G by using the method described in Section VI-C

Compilation: g++ -O3 HVC4DG.cpp/HVC4DGS.cpp

Usage: HVC4DG/HVC4DGS #1 word1 word2 word3

where #1 is the number of points(n), word1 is the input file name, word2 is the reference point file name, and word3 is the output file name. Results will be outputed to screen if there is no word3. Input file should contains n lines, each of which contains 4 numbers separated by blanks. Currently, codes can work but have not been optimized. Some comments in the codes may not be up-to-date.

# Test Set
Test sets in the numerical experiments (spherical, cliff, hard instances, and random sets with dominated points)

# Supplementary Materials
For your information, the supplementary materials of the published paper are uploaded.

# Contact
Jingda Deng

School of Mathematics and Statistics

Xi'an Jiaotong University

E-mail: jddeng@xjtu.edu.cn
