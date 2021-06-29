# The HVC4D-G Algorithm
An O(n^{3/2}logn) exact algorithm for computing 4-D hypervolume contributions of all the points in a point set

This is the implementation of our method proposed in:

Jingda Deng, Qingfu Zhang, Jianyong Sun, and Hui Li, "An O(n^{3/2}logn) Exact Algorithm for Computing Hypervolume Contributions in 4-D Space"

which was submitted to IEEE TEVC (under review)

# HVC4D-G/HVC4D-GS
HVC4DG.cpp: implementation of the basic HVC4D-G algorithm

HVC4DGS.cpp: implementation of an improved version of HVC4D-G by using the method described in Section IV-E

Compilation: g++ -O3 HVC4DG.cpp/HVC4DGS.cpp

Usage: HVC4DG/HVC4DGS #1 word1 word2 word3

where #1 is the number of points(n), word1 is the input file name, word2 is the reference point file name, and word3 is the output file name. Results will be outputed to screen if there is no word3. Input file should contains n lines, each of which contains 4 numbers separated by blanks. Currently, codes can work but have not been optimized. Some comments may not be up-to-date.

# Test Set
Test sets in the numerical experiments (spherical, cliff, hard instances, and random sets with dominated points)
# Contact
Jingda Deng

School of Mathematics and Statistics

Xi'an Jiaotong University

E-mail: jddeng@xjtu.edu.cn
