# The HVC4DG Algorithm
An O(n^{3/2}logn) exact algorithm for 4-D hypervolume contributions

This is the implementation of our method proposed in:

Jingda Deng, Qingfu Zhang, Jianyong Sun, and Hui Li, "An O(n^{3/2}logn) Exact Algorithm for Computing Hypervolume Contributions in 4-D Space"

which was submitted to IEEE TEVC (under review)

# HVC4DG/HVC4DGS
HVC4DG: the original algorithm described in Section III-B

HVC4DGS: an improved version of HVC4DG by using the method described in Section III-C

Compilation: g++ -O3 HVC4DG/HVC4DGS.cpp

Usage: HVC4DG/HVC4DGS <number of points(n)> <input file> <reference point file> <outputfile(optional)>
  
  
  input file should contains n lines, each of which contains 4 numbers separated by blanks

Currently, codes can work but have not been optimized. Some comments may not be up-to-date.

# Test Set
Test sets in the numerical experiments (spherical, cliff and hard instances)
# Contact
Jingda Deng

School of Mathematics and Statistics

Xi'an Jiaotong Universigy

E-mail: jddeng@xjtu.edu.cn
