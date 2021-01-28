/*-------------------------------------------------------------------

                  Copyright (c) 2021
            Jingda Deng <jddeng@xjtu.edu.cn>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

---------------------------------------------------------------------

This C++ program calculates the exact hypervolume contributions 
of a set of 4-dimensional points. This is the improved version of 
the HVC4D-G algorithm by using a new space partition strategy. 
Please refer to the following paper for a description of 
this HVC4D-GS algorithm: 

Jingda Deng, Qingfu Zhang, Jianyong Sun, and Hui Li, "An 
O(n^{3/2}logn) Exact Algorithm for Hypervolume Contributions 
in 4-D Space".

Compilation: g++ -O3 HVC4DGS.cpp
Usage: HVC4DGS <number of points> <input file> 
       <reference point file> <outputfile(optional)>
       
       Input file should contain n lines (n is the number of points).
       Each line contains four numbers separated by blanks.
       
Notice: (1) Codes for timing in the main function only work in Linux. 
        Our codes can work well in Windows platform (e.g., Visual 
        Studio) after removing or changing them.
        (2) These codes work for MINIMIZATION problem, but it can be
        easily revised for MAXIMIZATION problem by changing points 
        according to the reference point when reading data file.

Special thanks to Nicola Beume for providing source codes of 
the HOY algorithm. I have learned a lot from them. 

---------------------------------------------------------------------*/

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <sys/time.h>
#include <sys/resource.h>

using namespace std;
#define dimension 4

static int alter;
static int popsize;
static double dSqrtDataNumber;
static int *piles;
static int *treeProjection;
static int *boundaries;
static int *noBoundaries;
static double *contributions;
static double maxWeight;
static vector<double*> population;

template <typename Container>
struct compare_indirect_index_ascend
{
	const Container& container;
	const int dim;
	compare_indirect_index_ascend( const Container& container, const int dim ): container( container ), dim( dim ) { }
	bool operator () ( size_t lindex, size_t rindex ) const
	{
		return container[ lindex ][ dim ] < container[ rindex ][ dim ];
	}
};
template <typename Container>
struct compare_indirect_index_ascend2
{
	const Container& container;
	compare_indirect_index_ascend2( const Container& container ): container( container ) { }
	bool operator () ( size_t lindex, size_t rindex ) const
	{
		return container[ lindex ] < container[ rindex ];
	}
};
template <typename Container>
struct compare_indirect_index_descend
{
	const Container& container;
	const int dim;
	compare_indirect_index_descend( const Container& container, const int dim ): container( container ), dim( dim ) { }
	bool operator () ( size_t lindex, size_t rindex ) const
	{
		return container[ lindex ][ dim ] > container[ rindex ][ dim ];
	}
};

inline void Index_Ascend_Sort(vector<double*> x, int* beg, int n, int dim)
{	
	sort(beg, beg+n, compare_indirect_index_ascend <decltype(x)> ( x, dim ) );
}
inline void Index_Ascend_Sort(vector<int> x, int* beg, int n)
{	
	sort(beg, beg+n, compare_indirect_index_ascend2 <decltype(x)> ( x ) );
}
inline void Index_Descend_Sort(vector<double*> x, int* beg, int n, int dim)
{	
	sort(beg, beg+n, compare_indirect_index_descend <decltype(x)> ( x, dim ) );
}

inline bool Yildiz_cmp(double* a, double* b) { 
	return (a[dimension-1] > b[dimension-1]);
}

// basic BIT supporting single update and range query
class BIT {

public:
	double* BITArray;
	int setSize;
	BIT() {
	}
	~BIT() {
		delete [] BITArray;
	}
	inline void Add(int i, double v){
		for(; i<=setSize; i+=i&-i) {
			BITArray[i]+=v;
		}
	}
	inline double Sum(int i) {
		double r=0.;
		for(; i; i-=i&-i) {
			r+=BITArray[i];
		}
		return r;
	}
	inline double Sum(int i, int j) {
		if (i > j) {
			return 0.;
		} else {
			return Sum(j) - Sum(i-1);
		}
	}
};

// BIT supporting range update and single query by difference
class BIT2 {

public:
	double* BITArray;
	int setSize;
	BIT2() {
	}
	~BIT2() {
		delete [] BITArray;
	}
	inline void _Add(int i, double v){
		for(; i<=setSize; i+=i&-i) {
			BITArray[i]+=v;
		}
	}
	inline void Add(int i, double v) {
		_Add(i, v);
		_Add(i+1, -v);
	}
	inline void Add(int i, int j, double v) {
		_Add(i, v);
		_Add(j+1, -v);
	}
	inline double Sum(int i) {
		double r=0.;
		for(; i; i-=i&-i) {
			r+=BITArray[i];
		}
		return r;
	}
	// this function will be used in YildizeHC frequently
	inline double Clear(int i) {
		double r=Sum(i);
		Add(i, -r);
		return r;
	}
};

class YildizHC {	

public:
	BIT A[2], weightedA[2], secondA[2], secondWeightedA[2], weightedSubA[2];
	BIT2 dominated[2];
	double *C;
	double *projection[2];
	map<double, int> Gradient[2], secondGradient[2], SubGradient[2], WeightFinder[2], SubWeightFinder[2];
	double L[2], V[2], secondV[2];
	vector<bool> type;
	vector<double> weight, coordinates;
	int setSize;

	void init(int setsize) {
		setSize = setsize;
		V[0] = 0.;
		V[1] = 0.;
		secondV[0] = 0.;
		secondV[1] = 0.;
		C = new double[setSize+1];
		A[0].BITArray = new double[setSize+1];
		A[1].BITArray = new double[setSize+1];
		secondA[0].BITArray = new double[setSize+1];
		secondA[1].BITArray = new double[setSize+1];
		weightedA[0].BITArray = new double[setSize+1];
		weightedA[1].BITArray = new double[setSize+1];
		secondWeightedA[0].BITArray = new double[setSize+1];
		secondWeightedA[1].BITArray = new double[setSize+1];
		weightedSubA[0].BITArray = new double[setSize+1];
		weightedSubA[1].BITArray = new double[setSize+1];
		dominated[0].BITArray = new double[setSize+1];
		dominated[1].BITArray = new double[setSize+1];
		projection[0] = new double[setSize+1];
		projection[1] = new double[setSize+1];
		for (int i=1; i<=setSize; i++) {
			C[i] = 0.;
			A[0].BITArray[i] = 0.;
			A[1].BITArray[i] = 0.;
			secondA[0].BITArray[i] = 0.;
			secondA[1].BITArray[i] = 0.;
			weightedA[0].BITArray[i] = 0.;
			weightedA[1].BITArray[i] = 0.;
			secondWeightedA[0].BITArray[i] = 0.;
			secondWeightedA[1].BITArray[i] = 0.;
			weightedSubA[0].BITArray[i] = 0.;
			weightedSubA[1].BITArray[i] = 0.;
			dominated[0].BITArray[i] = 0.;
			dominated[1].BITArray[i] = 0.;
			projection[0][i] = 0.;
			projection[1][i] = 0.;
		}
		A[0].setSize = setSize;
		A[1].setSize = setSize;
		secondA[0].setSize = setSize;
		secondA[1].setSize = setSize;
		weightedA[0].setSize = setSize;
		weightedA[1].setSize = setSize;
		secondWeightedA[0].setSize = setSize;
		secondWeightedA[1].setSize = setSize;
		weightedSubA[0].setSize = setSize;
		weightedSubA[1].setSize = setSize;
		dominated[0].setSize = setSize;
		dominated[1].setSize = setSize;
	}

	// find the iterator w that is just heavier than index in Map, O(logn)
	inline void Determine_Other_Data(const double weightOfID, map<double, int> *Map, 
		map<double, int>::iterator &w, double &weight_, double &sweep_) 
	{
		map<double, int>::iterator ww;
		w = Map->upper_bound(weightOfID);
		if (w != Map->end()) {
			ww = w;
			++ww;
			if (ww != Map->end()) {
				sweep_ = coordinates[ww->second];
			} else {
				sweep_ = 0.;
			}
			ww = w;
			if (ww != Map->begin()) {
				--ww;
				weight_ = ww->first;
			} else {
				weight_ = 0.;
			}
		} 
	}

	// compute the intersection between lighter existing gradient and the opposite gradient that is just heavier than it
	// the opposite gradient should be lighter than current gradient
	// actually the existing gradient is either the gradient on the right of current gradient or gradients to be removed
	double Corner_Instersection(map<double, int> *WeightMap, map<double, int> *GradientMap, 
		BIT* weightedA, const int id, const double weight_, const double sweep_)
	{
		map<double, int>::iterator subw, subw2, subw3;
		subw = WeightMap->upper_bound(weight_);
		if (subw != WeightMap->end() && coordinates[subw->second] > sweep_) {
			if (subw->second > id) {
				return (weight[id] - weight_) * (coordinates[subw->second] - sweep_);
			} else {
				subw2 = GradientMap->upper_bound(sweep_);
				// since coordinates[subw->second] > sweep, we must have 
				// coordinates[subw2->second] <= coordinates[subw->second], and thus subw2->second >= subw->second
				if (subw2->second > id) {
					subw3 = WeightMap->upper_bound(weight[id]);
					return weightedA->Sum(subw->second, id) +	
						weight[id] * (coordinates[subw3->second] - sweep_) -
						weight_ * (coordinates[subw->second] - sweep_);
				} else {
					if (subw2 == GradientMap->begin()) {
						return weightedA->Sum(subw->second, subw2->second) - 
							weight[subw2->second] * sweep_ -
							weight_ * (coordinates[subw->second] - sweep_);
					} else {
						subw3 = subw2;
						--subw3;
						return weightedA->Sum(subw->second, subw2->second) - 
							weight[subw2->second] * (sweep_ - subw3->first) -
							weight_ * (coordinates[subw->second] - sweep_);
					}	
				}
			}
		} else {
			return 0.;
		}
	}

	// computing intersection of covering box
	double Corner_Instersection(map<double, int> *WeightMap, map<double, int> *GradientMap, 
		BIT* weightedA, const double coverWeight, const int nextID, const double weight_, const double sweep_)
	{
		map<double, int>::iterator subw, subw2, subw3;
		subw = WeightMap->upper_bound(weight_);
		if (subw != WeightMap->end() && coordinates[subw->second] > sweep_) {
			if (weight[subw->second] > coverWeight) {
				return (coverWeight - weight_) * (coordinates[subw->second] - sweep_);
			} else {
				subw2 = GradientMap->upper_bound(sweep_);
				if (weight[subw2->second] > coverWeight) {
					subw3 = WeightMap->upper_bound(coverWeight);
					return weightedA->Sum(subw->second, nextID) +	
						coverWeight * (coordinates[subw3->second] - sweep_) -
						weight_ * (coordinates[subw->second] - sweep_);
				} else {
					if (subw2 == GradientMap->begin()) {
						return weightedA->Sum(subw->second, subw2->second) - 
							weight[subw2->second] * sweep_ -
							weight_ * (coordinates[subw->second] - sweep_);
					} else {
						subw3 = subw2;
						--subw3;
						return weightedA->Sum(subw->second, subw2->second) - 
							weight[subw2->second] * (sweep_ - subw3->first) -
							weight_ * (coordinates[subw->second] - sweep_);
					}	
				}
			}
		} else {
			return 0.;
		}
	}

	// when subw2 and subw3 are known
	double Corner_Instersection(map<double, int> *WeightMap, map<double, int> *GradientMap, 
		BIT* weightedA, map<double, int>::iterator subw2, const double leftVolume, 
		const int id, const double weight_, const double sweep_)
	{
		map<double, int>::iterator subw;
		subw = WeightMap->upper_bound(weight_);
		if (subw != WeightMap->end() && coordinates[subw->second] > sweep_) {
			if (subw->second > id) {
				return (weight[id] - weight_) * (coordinates[subw->second] - sweep_);
			} else {
				if (subw2->second > id) {
					return weightedA->Sum(subw->second, id) + leftVolume -
						weight_ * (coordinates[subw->second] - sweep_);
				} else {
					if (subw2 == GradientMap->begin()) {
						return weightedA->Sum(subw->second, subw2->second) - leftVolume -
							weight_ * (coordinates[subw->second] - sweep_);
					} else {
						return weightedA->Sum(subw->second, subw2->second) - leftVolume -
							weight_ * (coordinates[subw->second] - sweep_);
					}
				}
			}
		} else {
			return 0.;
		}
	}

	void Insert(int index, double lastCoordinate, double coverWeight, double lazy) {
	
		// update secondGradient with Yildiz's algorithm for updating gradient structure
		Insert2(index);

		int i, j, id, gradient_id, other_gradient_id;
		vector<int> inserts;
		double sweep, delta, nextWeight, nextSweep, subBeg, secondAOfGradient[2];
		double otherWeight, otherSweep, otherWeight2, otherSweep2, kSweep, leftVolumeOfw = 0.;
		double tempGradient, tempGradientINDEX, tempGradientOTHER, tempL, tempL2, temp, tempINDEX, tempID;
		map<double, int>::iterator k, kk, k2, subk, backupk2, w, ww, www, w2, subw, subw2, tempw;

		gradient_id = type[index];
		other_gradient_id = 1 - type[index];
		// update lightest gradients with lazy
		for (i=0; i<2; i++) {
			if (!Gradient[i].empty()) {
				k = Gradient[i].end();
				--k;
				if (k != Gradient[i].begin()) {
					k2 = k;
					--k2;
					otherSweep = k2->first;
				} else {
					otherSweep = 0.;
				}
				if (!SubGradient[i].empty()) {
					k2 = SubGradient[i].end();
					--k2;
					if (otherSweep < k2->first) {
						otherSweep = k2->first;
					}
				}
				secondAOfGradient[i] = otherSweep;
				if (!Gradient[1-i].empty()) {
					C[k->second] -= lazy * (k->first - otherSweep) * (L[1-i] - Gradient[1-i].crbegin()->first);
				} else {
					C[k->second] -= lazy * (k->first - otherSweep) * L[1-i];
				}
			}
		}
		if (Gradient[gradient_id].empty()) {
			// update for empty tree
			if (Gradient[other_gradient_id].empty()) {				
				C[index] += lastCoordinate * coordinates[index] * (weight[index] - coverWeight) * L[other_gradient_id];
				V[gradient_id] += weight[index]*coordinates[index]*L[other_gradient_id];
				// dominated must be cleared because it is possible that Gradient[other_gradient_id] was ever nonempty
				dominated[gradient_id].Clear(index);
				projection[gradient_id][index] += weight[index]*coordinates[index];
				A[gradient_id].Add(index, coordinates[index]);
				weightedA[gradient_id].Add(index, weight[index]*coordinates[index]);
				Gradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
				WeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
			} else {
				dominated[other_gradient_id].Add(1, index, -lastCoordinate * coordinates[index]);
				Determine_Other_Data(weight[index], &WeightFinder[other_gradient_id], w, otherWeight, otherSweep);
				if (w != WeightFinder[other_gradient_id].end()) {
					temp = Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
						&weightedSubA[other_gradient_id], index, otherWeight, otherSweep);
					C[w->second] -= lastCoordinate * coordinates[index] *
						((weight[index] - otherWeight) * (coordinates[w->second] - otherSweep) - temp);
				}
				// make up the contribution of lightest gradient of Gradient[other_gradient_id] dominated by coverWeight
				// note that lightest gradient may also be "w" when Gradient[other_gradient_id].size() == 1
				// in this case Add() actually does not work
				if (!Gradient[other_gradient_id].empty()) {
					k = Gradient[other_gradient_id].end();
					--k;
					C[k->second] += lastCoordinate * coordinates[index] * coverWeight * (k->first - secondAOfGradient[other_gradient_id]);
				}
				C[index] += lastCoordinate * coordinates[index] * 
					( (weight[index] - coverWeight) * (L[other_gradient_id] - A[other_gradient_id].Sum(index+1, setSize)) - 
					(weightedA[other_gradient_id].Sum(1, index) - coverWeight * A[other_gradient_id].Sum(1, index)) );
				V[gradient_id] += weight[index]*coordinates[index]*(L[other_gradient_id] - 
					A[other_gradient_id].Sum(setSize) + A[other_gradient_id].Sum(index));
				V[other_gradient_id] -= coordinates[index]*weightedA[other_gradient_id].Sum(index-1);
				dominated[gradient_id].Clear(index);
				projection[gradient_id][index] += weight[index]*coordinates[index];
				A[gradient_id].Add(index, coordinates[index]);
				weightedA[gradient_id].Add(index, weight[index]*coordinates[index]);
				Gradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
				WeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
			}
		} else {
			k = Gradient[gradient_id].upper_bound(coordinates[index]);
			if (k == Gradient[gradient_id].end() || k != Gradient[gradient_id].end() && k->second < index) {
				if (Gradient[other_gradient_id].empty()) {
					tempGradientINDEX = 0.;
					tempINDEX = 0.;
					if (k == Gradient[gradient_id].end()) {
						--k;
						id = k->second;
						sweep = k->first;
						otherWeight2 = 0.;
						tempGradientINDEX += weight[index] * (coordinates[index] - sweep);
						tempINDEX += (coordinates[index] - sweep) * (weight[index] - coverWeight) * L[other_gradient_id];
						nextWeight = 0;
						subk = SubWeightFinder[gradient_id].begin();
						k2 = SubGradient[gradient_id].end();
						backupk2 = k2;	
						if (!SubGradient[gradient_id].empty()) {
							if (k2 != SubGradient[gradient_id].begin()) {
								--k2;
							} 
						}
						C[id] += lastCoordinate * L[other_gradient_id] * coverWeight * (k->first - secondAOfGradient[gradient_id]);
					} else {
						kk = k;
						++kk;
						if (kk != Gradient[gradient_id].end()) {
							nextWeight = weight[kk->second];
						} else {
							nextWeight = 0;
						}
						sweep = coordinates[index];
						subk = SubWeightFinder[gradient_id].begin();
						if (SubGradient[gradient_id].empty()) {
							k2 = SubGradient[gradient_id].end();
							backupk2 = k2;
						} else {
							k2 = SubGradient[gradient_id].upper_bound(sweep);
							backupk2 = k2;
							if (k2 != SubGradient[gradient_id].end() && nextWeight < weight[k2->second]) {
								nextWeight = weight[k2->second];
							}
							if (k2 != SubGradient[gradient_id].begin()) {
								--k2;
								subk = SubWeightFinder[gradient_id].find(weight[k2->second]);
								subBeg = k2->first;
							} else {
								subBeg = 0.;
							}
						}
						id = k->second;
						if (nextWeight == 0.) {
							C[id] += lastCoordinate * L[other_gradient_id] * coverWeight * (sweep - secondAOfGradient[gradient_id]);
						} 
						if (k == Gradient[gradient_id].begin()) {
							A[gradient_id].Add(id, -sweep);
							weightedA[gradient_id].Add(id, -weight[id]*sweep);
							A[gradient_id].Add(index, sweep);
							weightedA[gradient_id].Add(index, weight[index]*sweep);
							tempGradientINDEX += sweep * (weight[index] - weight[id]);
							tempINDEX += sweep * (weight[index] - weight[id]) * L[other_gradient_id];
							tempGradient = 0.;
							if (k2 == SubGradient[gradient_id].end() ||
								k2 == SubGradient[gradient_id].begin() && k2->first > sweep) 
							{
								tempGradient = (weight[id] - nextWeight) * sweep;
							} else {
								while (k2 != SubGradient[gradient_id].begin()) {
									tempGradient += (weight[id] - nextWeight) * (sweep - k2->first);
									sweep = k2->first;
									nextWeight = weight[k2->second];
									SubGradient[gradient_id].erase(k2--);
									weightedSubA[gradient_id].Add(subk->second, -subk->first*(sweep - k2->first));
									SubWeightFinder[gradient_id].erase(subk++);
								}
								tempGradient += (weight[id] - nextWeight) * (sweep - k2->first) + (weight[id] - weight[k2->second]) * k2->first;
								weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
								if (backupk2 != SubGradient[gradient_id].end()) {
									weightedSubA[gradient_id].Add(backupk2->second, weight[backupk2->second]*subBeg);
								}
								SubGradient[gradient_id].erase(k2);
								SubWeightFinder[gradient_id].erase(subk);
								k2 = SubGradient[gradient_id].end();
							}
							C[id] += - lastCoordinate * tempGradient * L[other_gradient_id];
							C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
							projection[gradient_id][id] -= tempGradient;
							V[gradient_id] += -weight[id]*coordinates[index]*L[other_gradient_id];
							C[index] += lastCoordinate*tempINDEX;
							V[gradient_id] += weight[index]*coordinates[index]*L[other_gradient_id];
							dominated[gradient_id].Clear(index);
							projection[gradient_id][index] += tempGradientINDEX;
							Gradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
							WeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
							return;
						} else {
							id = k->second;
							--k;
							delta = sweep - k->first;
							tempGradientINDEX += delta * (weight[index] - weight[id]);
							tempINDEX += delta * (weight[index] - weight[id]) * L[other_gradient_id];
							A[gradient_id].Add(id, - delta);
							weightedA[gradient_id].Add(id, -weight[id]*delta);
							tempGradient = 0.;
							if (k2 == SubGradient[gradient_id].end() || 
								k2 == SubGradient[gradient_id].begin() && k2->first > sweep || 
								k2->first < k->first) 
							{
								tempGradient += (weight[id] - nextWeight) * delta;
							} else {
								while (k2->first > k->first && k2 != SubGradient[gradient_id].begin()) {
									tempGradient += (weight[id] - nextWeight) * (sweep - k2->first);
									sweep = k2->first;
									nextWeight = weight[k2->second];
									SubGradient[gradient_id].erase(k2--);
									weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
									SubWeightFinder[gradient_id].erase(subk++);
								}
								if (k2->first > k->first) {
									tempGradient += (weight[id] - nextWeight) * (sweep - k2->first) + (weight[id] - weight[k2->second]) * (k2->first - k->first);								
									weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
									SubGradient[gradient_id].erase(k2);
									SubWeightFinder[gradient_id].erase(subk);
									k2 = SubGradient[gradient_id].end();
								} else {
									tempGradient += (weight[id] - nextWeight) * (sweep - k->first);
								}
							}
							C[id] += - lastCoordinate * tempGradient * L[other_gradient_id];
							C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
							projection[gradient_id][id] -= tempGradient;
							nextWeight = weight[id];
							V[gradient_id] += -weight[id]*delta*L[other_gradient_id];
							sweep = k->first;
							id = k->second;
						}
					}

					inserts.reserve(Gradient[gradient_id].size());
					while (id < index && k != Gradient[gradient_id].begin()) {
						inserts.push_back(id);
						WeightFinder[gradient_id].erase(weight[id]);
						Gradient[gradient_id].erase(k--);
						delta = sweep - k->first;
						tempGradientINDEX += delta * (weight[index] - weight[id]);
						tempINDEX += delta * (weight[index] - weight[id]) * L[other_gradient_id];
						A[gradient_id].Add(id, -delta);
						weightedA[gradient_id].Add(id, -weight[id]*delta);
						V[gradient_id] += -weight[id]*delta*L[other_gradient_id];
						if ( !(k2 == SubGradient[gradient_id].end() || 
							k2 == SubGradient[gradient_id].begin() && k2->first > sweep || 
							k2->first < k->first) )
						{
							while (k2->first > k->first && k2 != SubGradient[gradient_id].begin()) {
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[gradient_id].erase(k2--);
								weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[gradient_id].erase(subk++);
							}
							if (k2->first > k->first) {
								weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
								SubGradient[gradient_id].erase(k2);
								SubWeightFinder[gradient_id].erase(subk);
								k2 = SubGradient[gradient_id].end();
							} 
						} 
						C[id] += - lastCoordinate * projection[gradient_id][id] * L[other_gradient_id];
						C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
						projection[gradient_id][id] = 0.;
						nextWeight = weight[id];
						sweep = k->first;
						id = k->second;
					}
					if (id < index) {
						delta = coordinates[index];
						inserts.push_back(id);
						A[gradient_id].Add(id, - k->first);
						weightedA[gradient_id].Add(id, -weight[id]*k->first);
						WeightFinder[gradient_id].erase(weight[k->second]);
						V[gradient_id] += -weight[id]*k->first*L[other_gradient_id];
						Gradient[gradient_id].erase(k);
						tempGradientINDEX += sweep * (weight[index] - weight[id]);
						tempINDEX += sweep * (weight[index] - weight[id]) * L[other_gradient_id];
						if ( !(k2 == SubGradient[gradient_id].end() || 
							k2 == SubGradient[gradient_id].begin() && k2->first > sweep) )
						{
							while (k2 != SubGradient[gradient_id].begin()) {
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[gradient_id].erase(k2--);
								weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[gradient_id].erase(subk++);
							}
							weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
							SubGradient[gradient_id].erase(k2);
							SubWeightFinder[gradient_id].erase(subk);
							k2 = SubGradient[gradient_id].end();
						}					
						C[id] += - lastCoordinate * projection[gradient_id][id] * L[other_gradient_id];
						C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
						projection[gradient_id][id] = 0.;
					} else {
						delta = coordinates[index] - k->first;
						nextSweep = 0.;
						if (k != Gradient[gradient_id].begin()) {
							kk = k;
							--kk;
							nextSweep = kk->first;
						}
						tempGradient = 0.;
						if (k2 == SubGradient[gradient_id].end() || 
							k2 == SubGradient[gradient_id].begin() && k2->first > sweep) 
						{
							tempGradient = (weight[index] - nextWeight) * (sweep - nextSweep);
						} else {
							while (k2->first > nextSweep && k2->second < index && k2 != SubGradient[gradient_id].begin()) {
								tempGradient += (weight[index] - nextWeight) * (sweep - k2->first);
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[gradient_id].erase(k2--);
								weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[gradient_id].erase(subk++);
							}
							if (k2->first > nextSweep && k2->second < index) {
								tempGradient += (weight[index] - nextWeight) * (sweep - k2->first) + (weight[index] - weight[k2->second]) * (k2->first - nextSweep);
								weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
								SubGradient[gradient_id].erase(k2);
								SubWeightFinder[gradient_id].erase(subk);
								k2 = SubGradient[gradient_id].end();
							} else {
								if (k2->first > nextSweep) {
									tempGradient += (weight[index] - nextWeight) * (sweep - k2->first);
								} else {
									tempGradient += (weight[index] - nextWeight) * (sweep - nextSweep);
								}
							}
						}
						C[id] += - lastCoordinate * tempGradient * L[other_gradient_id];
						C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
						projection[gradient_id][id] -= tempGradient;
					}
					if (inserts.size() > 0) {
						for (j=0; j<inserts.size()-1; j++) {
							weightedSubA[gradient_id].Add(inserts[j], weight[inserts[j]] * (coordinates[inserts[j]] - coordinates[inserts[j+1]]));
							SubGradient[gradient_id].insert(pair<double, int>(coordinates[inserts[j]], inserts[j]));
							SubWeightFinder[gradient_id].insert(pair<double, int>(weight[inserts[j]], inserts[j]));
						}
						if (k2 != SubGradient[gradient_id].end()) {
							if (backupk2 != SubGradient[gradient_id].end() && k2->second == backupk2->second) {
								otherSweep2 = 0.;
							} else {
								otherSweep2 = k2->first;
							}
						} else {
							otherSweep2 = 0.;
						}
						if (backupk2 != SubGradient[gradient_id].end()) {
							weightedSubA[gradient_id].Add(backupk2->second, weight[backupk2->second]*(subBeg - coordinates[inserts[0]]));
						}
						weightedSubA[gradient_id].Add(inserts[j], weight[inserts[j]] * (coordinates[inserts[j]] - otherSweep2));
						SubGradient[gradient_id].insert(pair<double, int>(coordinates[inserts[j]], inserts[j]));
						SubWeightFinder[gradient_id].insert(pair<double, int>(weight[inserts[j]], inserts[j]));
						inserts.clear();
					} else {
						if (backupk2 != SubGradient[gradient_id].end()) {
							if (k2 != SubGradient[gradient_id].end() && k2->second != backupk2->second) {
								otherSweep2 = k2->first;
							} else {
								otherSweep2 = 0.;
							}
							weightedSubA[gradient_id].Add(backupk2->second, weight[backupk2->second]*(subBeg - otherSweep2));
						}
					}
					A[gradient_id].Add(index, delta);
					weightedA[gradient_id].Add(index, weight[index]*delta);
					C[index] += lastCoordinate*tempINDEX;
					V[gradient_id] += weight[index]*delta*L[other_gradient_id];
					dominated[gradient_id].Clear(index);
					projection[gradient_id][index] += tempGradientINDEX;
					Gradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
					WeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
				} else {
					double tempA = A[other_gradient_id].Sum(setSize);
					tempGradientINDEX = 0.;
					tempINDEX = 0.;
					tempL = L[other_gradient_id] - A[other_gradient_id].Sum(index+1, setSize);
					// contribution of "w" will change after current gradient is inserted
					// we do not update it for each sweep but record the changes and update it once before the next loop
					// "ww" and "leftVolumeOfw" are fixed for w and they will be reused when computing the changes
					tempGradientOTHER = 0.;
					Determine_Other_Data(weight[index], &WeightFinder[other_gradient_id], w, otherWeight, otherSweep);
					if (w != WeightFinder[other_gradient_id].end()) {
						ww = SubGradient[other_gradient_id].upper_bound(otherSweep);
						if (ww != SubGradient[other_gradient_id].end()) {
							if (ww->second > index) {
								www = SubWeightFinder[other_gradient_id].upper_bound(weight[index]);
								leftVolumeOfw = weight[index] * (coordinates[www->second] - otherSweep);
							} else {
								if (ww != SubGradient[other_gradient_id].begin()) {
									tempw = ww;
									--tempw;
									leftVolumeOfw = weight[ww->second] * (otherSweep - tempw->first);
								} else {
									leftVolumeOfw = weight[ww->second] * otherSweep;
								}
							}
						} else {
							leftVolumeOfw = 0.;
						}
					} 
					if (k == Gradient[gradient_id].end()) {
						--k;
						// in this case, no gradients in SubGradient[gradient_id] will locate between coordinates[index] and k->first
						// otherwise, it would be nondominated and thus in Gradient[gradient_id] 
						id = k->second;
						sweep = k->first;
						w2 = WeightFinder[other_gradient_id].begin();
						otherWeight2 = 0.;
						tempGradientINDEX += weight[index] * (coordinates[index] - sweep);
						tempINDEX += (coordinates[index] - sweep) * ((weight[index] - coverWeight) * tempL - 
							(weightedA[other_gradient_id].Sum(1, index) - coverWeight * A[other_gradient_id].Sum(1, index)) );
						dominated[other_gradient_id].Add(1, index, -lastCoordinate * (coordinates[index] - sweep));
						if (w != WeightFinder[other_gradient_id].end()) {
							tempGradientOTHER += (coordinates[index] - sweep) * ( (weight[index] - otherWeight) * (coordinates[w->second] - otherSweep) - 
								Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
								&weightedSubA[other_gradient_id], ww, leftVolumeOfw, index, otherWeight, otherSweep) );
						}
						// make up contribution of lightest gradient of Gradient[other_gradient_id] dominated by coverWeight
						// only if current gradient becomes new lightest gradient of Gradient[gradient_id]
						if (!Gradient[other_gradient_id].empty()) {
							k2 = Gradient[other_gradient_id].end();
							--k2;
							C[k2->second] += lastCoordinate * (coordinates[index] - sweep) * coverWeight * (k2->first - secondAOfGradient[other_gradient_id]);
						}
						nextWeight = 0;
						// in this case, sweep(=k->first) must be larger than the maximum of SubGradient
						// k2 is the closest gradient in SubGradient[gradient_id] whose coordinate is smaller than sweep if it is not SubGradient[gradient_id].begin(),
						// (if k2 == SubGradient[gradient_id].begin(), we do not change k2 because it will not be removed during the while loop)
						// k2 is calculated here to avoid using upper_bound() during the while loop, otherwise the complexity will increase O(logn)
						// the same for subk, which is the pointer for SubWeightFinder[gradient_id] corresponding to k2 with subk->second == k2->second
						// backupk2 is the gradient in SubGradient on the right of current gradient
						subk = SubWeightFinder[gradient_id].begin();
						k2 = SubGradient[gradient_id].end();
						backupk2 = k2;	
						if (!SubGradient[gradient_id].empty()) {
							if (k2 != SubGradient[gradient_id].begin()) {
								--k2;
							} 
						}
						C[id] += lastCoordinate * (L[other_gradient_id] - Gradient[other_gradient_id].crbegin()->first) * 
							coverWeight * (k->first - secondAOfGradient[gradient_id]);
					} else {
						// nextWeight is defined by gradient "k+1" (if any)
						kk = k;
						++kk;
						if (kk != Gradient[gradient_id].end()) {
							nextWeight = weight[kk->second];
						} else {
							nextWeight = 0;
						}
						sweep = coordinates[index];
						// find "k2" as above
						// subBeg is used to update weightedSubA for backupk2
						subk = SubWeightFinder[gradient_id].begin();
						if (SubGradient[gradient_id].empty()) {
							k2 = SubGradient[gradient_id].end();
							backupk2 = k2;
						} else {
							k2 = SubGradient[gradient_id].upper_bound(sweep);
							backupk2 = k2;
							if (k2 != SubGradient[gradient_id].end() && nextWeight < weight[k2->second]) {
								nextWeight = weight[k2->second];
							}
							if (k2 != SubGradient[gradient_id].begin()) {
								--k2;
								subk = SubWeightFinder[gradient_id].find(weight[k2->second]);
								subBeg = k2->first;
							} else {
								subBeg = 0.;
							}
						}
						id = k->second;
						Determine_Other_Data(weight[id], &WeightFinder[other_gradient_id], w2, otherWeight2, otherSweep2);
						if (nextWeight == 0.) {
							// contribution of id will be deducted due to index again
							// so make up contribution of id dominated by coverWeight
							C[id] += lastCoordinate * (L[other_gradient_id] - Gradient[other_gradient_id].crbegin()->first) * 
								coverWeight * (sweep - secondAOfGradient[gradient_id]);
						} 
						if (k == Gradient[gradient_id].begin()) {
							A[gradient_id].Add(id, -sweep);
							weightedA[gradient_id].Add(id, -weight[id]*sweep);
							A[gradient_id].Add(index, sweep);
							weightedA[gradient_id].Add(index, weight[index]*sweep);
							tempL2 = L[other_gradient_id] - A[other_gradient_id].Sum(id+1, setSize);
							tempGradientINDEX += sweep * (weight[index] - weight[id]);
							tempINDEX += sweep * ( weight[index] * tempL - weightedA[other_gradient_id].Sum(id, index) - weight[id] * tempL2 );
							// update dominated volumes in the other direction
							// if w2->second > index, we have w2 == w, so we only need to update contribution of w as later
							if (w2 != WeightFinder[other_gradient_id].end() && w2->second < index) {
								// update dominated for those which are not fully covered before but fully covered by current gradient
								dominated[other_gradient_id].Add(w2->second, index, -lastCoordinate*sweep);
								// make up contribution of w2 with the intersection between w2 and "k"
								temp = Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
									&weightedSubA[other_gradient_id], id, otherWeight2, otherSweep2);
								C[w2->second] += lastCoordinate * sweep * 
									((weight[id] - otherWeight2) * (coordinates[w2->second] - otherSweep2) - temp);
							}
							if (w != WeightFinder[other_gradient_id].end()) {
								// otherWeight need to be changed as weight[id] updated if w == w2, otherwise, otherWeight does not change
								otherWeight = max(otherWeight, weight[id]);
								temp = Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
									&weightedSubA[other_gradient_id], ww, leftVolumeOfw, index, otherWeight, otherSweep);
								C[w->second] -= lastCoordinate * sweep * 
									((weight[index] - otherWeight) * (coordinates[w->second] - otherSweep) - temp);
							}
							tempGradient = 0.;
							tempID = 0.;
							if (w2 != WeightFinder[other_gradient_id].end()) {
								kSweep = coordinates[w2->second];
							} else {
								kSweep = 0.;
							}
							// remove all gradients in SubGradient[gradient_id] from coordinates[index] to SubGradient[gradient_id].begin()
							if (k2 == SubGradient[gradient_id].end() ||
								k2 == SubGradient[gradient_id].begin() && k2->first > sweep) 
							{
								tempGradient = (weight[id] - nextWeight) * sweep;
								tempID = Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * sweep;
							} else {
								while (k2 != SubGradient[gradient_id].begin()) {
									tempGradient += (weight[id] - nextWeight) * (sweep - k2->first);
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first);
									sweep = k2->first;
									nextWeight = weight[k2->second];
									SubGradient[gradient_id].erase(k2--);
									weightedSubA[gradient_id].Add(subk->second, -subk->first*(sweep - k2->first));
									SubWeightFinder[gradient_id].erase(subk++);
								}
								tempGradient += (weight[id] - nextWeight) * (sweep - k2->first) + (weight[id] - weight[k2->second]) * k2->first;
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first); 
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, weight[k2->second], kSweep) * k2->first;
								weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
								if (backupk2 != SubGradient[gradient_id].end()) {
									weightedSubA[gradient_id].Add(backupk2->second, weight[backupk2->second]*subBeg);
								}
								SubGradient[gradient_id].erase(k2);
								SubWeightFinder[gradient_id].erase(subk);
								k2 = SubGradient[gradient_id].end();
							}
							C[id] += lastCoordinate * (tempID - tempGradient * tempL2);
							C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
							projection[gradient_id][id] -= tempGradient;
							V[gradient_id] += -weight[id]*coordinates[index]*(L[other_gradient_id] - tempA + A[other_gradient_id].Sum(id));
							V[other_gradient_id] -= -coordinates[index]*weightedA[other_gradient_id].Sum(id-1);
							C[index] += lastCoordinate*tempINDEX;
							V[gradient_id] += weight[index]*coordinates[index]*(L[other_gradient_id] - tempA + A[other_gradient_id].Sum(index));
							V[other_gradient_id] -= coordinates[index]*weightedA[other_gradient_id].Sum(index-1);
							dominated[gradient_id].Clear(index);
							projection[gradient_id][index] += tempGradientINDEX;
							Gradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
							WeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
							return;
						} else {
							id = k->second;
							--k;
							delta = sweep - k->first;
							tempL2 = L[other_gradient_id] - A[other_gradient_id].Sum(id+1, setSize);
							tempGradientINDEX += delta * (weight[index] - weight[id]);
							tempINDEX += delta * (weight[index] * tempL - weightedA[other_gradient_id].Sum(id, index) - weight[id] * tempL2);
							Determine_Other_Data(weight[id], &WeightFinder[other_gradient_id], w2, otherWeight2, otherSweep2);
							if (w2 != WeightFinder[other_gradient_id].end() && w2->second < index) {
								dominated[other_gradient_id].Add(w2->second, index, -lastCoordinate*delta);
								temp = Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
									&weightedSubA[other_gradient_id], id, otherWeight2, otherSweep2);
								C[w2->second] += lastCoordinate * delta * 
									((weight[id] - otherWeight2) * (coordinates[w2->second] - otherSweep2) - temp);
							}
							if (w != WeightFinder[other_gradient_id].end()) {
								otherWeight = max(otherWeight, weight[id]);
								tempGradientOTHER += delta * ( (weight[index] - otherWeight) * (coordinates[w->second] - otherSweep) - 
									Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
									&weightedSubA[other_gradient_id], ww, leftVolumeOfw, index, otherWeight, otherSweep) );
							}
							A[gradient_id].Add(id, - delta);
							weightedA[gradient_id].Add(id, -weight[id]*delta);
							// moving from coordinates[index] to k->first
							// update contribution of the gradient on the right of current gradient
							tempGradient = 0.;
							tempID = 0.;
							if (w2 != WeightFinder[other_gradient_id].end()) {
								kSweep = coordinates[w2->second];
							} else {
								kSweep = 0.;
							}
							if (k2 == SubGradient[gradient_id].end() || 
								k2 == SubGradient[gradient_id].begin() && k2->first > sweep ||
								k2->first < k->first) 
							{
								tempGradient += (weight[id] - nextWeight) * delta;
								tempID = Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * delta;
							} else {
								while (k2->first > k->first && k2 != SubGradient[gradient_id].begin()) {
									// gradient "k2" in SubGradient[gradient_id] is dominated by current gradient and we are going to move across it, so remove it
									// in this case, we must have weight[k2->second] > nextWeight so that nextWeight need to be updated, 
									// which may include 3 cases: 
									//    1) nextWeight is 0, so it should be updated;
									//    2) nextWeight is weight[id], so it should be updated;
									//    3) nextWeight is weight[(k+1)->second], so we have weight[k2->second] > weight[(k+1)->second],
									//       otherwise "k2" would be dominated by gradient "k" and gradient "k+1" and thus it should not be in SubGradient[gradient_id]
									tempGradient += (weight[id] - nextWeight) * (sweep - k2->first);
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first);
									sweep = k2->first;
									nextWeight = weight[k2->second];
									SubGradient[gradient_id].erase(k2--);
									weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
									SubWeightFinder[gradient_id].erase(subk++);
								}
								if (k2->first > k->first) {
									// moving across SubGradient.begin()
									// note that in this case it is dominated by current gradient and gradient "k", so it should be removed
									tempGradient += (weight[id] - nextWeight) * (sweep - k2->first) + (weight[id] - weight[k2->second]) * (k2->first - k->first);								
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first); 
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], id, weight[k2->second], kSweep) * (k2->first - k->first);
									weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
									SubGradient[gradient_id].erase(k2);
									SubWeightFinder[gradient_id].erase(subk);
									k2 = SubGradient[gradient_id].end();
								} else {
									// moving from sweep to k->first does not meet with new gradients in SubGradient
									tempGradient += (weight[id] - nextWeight) * (sweep - k->first);
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k->first); 
								}
							}
							C[id] += lastCoordinate * (tempID - tempGradient * tempL2);
							C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
							projection[gradient_id][id] -= tempGradient;
							nextWeight = weight[id];
							V[gradient_id] += -weight[id]*delta*(L[other_gradient_id] - tempA + A[other_gradient_id].Sum(id));
							V[other_gradient_id] -= -delta*weightedA[other_gradient_id].Sum(id-1);
							sweep = k->first;
							id = k->second;
						}
					}

					// move from sweep (== k->first) and remove gradients in Gradient
					inserts.reserve(Gradient[gradient_id].size());
					while (id < index && k != Gradient[gradient_id].begin()) {
						inserts.push_back(id);
						WeightFinder[gradient_id].erase(weight[id]);
						Gradient[gradient_id].erase(k--);
						tempL2 = L[other_gradient_id] - A[other_gradient_id].Sum(id+1, setSize);
						delta = sweep - k->first;
						tempGradientINDEX += delta * (weight[index] - weight[id]);
						tempINDEX += delta * (weight[index] * tempL - weightedA[other_gradient_id].Sum(id, index) - weight[id] * tempL2);
						A[gradient_id].Add(id, -delta);
						weightedA[gradient_id].Add(id, -weight[id]*delta);
						V[gradient_id] += -weight[id]*delta*(L[other_gradient_id] - tempA + A[other_gradient_id].Sum(id));
						V[other_gradient_id] -= -delta*weightedA[other_gradient_id].Sum(id-1);
						Determine_Other_Data(weight[id], &WeightFinder[other_gradient_id], w2, otherWeight2, otherSweep2);
						if (w2 != WeightFinder[other_gradient_id].end() && w2->second < index) {
							dominated[other_gradient_id].Add(w2->second, index, -lastCoordinate*delta);
							temp = Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
								&weightedSubA[other_gradient_id], id, otherWeight2, otherSweep2);
							C[w2->second] += lastCoordinate * delta * 
								((weight[id] - otherWeight2) * (coordinates[w2->second] - otherSweep2) - temp);
						}
						if (w != WeightFinder[other_gradient_id].end()) {
							otherWeight = max(otherWeight, weight[id]);
							tempGradientOTHER += delta * ( (weight[index] - otherWeight) * (coordinates[w->second] - otherSweep) - 
								Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
								&weightedSubA[other_gradient_id], ww, leftVolumeOfw, index, otherWeight, otherSweep) );
						}
						tempID = 0.;
						if (w2 != WeightFinder[other_gradient_id].end()) {
							kSweep = coordinates[w2->second];
						} else {
							kSweep = 0.;
						}
						if (k2 == SubGradient[gradient_id].end() || 
							k2 == SubGradient[gradient_id].begin() && k2->first > sweep || 
							k2->first < k->first ) 
						{
							tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k->first); 
						} else {
							while (k2->first > k->first && k2 != SubGradient[gradient_id].begin()) {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first);
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[gradient_id].erase(k2--);
								weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[gradient_id].erase(subk++);
							}
							if (k2->first > k->first) {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first); 
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, weight[k2->second], kSweep) * (k2->first - k->first);
								weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
								SubGradient[gradient_id].erase(k2);
								SubWeightFinder[gradient_id].erase(subk);
								k2 = SubGradient[gradient_id].end();
							} else {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k->first); 
							}
						} 
						C[id] += lastCoordinate * (tempID - projection[gradient_id][id] * tempL2);
						C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
						projection[gradient_id][id] = 0.;
						nextWeight = weight[id];
						sweep = k->first;
						id = k->second;
					}
					if (id < index) {
						// remove the begin of Gradient and sweep to 0
						delta = coordinates[index];
						inserts.push_back(id);
						A[gradient_id].Add(id, - k->first);
						weightedA[gradient_id].Add(id, -weight[id]*k->first);
						WeightFinder[gradient_id].erase(weight[k->second]);
						V[gradient_id] += -weight[id]*k->first*(L[other_gradient_id] - tempA + A[other_gradient_id].Sum(id));
						V[other_gradient_id] -= -k->first*weightedA[other_gradient_id].Sum(id-1);
						Gradient[gradient_id].erase(k);
						Determine_Other_Data(weight[id], &WeightFinder[other_gradient_id], w2, otherWeight2, otherSweep2);
						tempL2 = L[other_gradient_id] - A[other_gradient_id].Sum(id+1, setSize);
						tempGradientINDEX += sweep * (weight[index] - weight[id]);
						tempINDEX += sweep * (weight[index] * tempL - weightedA[other_gradient_id].Sum(id, index) - weight[id] * tempL2);
						if (w2 != WeightFinder[other_gradient_id].end() && w2->second < index) {
							dominated[other_gradient_id].Add(w2->second, index, -lastCoordinate*sweep);
							temp = Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
								&weightedSubA[other_gradient_id], id, otherWeight2, otherSweep2);
							C[w2->second] += lastCoordinate * sweep * 
								((weight[id] - otherWeight2) * (coordinates[w2->second] - otherSweep2) - temp);
						}
						if (w != WeightFinder[other_gradient_id].end()) {
							otherWeight = max(otherWeight, weight[id]);
							tempGradientOTHER += sweep * ( (weight[index] - otherWeight) * (coordinates[w->second] - otherSweep) - 
								Corner_Instersection(&SubWeightFinder[other_gradient_id], &SubGradient[other_gradient_id],
								&weightedSubA[other_gradient_id], ww, leftVolumeOfw, index, otherWeight, otherSweep) );
						}
						// remove all gradients in SubGradient[gradient_id] from k2 to SubGradient[gradient_id].begin()
						tempID = 0.;
						if (w2 != WeightFinder[other_gradient_id].end()) {
							kSweep = coordinates[w2->second];
						} else {
							kSweep = 0.;
						}
						if (k2 == SubGradient[gradient_id].end() || 
							k2 == SubGradient[gradient_id].begin() && k2->first > sweep) 
						{
							tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], id, nextWeight, kSweep) * sweep;
						} else {
							while (k2 != SubGradient[gradient_id].begin()) {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first);
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[gradient_id].erase(k2--);
								weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[gradient_id].erase(subk++);
							}
							tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first); 
							tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], id, weight[k2->second], kSweep) * k2->first;
							weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
							SubGradient[gradient_id].erase(k2);
							SubWeightFinder[gradient_id].erase(subk);
							k2 = SubGradient[gradient_id].end();
						}					
						C[id] += lastCoordinate * (tempID - projection[gradient_id][id] * tempL2);
						C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
						projection[gradient_id][id] = 0.;
					} else {
						// update contribution of the gradient on the left of current gradient
						// left gradient do not need consider coverWeight as its contribution is dominated by current gradient
						delta = coordinates[index] - k->first;
						nextSweep = 0.;
						if (k != Gradient[gradient_id].begin()) {
							kk = k;
							--kk;
							nextSweep = kk->first;
						}
						Determine_Other_Data(weight[index], &WeightFinder[other_gradient_id], w2, otherWeight2, otherSweep2);
						if (w2 != WeightFinder[other_gradient_id].end()) {
							kSweep = coordinates[w2->second];
						} else {
							kSweep = 0.;
						}
						tempGradient = 0.;
						tempID = 0.;
						if (k2 == SubGradient[gradient_id].end() || k2 == SubGradient[gradient_id].begin() && k2->first > sweep) {
							tempGradient = (weight[index] - nextWeight) * (sweep - nextSweep);
							tempID = Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], index, nextWeight, kSweep) * (sweep - nextSweep);
						} else {
							while (k2->first > nextSweep && k2->second < index && k2 != SubGradient[gradient_id].begin()) {
								tempGradient += (weight[index] - nextWeight) * (sweep - k2->first);
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], index, nextWeight, kSweep) * (sweep - k2->first);
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[gradient_id].erase(k2--);
								weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[gradient_id].erase(subk++);
							}
							if (k2->first > nextSweep && k2->second < index) {
								tempGradient += (weight[index] - nextWeight) * (sweep - k2->first) + (weight[index] - weight[k2->second]) * (k2->first - nextSweep);
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], index, nextWeight, kSweep) * (sweep - k2->first);
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], index, weight[k2->second], kSweep) * (k2->first - nextSweep);
								weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
								SubGradient[gradient_id].erase(k2);
								SubWeightFinder[gradient_id].erase(subk);
								k2 = SubGradient[gradient_id].end();
							} else {
								if (k2->first > nextSweep) {
									tempGradient += (weight[index] - nextWeight) * (sweep - k2->first);
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], index, nextWeight, kSweep) * (sweep - k2->first);
								} else {
									tempGradient += (weight[index] - nextWeight) * (sweep - nextSweep);
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], index, nextWeight, kSweep) * (sweep - nextSweep);
								}
							}
						}
						C[id] += lastCoordinate * (tempID - tempGradient * tempL);
						C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
						projection[gradient_id][id] -= tempGradient;
					}
					// update SubGradient[gradient_id] with elements in inserts (they are in ascending order) 
					// only insertions are needed and no gradient currently in SubGradient[gradient_id] locates between gradients in inserts
					if (inserts.size() > 0) {
						for (j=0; j<inserts.size()-1; j++) {
							weightedSubA[gradient_id].Add(inserts[j], weight[inserts[j]] * (coordinates[inserts[j]] - coordinates[inserts[j+1]]));
							SubGradient[gradient_id].insert(pair<double, int>(coordinates[inserts[j]], inserts[j]));
							SubWeightFinder[gradient_id].insert(pair<double, int>(weight[inserts[j]], inserts[j]));
						}
						if (k2 != SubGradient[gradient_id].end()) {
							if (backupk2 != SubGradient[gradient_id].end() && k2->second == backupk2->second) {
								otherSweep2 = 0.;
							} else {
								otherSweep2 = k2->first;
							}
						} else {
							otherSweep2 = 0.;
						}
						if (backupk2 != SubGradient[gradient_id].end()) {
							weightedSubA[gradient_id].Add(backupk2->second, weight[backupk2->second]*(subBeg - coordinates[inserts[0]]));
						}
						weightedSubA[gradient_id].Add(inserts[j], weight[inserts[j]] * (coordinates[inserts[j]] - otherSweep2));
						SubGradient[gradient_id].insert(pair<double, int>(coordinates[inserts[j]], inserts[j]));
						SubWeightFinder[gradient_id].insert(pair<double, int>(weight[inserts[j]], inserts[j]));
						inserts.clear();
					} else {
						if (backupk2 != SubGradient[gradient_id].end()) {
							if (k2 != SubGradient[gradient_id].end() && k2->second != backupk2->second) {
								otherSweep2 = k2->first;
							} else {
								otherSweep2 = 0.;
							}
							weightedSubA[gradient_id].Add(backupk2->second, weight[backupk2->second]*(subBeg - otherSweep2));
						}
					}
					// update contribution of w
					if (w != WeightFinder[other_gradient_id].end()) {
						C[w->second] -= lastCoordinate*tempGradientOTHER;
					}
					// update contribution of current gradient
					A[gradient_id].Add(index, delta);
					weightedA[gradient_id].Add(index, weight[index]*delta);
					C[index] += lastCoordinate*tempINDEX;
					V[gradient_id] += weight[index]*delta*(L[other_gradient_id] - tempA + A[other_gradient_id].Sum(index));
					V[other_gradient_id] -= delta*weightedA[other_gradient_id].Sum(index-1);
					dominated[gradient_id].Clear(index);
					projection[gradient_id][index] += tempGradientINDEX;
					Gradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
					WeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
				}
			} else {
				// current gradient is dominated by "k" in Gradient[gradient_id]
				// it does not affect dominated volumes of gradients in the other direction
				// it can at most affect the contribution of "k" in this gradient
				// because dominated volumes of other gradients have been deducted due to "k"
				id = k->second;
				kk = k;
				++kk;
				if (kk == Gradient[gradient_id].end()) {
					nextWeight = 0.;
				} else {
					if (kk->second < index) {
						nextWeight = weight[kk->second];
					} else {
						// current gradient is dominated by "k" and "kk"
						// no update is needed for SubGradient[gradient_id] nor dominated
						return;
					}
				}
				if (k != Gradient[gradient_id].begin()) {
					--k;
					nextSweep = k->first;
				} else {
					nextSweep = 0.;
				}
				if (Gradient[other_gradient_id].empty()) {
					if (SubGradient[gradient_id].empty()) {
						if (nextWeight == 0) {
							C[id] += lastCoordinate * L[other_gradient_id] * coverWeight * (coordinates[index] - nextSweep);
						}
						C[id] -= lastCoordinate * (coordinates[index] - nextSweep) * (weight[index] - nextWeight) * L[other_gradient_id];
						C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
						projection[gradient_id][id] -= (weight[index] - nextWeight) * (coordinates[index] - nextSweep);
						weightedSubA[gradient_id].Add(index, weight[index]*coordinates[index]);
						SubGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
						SubWeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
					} else {
						k2 = SubGradient[gradient_id].upper_bound(coordinates[index]);
						if (k2 == SubGradient[gradient_id].end() || k2 != SubGradient[gradient_id].end() && k2->second < index) {
							// current gadient is not dominated by SubGradient[gradient_id]
							backupk2 = k2;
							if (k2 == SubGradient[gradient_id].end()) {
								--k2;
								if (nextWeight == 0) {
									C[id] += lastCoordinate * L[other_gradient_id] * coverWeight * (coordinates[index] - max(k2->first, nextSweep));
								}
							} else {
								nextWeight = max(nextWeight, weight[k2->second]);
								if (k2 == SubGradient[gradient_id].begin()) {
									C[id] -= lastCoordinate * (coordinates[index] - nextSweep) * (weight[index] - nextWeight) * L[other_gradient_id];
									C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
									projection[gradient_id][id] -= (weight[index] - nextWeight) * (coordinates[index] - nextSweep);
									weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*coordinates[index]);
									weightedSubA[gradient_id].Add(index, weight[index]*coordinates[index]);
									SubGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
									SubWeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
									return;
								} else {
									--k2;
									weightedSubA[gradient_id].Add(backupk2->second, weight[backupk2->second]*(k2->first - coordinates[index]));
								}
							}		
							tempGradient = 0.;
							if (k2->first < nextSweep) {
								// similar to the non-dominated case, here we must have k2->second > index
								// otherwise, we have 0 < k2->first < nextSweep, and k2->second < index < id
								// therefore "k2" is dominated by "k" and the left gradient of "k", which is a contradiction
								// thus, "k2" is not dominated by current gradient and does not need to be removed
								delta = coordinates[index] - k2->first;
								tempGradient += (weight[index] - nextWeight) * (coordinates[index] - nextSweep);
							} else {
								sweep = coordinates[index];
								subk = SubWeightFinder[gradient_id].find(weight[k2->second]);
								while (sweep > nextSweep && k2->second < index && k2 != SubGradient[gradient_id].begin()) {
									tempGradient += (weight[index] - nextWeight) * (sweep - k2->first);
									sweep = k2->first;
									nextWeight = weight[k2->second];
									SubGradient[gradient_id].erase(k2--);
									weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
									SubWeightFinder[gradient_id].erase(subk++);
								}
								if (sweep > nextSweep && k2->second < index)  {
									delta = coordinates[index];
									tempGradient += (weight[index] - nextWeight) * (sweep - k2->first) +
										(weight[index] - weight[k2->second]) * (k2->first - nextSweep);
									weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
									SubGradient[gradient_id].erase(k2);
									SubWeightFinder[gradient_id].erase(subk);
								} else {
									delta = coordinates[index] - k2->first;
									if (k2->first > nextSweep) {
										tempGradient += (weight[index] - nextWeight) * (sweep - k2->first);
									} else {
										tempGradient += (weight[index] - nextWeight) * (sweep - nextSweep);
									} 
								}
							}
							C[id] += - lastCoordinate * tempGradient * L[other_gradient_id];
							C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
							projection[gradient_id][id] -= tempGradient;
							weightedSubA[gradient_id].Add(index, weight[index]*delta);
							SubGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
							SubWeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
						}
					}
				} else {
					tempL = L[other_gradient_id] - A[other_gradient_id].Sum(index+1, setSize);
					Determine_Other_Data(weight[index], &WeightFinder[other_gradient_id], w2, otherWeight2, otherSweep2);
					if (w2 != WeightFinder[other_gradient_id].end()) {
						kSweep = coordinates[w2->second];
					} else {
						kSweep = 0.;
					}
					if (SubGradient[gradient_id].empty()) {
						if (nextWeight == 0) {
							C[id] += lastCoordinate * (L[other_gradient_id] - Gradient[other_gradient_id].crbegin()->first) * 
								coverWeight * (coordinates[index] - nextSweep);
						}
						C[id] -= lastCoordinate * (coordinates[index] - nextSweep) * ( (weight[index] - nextWeight) * tempL -
							Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
							&weightedA[other_gradient_id], index, nextWeight, kSweep) );
						C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
						projection[gradient_id][id] -= (weight[index] - nextWeight) * (coordinates[index] - nextSweep);
						weightedSubA[gradient_id].Add(index, weight[index]*coordinates[index]);
						SubGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
						SubWeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
					} else {
						k2 = SubGradient[gradient_id].upper_bound(coordinates[index]);
						if (k2 == SubGradient[gradient_id].end() || k2 != SubGradient[gradient_id].end() && k2->second < index) {
							// current gadient is not dominated by SubGradient[gradient_id]
							backupk2 = k2;
							if (k2 == SubGradient[gradient_id].end()) {
								--k2;
								if (nextWeight == 0) {
									C[id] += lastCoordinate * (L[other_gradient_id] - Gradient[other_gradient_id].crbegin()->first) * 
										coverWeight * (coordinates[index] - max(k2->first, nextSweep));
								}
							} else {
								nextWeight = max(nextWeight, weight[k2->second]);
								if (k2 == SubGradient[gradient_id].begin()) {
									C[id] -= lastCoordinate * (coordinates[index] - nextSweep) * ( (weight[index] - nextWeight) * tempL -
										Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], index, nextWeight, kSweep) );
									C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id];
									projection[gradient_id][id] -= (weight[index] - nextWeight) * (coordinates[index] - nextSweep);
									weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*coordinates[index]);
									weightedSubA[gradient_id].Add(index, weight[index]*coordinates[index]);
									SubGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
									SubWeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
									return;
								} else {
									--k2;
									weightedSubA[gradient_id].Add(backupk2->second, weight[backupk2->second]*(k2->first - coordinates[index]));
								}
							}
							tempGradient = 0.;
							tempID = 0.;
							if (k2->first < nextSweep) {
								// similar to the non-dominated case, here we must have k2->second > index
								// otherwise, we have 0 < k2->first < nextSweep, and k2->second < index < id
								// therefore "k2" is dominated by "k" and the left gradient of "k", which is a contradiction
								// thus, "k2" is not dominated by current gradient and does not need to be removed
								delta = coordinates[index] - k2->first;
								tempGradient += (weight[index] - nextWeight) * (coordinates[index] - nextSweep);
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], index, nextWeight, kSweep) * (coordinates[index] - nextSweep);
							} else {
								sweep = coordinates[index];
								subk = SubWeightFinder[gradient_id].find(weight[k2->second]);
								while (sweep > nextSweep && k2->second < index && k2 != SubGradient[gradient_id].begin()) {
									tempGradient += (weight[index] - nextWeight) * (sweep - k2->first);
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], index, nextWeight, kSweep) * (sweep - k2->first);
									sweep = k2->first;
									nextWeight = weight[k2->second];
									SubGradient[gradient_id].erase(k2--);
									weightedSubA[gradient_id].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
									SubWeightFinder[gradient_id].erase(subk++);
								}
								if (sweep > nextSweep && k2->second < index)  {
									delta = coordinates[index];
									tempGradient += (weight[index] - nextWeight) * (sweep - k2->first) +
										(weight[index] - weight[k2->second]) * (k2->first - nextSweep);
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], index, nextWeight, kSweep) * (sweep - k2->first);
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], index, weight[k2->second], kSweep) * (k2->first - nextSweep);
									weightedSubA[gradient_id].Add(k2->second, -weight[k2->second]*k2->first);
									SubGradient[gradient_id].erase(k2);
									SubWeightFinder[gradient_id].erase(subk);
								} else {
									delta = coordinates[index] - k2->first;
									if (k2->first > nextSweep) {
										tempGradient += (weight[index] - nextWeight) * (sweep - k2->first);
										tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
											&weightedA[other_gradient_id], index, nextWeight, kSweep) * (sweep - k2->first);
									} else {
										tempGradient += (weight[index] - nextWeight) * (sweep - nextSweep);
										tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
											&weightedA[other_gradient_id], index, nextWeight, kSweep) * (sweep - nextSweep);
									} 
								}
							}
							C[id] += lastCoordinate * (tempID - tempGradient * tempL);
							C[id] += dominated[gradient_id].Clear(id) * projection[gradient_id][id]; 
							projection[gradient_id][id] -= tempGradient;
							weightedSubA[gradient_id].Add(index, weight[index]*delta);
							SubGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
							SubWeightFinder[gradient_id].insert(pair<double, int>(weight[index], index));
						}
					}
				}
			}
		}
	}

	// this algorithm is the same as that in Yildiz's paper
	void Insert2(int index) {
		
		// insert the gradient whose weight ranks index
		map<double, int>::iterator k, kk;
		int id, gradient_id, other_gradient_id;
		double delta, tempA;
		gradient_id = type[index];
		other_gradient_id = 1 - type[index];
		if (secondGradient[gradient_id].empty()) {
			// update for empty case
			secondA[gradient_id].Add(index, coordinates[index]);
			secondWeightedA[gradient_id].Add(index, weight[index]*coordinates[index]);
			if (secondGradient[other_gradient_id].empty()) {
				secondV[gradient_id] += weight[index]*coordinates[index]*L[other_gradient_id];
			} else {
				secondV[gradient_id] += weight[index]*coordinates[index]*(L[other_gradient_id] - 
					secondA[other_gradient_id].Sum(setSize) + secondA[other_gradient_id].Sum(index));
				secondV[other_gradient_id] -= coordinates[index]*secondWeightedA[other_gradient_id].Sum(index-1);
			}
			secondGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
		} else {
			k = secondGradient[gradient_id].upper_bound(coordinates[index]);
			// find the closest gradient "k" whose coordinate is greater than this gradient (if any)
			// insert the gradient if it is not overlapped by "k"
			if (k == secondGradient[gradient_id].end() || k != secondGradient[gradient_id].end() && k->second < index) {
				if (secondGradient[other_gradient_id].empty()) {
					if (k == secondGradient[gradient_id].end()) {
						// this gradient is not in the gradient
						--k;
					} else {
						// this gradient is in the gradient
						if (k == secondGradient[gradient_id].begin()) {
							// "k" has the smallest coordinate, directly finish the update
							// update secondA-value for gradient "k"
							delta = - coordinates[index];
							secondA[gradient_id].Add(k->second, delta);
							secondWeightedA[gradient_id].Add(k->second, weight[k->second]*delta);
							secondV[gradient_id] += weight[k->second]*delta*L[other_gradient_id];
							// update secondA-value for this gradient
							secondA[gradient_id].Add(index, coordinates[index]);
							secondWeightedA[gradient_id].Add(index, weight[index]*coordinates[index]);
							secondV[gradient_id] += weight[index]*coordinates[index]*L[other_gradient_id];
							secondGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
							return;
						} else {
							// (k-1) is available now
							// update secondA-value for gradient "k", and backward traverse gradients with "k"
							id = k->second;
							--k;
							delta = k->first - coordinates[index];
							secondA[gradient_id].Add(id, delta);
							secondWeightedA[gradient_id].Add(id, weight[id]*delta);
							secondV[gradient_id] += weight[id]*delta*L[other_gradient_id];
						}
					} 
					kk = k;
					// remove all gradients overlapped by this gradient
					// backward traverse gradients with "kk" when "--kk" is available
					while (k->second < index && k != secondGradient[gradient_id].begin()) {
						--kk;
						delta = kk->first - k->first;
						secondA[gradient_id].Add(k->second, delta);
						secondWeightedA[gradient_id].Add(k->second, weight[k->second]*delta);
						secondV[gradient_id] += weight[k->second]*delta*L[other_gradient_id];
						secondGradient[gradient_id].erase(k);
						k = kk;
					}
					if (k->second < index) {
						// in this case, "k" is secondGradient[gradient_id].begin() and k->second < index, so remove it
						delta = - k->first;
						secondA[gradient_id].Add(k->second, delta);
						secondWeightedA[gradient_id].Add(k->second, weight[k->second]*delta);
						secondV[gradient_id] += weight[k->second]*delta*L[other_gradient_id];
						secondGradient[gradient_id].erase(k);
						delta = coordinates[index];
					} else {
						delta = coordinates[index] - k->first;
					}
					secondA[gradient_id].Add(index, delta);
					secondWeightedA[gradient_id].Add(index, weight[index]*delta);
					secondV[gradient_id] += weight[index]*delta*L[other_gradient_id];
					secondGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
				} else {
					tempA = secondA[other_gradient_id].Sum(setSize);
					if (k == secondGradient[gradient_id].end()) {
						// this gradient is not in the gradient
						--k;
					} else {
						// this gradient is in the gradient
						if (k == secondGradient[gradient_id].begin()) {
							// "k" has the smallest coordinate, directly finish the update
							// update secondA-value for gradient "k"
							delta = - coordinates[index];
							secondA[gradient_id].Add(k->second, delta);
							secondWeightedA[gradient_id].Add(k->second, weight[k->second]*delta);
							secondV[gradient_id] += weight[k->second]*delta*(L[other_gradient_id] - tempA + secondA[other_gradient_id].Sum(k->second));
							secondV[other_gradient_id] -= delta*secondWeightedA[other_gradient_id].Sum(k->second-1);
							// update secondA-value for this gradient
							secondA[gradient_id].Add(index, coordinates[index]);
							secondWeightedA[gradient_id].Add(index, weight[index]*coordinates[index]);
							secondV[gradient_id] += weight[index]*coordinates[index]*(L[other_gradient_id] - tempA + secondA[other_gradient_id].Sum(index));
							secondV[other_gradient_id] -= coordinates[index]*secondWeightedA[other_gradient_id].Sum(index-1);
							secondGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
							return;
						} else {
							// (k-1) is available now
							// update secondA-value for gradient "k", and backward traverse gradients with "k"
							id = k->second;
							--k;
							delta = k->first - coordinates[index];
							secondA[gradient_id].Add(id, delta);
							secondWeightedA[gradient_id].Add(id, weight[id]*delta);
							secondV[gradient_id] += weight[id]*delta*(L[other_gradient_id] - tempA + secondA[other_gradient_id].Sum(id));
							secondV[other_gradient_id] -= delta*secondWeightedA[other_gradient_id].Sum(id-1);
						}
					} 
					kk = k;
					// remove all gradients overlapped by this gradient
					// backward traverse gradients with "kk" when "--kk" is available
					while (k->second < index && k != secondGradient[gradient_id].begin()) {
						--kk;
						delta = kk->first - k->first;
						secondA[gradient_id].Add(k->second, delta);
						secondWeightedA[gradient_id].Add(k->second, weight[k->second]*delta);
						secondV[gradient_id] += weight[k->second]*delta*(L[other_gradient_id] - tempA + secondA[other_gradient_id].Sum(k->second));
						secondV[other_gradient_id] -= delta*secondWeightedA[other_gradient_id].Sum(k->second-1);
						secondGradient[gradient_id].erase(k);
						k = kk;
					}
					if (k->second < index) {
						// in this case, "k" is secondGradient[gradient_id].begin() and k->second < index, so remove it
						delta = - k->first;
						secondA[gradient_id].Add(k->second, delta);
						secondWeightedA[gradient_id].Add(k->second, weight[k->second]*delta);
						secondV[gradient_id] += weight[k->second]*delta*(L[other_gradient_id] - tempA + secondA[other_gradient_id].Sum(k->second));
						secondV[other_gradient_id] -= delta*secondWeightedA[other_gradient_id].Sum(k->second-1);
						secondGradient[gradient_id].erase(k);
						delta = coordinates[index];
					} else {
						delta = coordinates[index] - k->first;
					}
					secondA[gradient_id].Add(index, delta);
					secondWeightedA[gradient_id].Add(index, weight[index]*delta);
					secondV[gradient_id] += weight[index]*delta*(L[other_gradient_id] - tempA + secondA[other_gradient_id].Sum(index));
					secondV[other_gradient_id] -= delta*secondWeightedA[other_gradient_id].Sum(index-1);
					secondGradient[gradient_id].insert(pair<double, int>(coordinates[index], index));
				}
			}
		}
	}

	// return lowestWeight and secondLowestWeight respect to gradients which are not dominated by covering box
	void Remove(double fullyCoverWeight, double coverWeight, double lastCoordinate, double lazy, double &lowestWeight, double &secondLowestWeight) {

		lowestWeight = maxWeight;
		secondLowestWeight = maxWeight;
		int i, id;
		double sweep, delta, nextWeight, nextSweep, otherWeight, otherSweep, kSweep, tempGradient, tempL, tempID;
		map<double, int>::iterator k, kk, k2, subk, backupk, w;

		int other_gradient_id;
		for (i=0; i<2; i++) {
			// clear dominated of gradients in Gradient[i] but not update Gradient/WeightFinder/A/weightedA
			// since the update of Gradient[other_gradient_id] requires the old information of Gradient[i]
			// therefore we only update subgradients, projection, and dominated here
			nextWeight = 0.;
			other_gradient_id = 1 - i;
			if (!Gradient[i].empty()) {
				if (Gradient[other_gradient_id].empty()) {
					k2 = SubGradient[i].end();
					if (k2 != SubGradient[i].begin()) {
						--k2;
						otherSweep = k2->first;
					} else {
						otherSweep = 0.;
					}
					k = Gradient[i].end();
					--k;
					if (k != Gradient[i].begin()) {
						kk = k;
						--kk;
						if (otherSweep < kk->first) {
							otherSweep = kk->first;
						}
					} 
					C[k->second] -= lazy * (k->first - otherSweep) * L[other_gradient_id];
					C[k->second] += lastCoordinate * coverWeight * (k->first - otherSweep) * L[other_gradient_id];
					backupk = k;
					subk = SubWeightFinder[i].begin();
					sweep = k->first;
					id = k->second;
					while (weight[k->second] < fullyCoverWeight && k != Gradient[i].begin()) {
						--k;
						delta = sweep - k->first;
						if (k2 != SubGradient[i].end() && k2->first > k->first) {
							while (k2->first > k->first && k2 != SubGradient[i].begin()) {
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[i].erase(k2--);
								weightedSubA[i].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[i].erase(subk++);
							}
							if (k2->first > k->first) {
								weightedSubA[i].Add(k2->second, -weight[k2->second]*k2->first);
								SubGradient[i].erase(k2);
								SubWeightFinder[i].erase(subk);
								k2 = SubGradient[i].end();
							} 
						} 
						C[id] += - lastCoordinate * projection[i][id] * L[other_gradient_id];
						C[id] += dominated[i].Clear(id) * projection[i][id];
						projection[i][id] = 0.;
						nextWeight = weight[id];
						sweep = k->first;
						id = k->second;
					}
					if (weight[k->second] < fullyCoverWeight) {
						delta = k->first;
						if (k2 != SubGradient[i].end()) {
							while (k2 != SubGradient[i].begin()) {
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[i].erase(k2--);
								weightedSubA[i].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[i].erase(subk++);
							}
							weightedSubA[i].Add(k2->second, -weight[k2->second]*k2->first);
							SubGradient[i].erase(k2);
							SubWeightFinder[i].erase(subk);
							k2 = SubGradient[i].end();
						}	
						C[id] += -lastCoordinate * projection[i][id] * L[other_gradient_id];
						C[id] += dominated[i].Clear(id) * projection[i][id];
						projection[i][id] = 0.;
					} else {
						nextSweep = 0.;
						if (k != Gradient[i].begin()) {
							kk = k;
							--kk;
							nextSweep = kk->first;
						}
						C[id] += dominated[i].Clear(id) * projection[i][id];
						double tempGradient2;
						if (k2 == SubGradient[i].end()) {
							tempGradient2 = nextWeight * (sweep - nextSweep);
						} else {
							tempGradient2 = nextWeight * (sweep - max(nextSweep, k2->first));
						}	
						if (k2 != SubGradient[i].end() && k2->first > nextSweep) {
							while (weight[k2->second] < fullyCoverWeight && k2 != SubGradient[i].begin()) {
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[i].erase(k2--);
								weightedSubA[i].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								tempGradient2 += weight[subk->second]*(sweep - max(nextSweep, k2->first));
								SubWeightFinder[i].erase(subk++);
							}
							if (weight[k2->second] < fullyCoverWeight) {
								tempGradient2 += weight[k2->second]*(k2->first - nextSweep);
								SubGradient[i].erase(k2);
								SubWeightFinder[i].erase(subk);
								k2 = SubGradient[i].end();
							} 
						}
						C[id] += lastCoordinate * tempGradient2 * L[other_gradient_id];
						if (!SubGradient[i].empty() && k2->first > nextSweep) {
							nextSweep = k2->first;
						}
						C[id] -= lastCoordinate * (k->first - nextSweep) * fullyCoverWeight * L[other_gradient_id];
						projection[i][id] += tempGradient2;
					}
					if (k2 != SubGradient[i].end() && weight[k2->second] < secondLowestWeight) {
						secondLowestWeight = weight[k2->second];
					}
				} else {
					k2 = SubGradient[i].end();
					if (k2 != SubGradient[i].begin()) {
						--k2;
						otherSweep = k2->first;
					} else {
						otherSweep = 0.;
					}
					k = Gradient[i].end();
					--k;
					if (k != Gradient[i].begin()) {
						kk = k;
						--kk;
						if (otherSweep < kk->first) {
							otherSweep = kk->first;
						}
					} 
					// clear lazy
					if (!Gradient[other_gradient_id].empty()) {
						// L[other_gradient_id] - Gradient[other_gradient_id].crbegin()->first is actually one length of "A" for leaf node
						C[k->second] -= lazy * (k->first - otherSweep) * (L[other_gradient_id] - Gradient[other_gradient_id].crbegin()->first);
						// make up dominated dominated by coverWeight 
						// at last we will either remove "k" or deduct dominated dominated by fullyCoverWeight
						// in both cases this part will be deducted again, so we need to make it up here
						// here we do not need corner_intersection because coverWeight is smaller than any weight in the gradient structure
						C[k->second] += lastCoordinate * coverWeight * (k->first - otherSweep) * (L[other_gradient_id] - Gradient[other_gradient_id].crbegin()->first);

					} else {
						C[k->second] -= lazy * (k->first - otherSweep) * L[other_gradient_id];
						C[k->second] += lastCoordinate * coverWeight * (k->first - otherSweep) * L[other_gradient_id];
					}
					backupk = k;
					subk = SubWeightFinder[i].begin();
					sweep = k->first;
					id = k->second;
					while (weight[k->second] < fullyCoverWeight && k != Gradient[i].begin()) {
						--k;
						tempL = L[other_gradient_id] - A[other_gradient_id].Sum(id+1, setSize);
						delta = sweep - k->first;
						tempID = 0.;
						w = WeightFinder[other_gradient_id].upper_bound(weight[id]);
						if (w != WeightFinder[other_gradient_id].end()) {
							kSweep = coordinates[w->second];
						} else {
							kSweep = 0.;
						}
						if (k2 == SubGradient[i].end() || k2->first < k->first) {
							tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k->first); 
						} else {
							while (k2->first > k->first && k2 != SubGradient[i].begin()) {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first);
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[i].erase(k2--);
								weightedSubA[i].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[i].erase(subk++);
							}
							if (k2->first > k->first) {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first); 
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, weight[k2->second], kSweep) * (k2->first - k->first);
								weightedSubA[i].Add(k2->second, -weight[k2->second]*k2->first);
								SubGradient[i].erase(k2);
								SubWeightFinder[i].erase(subk);
								k2 = SubGradient[i].end();
							} else {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k->first); 
							}
						} 
						C[id] += lastCoordinate * (tempID - projection[i][id] * tempL);
						C[id] += dominated[i].Clear(id) * projection[i][id];
						projection[i][id] = 0.;
						nextWeight = weight[id];
						sweep = k->first;
						id = k->second;
					}
					if (weight[k->second] < fullyCoverWeight) {
						// remove the begin of Gradient and sweep to 0
						delta = k->first;
						w = WeightFinder[other_gradient_id].upper_bound(weight[id]);
						tempL = L[other_gradient_id] - A[other_gradient_id].Sum(id+1, setSize);
						// remove all gradients in SubGradient[i] from k2 to SubGradient[i].begin()
						tempID = 0.;
						if (w != WeightFinder[other_gradient_id].end()) {
							kSweep = coordinates[w->second];
						} else {
							kSweep = 0.;
						}
						if (k2 == SubGradient[i].end()) {
							tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], id, nextWeight, kSweep) * sweep;
						} else {
							while (k2 != SubGradient[i].begin()) {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first);
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[i].erase(k2--);
								weightedSubA[i].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								SubWeightFinder[i].erase(subk++);
							}
							tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], id, nextWeight, kSweep) * (sweep - k2->first); 
							tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], id, weight[k2->second], kSweep) * k2->first;
							weightedSubA[i].Add(k2->second, -weight[k2->second]*k2->first);
							SubGradient[i].erase(k2);
							SubWeightFinder[i].erase(subk);
							k2 = SubGradient[i].end();
						}	
						C[id] += lastCoordinate * (tempID - projection[i][id] * tempL);
						C[id] += dominated[i].Clear(id) * projection[i][id];
						projection[i][id] = 0.;
					} else {
						// update contribution of new lightest gradient (may not change) with fullyCoverWeight
						// we first remove subgradients dominated by fullyCoverWeight and increase the contribution
						// and then deduct the contribution dominated by fullyCoverWeight
						nextSweep = 0.;
						if (k != Gradient[i].begin()) {
							kk = k;
							--kk;
							nextSweep = kk->first;
						}
						C[id] += dominated[i].Clear(id) * projection[i][id];
						// make up dominated dominated by gradients to be removed in projection
						// this also work if nextWeight == 0, that is, lightest gradient does not change
						double tempGradient2;
						if (k2 == SubGradient[i].end()) {
							tempGradient2 = nextWeight * (sweep - nextSweep);
						} else {
							tempGradient2 = nextWeight * (sweep - max(nextSweep, k2->first));
						}	
						int kID;
						Determine_Other_Data(fullyCoverWeight, &WeightFinder[other_gradient_id], w, otherWeight, otherSweep);
						if (w != WeightFinder[other_gradient_id].end()) {
							kID = w->second;
							kSweep = coordinates[w->second];
						} else {
							kID = k->second; // setSize is also OK, I think
							kSweep = 0.;
						}
						tempL = L[other_gradient_id] - kSweep;
						tempGradient = 0.;
						tempID = 0.;
						// note that if between fullyCoverWeight and nextWeight there are some gradients in Gradient[other_gradient_id]
						// it will be/have been fully covered by fullyCoverWeight and updated in (other_gradient_id)-th loop
						if (k2 == SubGradient[i].end() || k2->first < nextSweep) {
							tempID = Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
								&weightedA[other_gradient_id], fullyCoverWeight, kID, nextWeight, kSweep) * (sweep - nextSweep);
						} else {
							while (weight[k2->second] < fullyCoverWeight && k2 != SubGradient[i].begin()) {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], fullyCoverWeight, kID, nextWeight, kSweep) * (sweep - k2->first);
								sweep = k2->first;
								nextWeight = weight[k2->second];
								SubGradient[i].erase(k2--);
								weightedSubA[i].Add(subk->second, -weight[subk->second]*(sweep - k2->first));
								tempGradient2 += weight[subk->second]*(sweep - max(nextSweep, k2->first));
								SubWeightFinder[i].erase(subk++);
							}
							if (weight[k2->second] < fullyCoverWeight) {
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], fullyCoverWeight, kID, nextWeight, kSweep) * (sweep - k2->first);
								tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
									&weightedA[other_gradient_id], fullyCoverWeight, kID, weight[k2->second], kSweep) * (k2->first - nextSweep);
								weightedSubA[i].Add(k2->second, -weight[k2->second]*k2->first);
								tempGradient2 += weight[k2->second]*(k2->first - nextSweep);
								SubGradient[i].erase(k2);
								SubWeightFinder[i].erase(subk);
								k2 = SubGradient[i].end();
							} else {
								if (k2->first > nextSweep) {
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], fullyCoverWeight, kID, nextWeight, kSweep) * (sweep - k2->first);
								} else {
									tempID += Corner_Instersection(&WeightFinder[other_gradient_id], &Gradient[other_gradient_id],
										&weightedA[other_gradient_id], fullyCoverWeight, kID, nextWeight, kSweep) * (sweep - nextSweep);
								}
							}
						}
						// make up volume domianted by Gradient[other_gradient_id]
						C[id] += lastCoordinate * tempID;
						// make up volume domianted by gradients and subgradients of Gradient[i]
						C[id] += lastCoordinate * tempGradient2 * tempL;
						if (!SubGradient[i].empty() && k2->first > nextSweep) {
							nextSweep = k2->first;
						}
						// deduct dominated dominated by fullyCoverWeight
						C[id] -= lastCoordinate * (k->first - nextSweep) * fullyCoverWeight * tempL;
						projection[i][id] += tempGradient2;
					}
					if (k2 != SubGradient[i].end() && weight[k2->second] < secondLowestWeight) {
						secondLowestWeight = weight[k2->second];
					}
				}
			}
		}
		// remove dominated gradients and update Maps/BITs here
		// also update V[0] and V[1]
		for (i=0; i<2; i++) {
			if (!Gradient[i].empty()) {
				if (Gradient[1-i].empty()) {
					k = Gradient[i].end();
					--k;
					sweep = k->first;
					id = k->second;
					while (weight[id] < fullyCoverWeight && k != Gradient[i].begin()) {
						WeightFinder[i].erase(weight[id]);
						Gradient[i].erase(k--);
						A[i].Add(id, k->first - sweep);
						weightedA[i].Add(id, weight[id]*(k->first - sweep));
						V[i] += weight[id]*(k->first - sweep)*L[1-i];
						sweep = k->first;
						id = k->second;
					}
					if (weight[id] < fullyCoverWeight) {
						WeightFinder[i].erase(weight[id]);
						A[i].Add(id, -k->first);
						weightedA[i].Add(id, -weight[id]*k->first);
						V[i] += -weight[id]*k->first*L[1-i];
						Gradient[i].erase(k);
					} else {
						if (lowestWeight > weight[k->second]) {
							lowestWeight = weight[k->second];
						}
					}
				} else {
					k = Gradient[i].end();
					--k;
					sweep = k->first;
					id = k->second;
					while (weight[id] < fullyCoverWeight && k != Gradient[i].begin()) {
						WeightFinder[i].erase(weight[id]);
						Gradient[i].erase(k--);
						A[i].Add(id, k->first - sweep);
						weightedA[i].Add(id, weight[id]*(k->first - sweep));
						V[i] += weight[id]*(k->first - sweep)*(L[1-i] - A[1-i].Sum(id+1, setSize));
						V[1-i] += (sweep - k->first)*weightedA[1-i].Sum(id-1);
						sweep = k->first;
						id = k->second;
					}
					if (weight[id] < fullyCoverWeight) {
						WeightFinder[i].erase(weight[id]);
						A[i].Add(id, -k->first);
						weightedA[i].Add(id, -weight[id]*k->first);
						V[i] += -weight[id]*k->first*(L[1-i] - A[1-i].Sum(id+1, setSize));
						V[1-i] += k->first*weightedA[1-i].Sum(id-1);
						Gradient[i].erase(k);
					} else {
						if (lowestWeight > weight[k->second]) {
							lowestWeight = weight[k->second];
						}
					}
				}
			} 
		}
	}

	// return secondLowestWeight of secondGradient, which may be the same as lowestWeight and thus become useless
	// this algorithm is the same as that in Yildiz's paper
	double Remove2(double fullyCoverWeight) {
		map<double, int>::iterator k, kk;
		double delta;
		double lowestWeight = maxWeight;

		for (int i=0; i<2; i++) {
			if (!secondGradient[i].empty()) {
				if (secondGradient[1-i].empty()) {
					k = secondGradient[i].end();
					--k;
					kk = k;
					while (weight[k->second] < fullyCoverWeight && k != secondGradient[i].begin()) {
						--kk;
						delta = kk->first - k->first;
						secondA[i].Add(k->second, delta);
						secondWeightedA[i].Add(k->second, weight[k->second]*delta);
						secondV[i] += weight[k->second]*delta*L[1-i];
						secondGradient[i].erase(k);
						k = kk;
					}
					if (weight[k->second] < fullyCoverWeight) {
						delta = - k->first;
						secondA[i].Add(k->second, delta);
						secondWeightedA[i].Add(k->second, weight[k->second]*delta);
						secondV[i] += weight[k->second]*delta*L[1-i];
						secondGradient[i].erase(k);
					} else {
						if (lowestWeight > weight[k->second]) {
							lowestWeight = weight[k->second];
						}
					}
				} else {
					k = secondGradient[i].end();
					--k;
					kk = k;
					while (weight[k->second] < fullyCoverWeight && k != secondGradient[i].begin()) {
						--kk;
						delta = kk->first - k->first;
						secondA[i].Add(k->second, delta);
						secondWeightedA[i].Add(k->second, weight[k->second]*delta);
						secondV[i] += weight[k->second]*delta*(L[1-i] - secondA[1-i].Sum(k->second+1, setSize));
						secondV[1-i] -= delta*secondWeightedA[1-i].Sum(k->second-1);
						secondGradient[i].erase(k);
						k = kk;
					}
					if (weight[k->second] < fullyCoverWeight) {
						delta = - k->first;
						secondA[i].Add(k->second, delta);
						secondWeightedA[i].Add(k->second, weight[k->second]*delta);
						secondV[i] += weight[k->second]*delta*(L[1-i] - secondA[1-i].Sum(k->second+1, setSize));
						secondV[1-i] -= delta*secondWeightedA[1-i].Sum(k->second-1);
						secondGradient[i].erase(k);
					} else {
						if (lowestWeight > weight[k->second]) {
							lowestWeight = weight[k->second];
						}
					}
				}
			}
		}
		return lowestWeight;
	}
};

inline bool covers(const double* cub, const double regUp[]) {
	static int i;
	for (i=0; i<dimension-2; i++) {
		if (cub[i] < regUp[i]) {
			return false;
		}
	}
	return true;
}

inline bool partCovers(const double* cub, const double regLow[]) {
	static int i;
	for (i=0; i<dimension-2; i++) {
		if (cub[i] <= regLow[i]) {
			return false;
		}
	}
	return true;
}

inline int isPile(const double* cub, const double regUp[]) {
	static int pile;
	static int k;

	pile = dimension;
	// check all dimensions of the node
	for (k=0; k<dimension-2; k++) {
		// k-boundary of the node's region contained in the cuboid? 
		if (cub[k] < regUp[k]) {
			if (pile != dimension) {
				// second dimension occured that is not completely covered
				// ==> cuboid is no pile
				return -1;
			}
			pile = k;
		}
	}
	// if pile == this.dimension then
	// cuboid completely covers region
	// case is not possible since covering cuboids have been removed before

	// region in only one dimenison not completly covered 
	// ==> cuboid is a pile 
	return pile;
}

inline int containsBoundary(const double* cub, const double regUp[], const int split, vector<int> &order) {

	// condition only checked for split>0
	if (regUp[order[split]] <= cub[order[split]]){
		// boundary in dimension split not contained in region, thus
		// boundary is no candidate for the splitting line
		return -1;
	} 
	else {

		static int j;
		for (j=0; j<split+dimension-2-order.size(); j++) { // check boundaries
			if (regUp[order[j]] > cub[order[j]]) {
				// boundary contained in region
				return 1;
			}
		}

	}
	// no boundary contained in region
	return 0;
}

struct YildizTreeNode
{
	double *lowerbound, *upperbound;			// region
	double A;									// 2D volume dominated by children excluding second covering box and subgradients
	// as defined in original Yildiz's algorithm
	// used to calculate initial contribution when first covering box is inserted
	double dominatedA;							// 3D volume which is dominated by second covering box between two valid first covering box/gradients
	// used to calculate initial contribution when first covering box is inserted
	double secondA;								// 2D volume dominated by children including second covering box and subgradients
	// used to clear lazy when current first covering box is to be dominated
	double fullVolume;
	double lowestWeight;						// lowest weight of existing covering and partially covering box below this node
	double highestWeight;						// highest weight of existing covering and partially covering box of fathers
	double secondLowestWeight;					// dominated lowest weight of existing covering and partially covering box below this node
	double lazy;
	int firstCoveringID;						// heaviest box that fully covers this node
	double secondCoveringWeight;				// 2nd heaviest box that fully covers this node, its id is useless
	int partialCoverNumber;						// used for building tree
	int *partialCoverIndex;						// used for building tree	
	bool *boxType;
	vector<int> dims;								// used for building tree	
	YildizHC gradientStructure;					// maintain the gradient contributions	
	bool isNewLinkedList;
	vector<int*> LinkedList;
	YildizTreeNode *leftChild, *rightChild;		
};

inline void buildTree(YildizTreeNode *node);
inline void buildTree2(YildizTreeNode *node);
inline void fullyCoverUpdate(YildizTreeNode *node, double weight, double coverWeight, double lastCoordinate);
inline void fullyCoverUpdate2(YildizTreeNode *node, double weight);
inline void insertPoint(YildizTreeNode *node, int id, double lazy, double highestWeight);
inline void insertPoint2(YildizTreeNode *node, int id);
inline void updateA(YildizTreeNode *node, YildizTreeNode *child);
inline void updateContribution(YildizTreeNode *node);

inline void buildTree(YildizTreeNode *node) {

	//--- init --------------------------------------------------------------//

	unsigned int i, j;
	int count = 0, iterCount;
	double firstCoveringWeight = 0., secondCoveringWeight = 0.;

	int nonPiles = 0;
	int dimSize = node->dims.size();
	for (i = 0; i<node->partialCoverNumber; i++) {	
		if (node->boxType[i] && population[node->partialCoverIndex[i]][dimension-2] > secondCoveringWeight ||
			!node->boxType[i] && population[node->partialCoverIndex[i]][dimension-2] > firstCoveringWeight) {
			if (!covers(population[node->partialCoverIndex[i]], node->upperbound)) {
				node->partialCoverIndex[count] = node->partialCoverIndex[i];
				if (node->boxType[i] && population[node->partialCoverIndex[i]][dimension-2] > firstCoveringWeight) {
					node->boxType[count] = true;
				} else {
					node->boxType[count] = false;
				}
				if (isPile(population[node->partialCoverIndex[i]], node->upperbound) == -1) {
					nonPiles++;
				}
				count++;
				treeProjection[node->partialCoverIndex[i]] = 1;
			} else {
				if (population[node->partialCoverIndex[i]][dimension-2] > firstCoveringWeight) {
					secondCoveringWeight = firstCoveringWeight;
					firstCoveringWeight = population[node->partialCoverIndex[i]][dimension-2];
				} else {
					secondCoveringWeight = population[node->partialCoverIndex[i]][dimension-2];
				}
				treeProjection[node->partialCoverIndex[i]] = 0;
			}
		} else {
			treeProjection[node->partialCoverIndex[i]] = 0;
		}
	}
	
	if (nonPiles == 0) {
		
		int *newList;
		if (count == node->partialCoverNumber) {
			newList = node->LinkedList[dimSize];
		} else {
			iterCount = 0;
			newList = new int[count];
			for (j=0; j<node->partialCoverNumber; j++) {
				if (treeProjection[node->LinkedList[dimSize][j]]) {
					newList[iterCount] = node->LinkedList[dimSize][j];
					iterCount++;
				}
			}
		}
		
		node->gradientStructure.init(count);
		node->fullVolume = 1.0;
		for (i=0; i<dimension-2; i++) {
			node->gradientStructure.L[i] = node->upperbound[i] - node->lowerbound[i];
			node->fullVolume *= node->gradientStructure.L[i];
		}
		node->gradientStructure.type = vector<bool>(count+1);
		node->gradientStructure.weight = vector<double>(count+1);
		node->gradientStructure.coordinates = vector<double>(count+1);
		double old_weight = -1.;
		
		for (i=1; i<=count; i++) {
			node->gradientStructure.weight[i] = population[newList[i-1]][dimension-2];
			if (population[newList[i-1]][0] < node->upperbound[0]) {
				node->gradientStructure.type[i] = false;
				node->gradientStructure.coordinates[i] = min(node->gradientStructure.L[0], population[newList[i-1]][0] - node->lowerbound[0]);
			} else {
				node->gradientStructure.type[i] = true;
				node->gradientStructure.coordinates[i] = min(node->gradientStructure.L[1], population[newList[i-1]][1] - node->lowerbound[1]);
			}
		}

		node->A = 0.;
		node->secondA = 0.;
		node->dominatedA = 0.;
		node->lazy = 0.;
		node->firstCoveringID = -1;
		node->secondCoveringWeight = 0.;
		node->highestWeight = 0.;
		node->lowestWeight = maxWeight;
		node->secondLowestWeight = maxWeight;
		node->leftChild = NULL;
		node->rightChild = NULL;
		// reset partialCoverIndex and partialCoverNumber for leaf node
		// this will be used in leafContribution
		node->partialCoverNumber = count;
		for (i=0; i<count; i++) {
			node->partialCoverIndex[i] = newList[i];
		}
		if (count != node->partialCoverNumber) {
			delete [] newList;
		}
		if (node->isNewLinkedList) {
			for (i=0; i<dimSize+1; i++) {
				delete [] node->LinkedList[i];
			}
		}
		return;
	}
	
	if (nonPiles < dSqrtDataNumber) {
			vector<int*> newList;
			if (count == node->partialCoverNumber) {
				newList = node->LinkedList;
			} else {
				newList = vector<int*>(dimension-1);
				for (i=0; i<dimension-1; i++) {
					iterCount = 0;
					newList[i] = new int[count];
					for (j=0; j<node->partialCoverNumber; j++) {
						if (treeProjection[node->LinkedList[i][j]]) {
							newList[i][iterCount] = node->LinkedList[i][j];
							iterCount++;
						}
					}
				}
				if (node->isNewLinkedList) {
					for (i=0; i<dimension-1; i++) {
						delete [] node->LinkedList[i];
					}
				}
			}
			node->partialCoverNumber = count;
			vector<int*> rightLinkedList = vector<int*>(dimension-1);
			for (i=0; i<dimension-2; i++) {
				rightLinkedList[node->dims[i]] = newList[i];
			}
			rightLinkedList[dimension-2] = newList[dimension-2];
			node->LinkedList = rightLinkedList;	
			buildTree2(node);
		return;
	}
	
	vector<int*> newList;
	if (count == node->partialCoverNumber) {
		newList = node->LinkedList;
	} else {
		newList = vector<int*>(dimSize+1);
		for (i=0; i<dimSize+1; i++) {
			iterCount = 0;
			newList[i] = new int[count];
			for (j=0; j<node->partialCoverNumber; j++) {
				if (treeProjection[node->LinkedList[i][j]]) {
					newList[i][iterCount] = node->LinkedList[i][j];
					iterCount++;
				}
			}
		}
	}
	
	int split = 0;
	int middleIndex = -1;
	int boundSize = 0, noBoundSize = 0;

	do {
		for (i=0; i<count; i++) {
			int contained = containsBoundary(population[newList[split][i]], node->upperbound, split, node->dims);
			if (contained == 1) {
				boundaries[boundSize] = newList[split][i];
				boundSize++;
			} else if (contained == 0) {
				noBoundaries[noBoundSize] = newList[split][i];
				noBoundSize++;
			}
		}

		if (boundSize > 0) {
			middleIndex = boundaries[boundSize/2];
		}
		else if (noBoundSize > dSqrtDataNumber) {
			middleIndex = noBoundaries[noBoundSize/2];
		}
		else {
			split++;
			noBoundSize = 0;
		}
		
	} while (middleIndex == -1);
	
	// left child
	// reduce maxPoint
	YildizTreeNode *leftChild = new YildizTreeNode();
	leftChild->partialCoverNumber = count;
	leftChild->partialCoverIndex = new int[count];
	leftChild->boxType = new bool[count];
	leftChild->lowerbound = new double[dimension-2];
	leftChild->upperbound = new double[dimension-2];
	for (i=0; i<dimension-2; i++) {
		leftChild->lowerbound[i] = node->lowerbound[i];
		leftChild->upperbound[i] = node->upperbound[i];
	}
	int trueDimension = node->dims[split];
	leftChild->upperbound[trueDimension] = population[middleIndex][trueDimension];
	leftChild->LinkedList = vector<int*>(dimSize + 1 - split);
	for (i=0; i<dimSize + 1 - split; i++) {
		leftChild->LinkedList[i] = newList[i + split];
	}
	if (dimSize > 1) {
		if (split + 1 < dimSize) {
			// alter coordinate
			leftChild->dims = vector<int>(2);
			leftChild->dims[0] = node->dims[1];
			leftChild->dims[1] = node->dims[0];
			int *temp = leftChild->LinkedList[0];
			leftChild->LinkedList[0] = leftChild->LinkedList[1];
			leftChild->LinkedList[1] = temp;
		} else {
			// only one coordinate
			leftChild->dims = vector<int>(1);
			leftChild->dims[0] = node->dims[dimSize - 1];
		}
	} else {
		// only one coordinate
		leftChild->dims = node->dims;
	}
	for (i=0; i<count; i++) {
		leftChild->partialCoverIndex[i] = node->partialCoverIndex[i];
		leftChild->boxType[i] = node->boxType[i];
	}	
	if (count != node->partialCoverNumber) {
		leftChild->isNewLinkedList = true;
		for (i=0; i<split; i++) {
			delete [] newList[i];
		}
		if (node->isNewLinkedList) {
			for (i=0; i<dimSize+1; i++) {
				delete [] node->LinkedList[i];
			}
		}
	} else {
		leftChild->isNewLinkedList = false;
	}
	// right child
	// increase minPoint
	YildizTreeNode *rightChild = new YildizTreeNode();
	rightChild->partialCoverNumber = 0;
	rightChild->partialCoverIndex = new int[count];
	rightChild->boxType = new bool[count];
	for (i=0; i<count; i++) {
		if (population[node->partialCoverIndex[i]][trueDimension] > population[middleIndex][trueDimension]) {	
			rightChild->partialCoverIndex[rightChild->partialCoverNumber] = node->partialCoverIndex[i];
			rightChild->boxType[rightChild->partialCoverNumber] = node->boxType[i];
			rightChild->partialCoverNumber++;
		} else {
			treeProjection[node->partialCoverIndex[i]] = 0;
		}
	}
	if (rightChild->partialCoverNumber > 0) {
		rightChild->dims = leftChild->dims;
		rightChild->lowerbound = new double[dimension-2];
		rightChild->upperbound = new double[dimension-2];
		for (i=0; i<dimension-2; i++) {
			rightChild->lowerbound[i] = node->lowerbound[i];
			rightChild->upperbound[i] = node->upperbound[i];
		}
		rightChild->lowerbound[trueDimension] = population[middleIndex][trueDimension];
		rightChild->isNewLinkedList = true;
		rightChild->LinkedList = vector<int*>(dimSize + 1 - split);
		for (i=0; i<dimSize + 1 - split; i++) {
			iterCount = 0;
			rightChild->LinkedList[i] = new int[rightChild->partialCoverNumber];
			for (j=0; j<count; j++) {
				if (treeProjection[leftChild->LinkedList[i][j]]) {
					rightChild->LinkedList[i][iterCount] = leftChild->LinkedList[i][j];
					iterCount++;
				} 
			}
		}
		buildTree(leftChild);
		if (node->isNewLinkedList && !leftChild->isNewLinkedList) {
			for (i=0; i<dimSize+1; i++) {
				delete [] node->LinkedList[i];
			}
		}
		buildTree(rightChild);
	} else {
		delete [] rightChild->partialCoverIndex;
		delete [] rightChild->boxType;
		rightChild->gradientStructure.A[1].BITArray = NULL;
		rightChild->gradientStructure.weightedA[0].BITArray = NULL;
		rightChild->gradientStructure.weightedA[1].BITArray = NULL;
		rightChild->gradientStructure.secondA[0].BITArray = NULL;
		rightChild->gradientStructure.secondA[1].BITArray = NULL;
		rightChild->gradientStructure.secondWeightedA[0].BITArray = NULL;
		rightChild->gradientStructure.secondWeightedA[1].BITArray = NULL;
		rightChild->gradientStructure.weightedSubA[0].BITArray = NULL;
		rightChild->gradientStructure.weightedSubA[1].BITArray = NULL;
		rightChild->gradientStructure.dominated[0].BITArray = NULL;
		rightChild->gradientStructure.dominated[1].BITArray = NULL;
		delete rightChild;
		rightChild = NULL;
		buildTree(leftChild);
		if (node->isNewLinkedList && !leftChild->isNewLinkedList) {
			for (i=0; i<dimSize+1; i++) {
				delete [] node->LinkedList[i];
			}
		}
	}

	node->fullVolume = 1.0;
	for (i=0; i<dimension-2; i++) {
		node->fullVolume *= node->upperbound[i] - node->lowerbound[i];
	}
	node->A = 0.;
	node->secondA = 0.;
	node->dominatedA = 0.;
	node->lazy = 0.;
	node->firstCoveringID = -1;
	node->secondCoveringWeight = 0.;
	node->highestWeight = 0.;
	node->lowestWeight = maxWeight;
	node->secondLowestWeight = maxWeight;
	node->leftChild = leftChild;
	node->rightChild = rightChild;
	delete [] node->partialCoverIndex;
	// partialCoverIndex and partialCoverNumber is set zero for internal nodes
	node->partialCoverIndex = NULL;
	node->partialCoverNumber = 0;
}

inline void buildTree2(YildizTreeNode *node) {
	
	int i, j;
	int count = 0;
	int nonPiles = 0;
	double firstCoveringWeight = 0., secondCoveringWeight = 0.;
	for (i = 0; i<node->partialCoverNumber; i++) {
		if (node->boxType[i] && population[node->partialCoverIndex[i]][dimension-2] > secondCoveringWeight ||
			!node->boxType[i] && population[node->partialCoverIndex[i]][dimension-2] > firstCoveringWeight) {
			if (!covers(population[node->partialCoverIndex[i]], node->upperbound)) {
				if (node->boxType[i] && population[node->partialCoverIndex[i]][dimension-2] > firstCoveringWeight) {
					node->boxType[count] = true;
				} else {
					node->boxType[count] = false;
				}
				piles[node->partialCoverIndex[i]] = isPile(population[node->partialCoverIndex[i]], node->upperbound);
				if (piles[node->partialCoverIndex[i]] == -1) {
					nonPiles++; 
				} 
				node->partialCoverIndex[count] = node->partialCoverIndex[i];
				count++;
				treeProjection[node->partialCoverIndex[i]] = 1;
			} else {
				if (population[node->partialCoverIndex[i]][dimension-2] > firstCoveringWeight) {
					secondCoveringWeight = firstCoveringWeight;
					firstCoveringWeight = population[node->partialCoverIndex[i]][dimension-2];
				} else {
					secondCoveringWeight = population[node->partialCoverIndex[i]][dimension-2];
				}
				treeProjection[node->partialCoverIndex[i]] = 0;
			}
		} else {			
			treeProjection[node->partialCoverIndex[i]] = 0;
		}
	}
	
	if (nonPiles == 0) {
		
		int *newList;
		if (count == node->partialCoverNumber) {
			newList = node->LinkedList[dimension-2];
		} else {
			int iterCount = 0;
			newList = new int[count];
			for (j=0; j<node->partialCoverNumber; j++) {
				if (treeProjection[node->LinkedList[dimension-2][j]]) {
					newList[iterCount] = node->LinkedList[dimension-2][j];
					iterCount++;
				}
			}
		}
	
		node->gradientStructure.init(count);
		node->fullVolume = 1.0;
		for (i=0; i<dimension-2; i++) {
			node->gradientStructure.L[i] = node->upperbound[i] - node->lowerbound[i];
			node->fullVolume *= node->gradientStructure.L[i];
		}
		node->gradientStructure.type = vector<bool>(count+1);
		node->gradientStructure.weight = vector<double>(count+1);
		node->gradientStructure.coordinates = vector<double>(count+1);

		for (i=1; i<=count; i++) {
			node->gradientStructure.weight[i] = population[newList[i-1]][dimension-2];
			if (population[newList[i-1]][0] < node->upperbound[0]) {
				node->gradientStructure.type[i] = false;
				node->gradientStructure.coordinates[i] = min(node->gradientStructure.L[0], population[newList[i-1]][0] - node->lowerbound[0]);
			} else {
				node->gradientStructure.type[i] = true;
				node->gradientStructure.coordinates[i] = min(node->gradientStructure.L[1], population[newList[i-1]][1] - node->lowerbound[1]);
			}
		}

		node->A = 0.;
		node->secondA = 0.;
		node->dominatedA = 0.;
		node->lazy = 0.;
		node->firstCoveringID = -1;
		node->secondCoveringWeight = 0.;
		node->highestWeight = 0.;
		node->lowestWeight = maxWeight;
		node->secondLowestWeight = maxWeight;
		node->leftChild = NULL;
		node->rightChild = NULL;
		node->partialCoverNumber = count;
		for (i=0; i<count; i++) {
			node->partialCoverIndex[i] = newList[i];
		}
		if (count != node->partialCoverNumber) {
			delete [] newList;
		}
		if (node->isNewLinkedList) {
			for (i=0; i<dimension-1; i++) {
				delete [] node->LinkedList[i];
			}
		}
		return;
	}
		
	YildizTreeNode *leftChild = new YildizTreeNode();

	int iterCount, nonPileCount;
	static vector<int> Ids;
	Ids = vector<int>(dimension-2);	
	if (count == node->partialCoverNumber) {
		leftChild->LinkedList = node->LinkedList;
		for (i=0; i<dimension-2; i++) {
			nonPileCount = 0;
			for (j=0; j<node->partialCoverNumber; j++) {
				if (piles[leftChild->LinkedList[i][j]] == -1) {
					nonPileCount++;
					if (nonPileCount>=nonPiles/2) {
						Ids[i] = j;
						break;
					}
				}
			}
		}
	} else {
		leftChild->LinkedList = vector<int*>(dimension-1);
		for (i=0; i<dimension-1; i++) {
			iterCount = 0;
			leftChild->LinkedList[i] = new int[count];
			for (j=0; j<node->partialCoverNumber; j++) {
				if (treeProjection[node->LinkedList[i][j]]) {
					leftChild->LinkedList[i][iterCount] = node->LinkedList[i][j];
					iterCount++;
				}
			}
		}
		for (i=0; i<dimension-2; i++) {
			nonPileCount = 0;
			for (j=0; j<count; j++) {
				if (piles[leftChild->LinkedList[i][j]] == -1) {
					nonPileCount++;
					if (nonPileCount>=nonPiles/2) {
						Ids[i] = j;
						break;
					}
				}
			}
		}
	}
	int split;
	if (Ids[0] == Ids[1]) {
		split = alter;
		alter = 1 - alter;
	} else if (Ids[0] > Ids[1]) {
		split = 0;
	} else {
		split = 1;
	}
	int select = leftChild->LinkedList[split][Ids[split]];
	
	leftChild->partialCoverNumber = count;
	leftChild->partialCoverIndex = new int[count];
	leftChild->boxType = new bool[count];
	leftChild->lowerbound = new double[dimension-2];
	leftChild->upperbound = new double[dimension-2];
	for (i=0; i<dimension-2; i++) {
		leftChild->lowerbound[i] = node->lowerbound[i];
		leftChild->upperbound[i] = node->upperbound[i];
	}
	leftChild->upperbound[split] = population[select][split];
	for (i=0; i<count; i++) {
		leftChild->partialCoverIndex[i] = node->partialCoverIndex[i];
		leftChild->boxType[i] = node->boxType[i];
	}	
	if (count != node->partialCoverNumber) {
		leftChild->isNewLinkedList = true;
		if (node->isNewLinkedList) {
			for (i=0; i<dimension-1; i++) {
				delete [] node->LinkedList[i];
			}
		}
	} else {
		leftChild->isNewLinkedList = false;
	}
	YildizTreeNode *rightChild = new YildizTreeNode();
	rightChild->partialCoverNumber = 0;
	rightChild->partialCoverIndex = new int[count];
	rightChild->boxType = new bool[count];
	for (i=0; i<count; i++) {
		if (population[node->partialCoverIndex[i]][split] > population[select][split]) {	
			rightChild->partialCoverIndex[rightChild->partialCoverNumber] = node->partialCoverIndex[i];
			rightChild->boxType[rightChild->partialCoverNumber] = node->boxType[i];
			rightChild->partialCoverNumber++;
		} else {
			treeProjection[node->partialCoverIndex[i]] = 0;
		}
	}
	if (rightChild->partialCoverNumber > 0) {
		rightChild->lowerbound = new double[dimension-2];
		rightChild->upperbound = new double[dimension-2];
		for (i=0; i<dimension-2; i++) {
			rightChild->lowerbound[i] = node->lowerbound[i];
			rightChild->upperbound[i] = node->upperbound[i];
		}
		rightChild->lowerbound[split] = population[select][split];
		rightChild->isNewLinkedList = true;
		rightChild->LinkedList = vector<int*>(dimension-1);
		for (i=0; i<dimension-1; i++) {
			iterCount = 0;
			rightChild->LinkedList[i] = new int[rightChild->partialCoverNumber];
			for (j=0; j<count; j++) {
				if (treeProjection[leftChild->LinkedList[i][j]]) {
					rightChild->LinkedList[i][iterCount] = leftChild->LinkedList[i][j];
					iterCount++;
				} 
			}
		}
		buildTree2(leftChild);
		if (node->isNewLinkedList && !leftChild->isNewLinkedList) {
			for (i=0; i<dimension-1; i++) {
				delete [] node->LinkedList[i];
			}
		}
		buildTree2(rightChild);
	} else {
		delete [] rightChild->partialCoverIndex;
		delete [] rightChild->boxType;
		rightChild->gradientStructure.A[1].BITArray = NULL;
		rightChild->gradientStructure.weightedA[0].BITArray = NULL;
		rightChild->gradientStructure.weightedA[1].BITArray = NULL;
		rightChild->gradientStructure.secondA[0].BITArray = NULL;
		rightChild->gradientStructure.secondA[1].BITArray = NULL;
		rightChild->gradientStructure.secondWeightedA[0].BITArray = NULL;
		rightChild->gradientStructure.secondWeightedA[1].BITArray = NULL;
		rightChild->gradientStructure.weightedSubA[0].BITArray = NULL;
		rightChild->gradientStructure.weightedSubA[1].BITArray = NULL;
		rightChild->gradientStructure.dominated[0].BITArray = NULL;
		rightChild->gradientStructure.dominated[1].BITArray = NULL;
		delete rightChild;
		rightChild = NULL;
		buildTree2(leftChild);
		if (node->isNewLinkedList && !leftChild->isNewLinkedList) {
			for (i=0; i<dimension-1; i++) {
				delete [] node->LinkedList[i];
			}
		}
	}

	node->fullVolume = 1.0;
	for (i=0; i<dimension-2; i++) {
		node->fullVolume *= node->upperbound[i] - node->lowerbound[i];
	}
	node->A = 0.;
	node->secondA = 0.;
	node->dominatedA = 0.;
	node->lazy = 0.;
	node->firstCoveringID = -1;
	node->secondCoveringWeight = 0.;
	node->highestWeight = 0.;
	node->lowestWeight = maxWeight;
	node->secondLowestWeight = maxWeight;
	node->leftChild = leftChild;
	node->rightChild = rightChild;
	delete [] node->partialCoverIndex;
	// partialCoverIndex and partialCoverNumber is set zero for internal nodes
	node->partialCoverIndex = NULL;
	node->partialCoverNumber = 0;
}

inline int findID(vector<double> &weights, double weight) {
	// binary search, index of weights starts from 1
	int l = 1, mid, r = weights.size();
	while (l <= r) {
		mid = (l + r)/2;
		if (weight == weights[mid]) {
			return mid;
		}
		if (weight < weights[mid]) {
			r = mid - 1;
		} else {
			l = mid + 1;
		}
	}
	return -1;
}

inline void updateA(YildizTreeNode *node, YildizTreeNode *child) {
	if (child->firstCoveringID != -1) {
		node->A += child->fullVolume;
		node->lowestWeight = min(node->lowestWeight, population[child->firstCoveringID][dimension-2]);
		//node->dominatedA += 0.;
		node->secondA += child->fullVolume;
		if (child->secondCoveringWeight > 0) {
			node->secondLowestWeight = min(node->secondLowestWeight, child->secondCoveringWeight);
		} else {
			node->secondLowestWeight = min(node->secondLowestWeight, child->secondLowestWeight);
		}
	} else {
		node->A += child->A;
		node->lowestWeight = min(node->lowestWeight, child->lowestWeight);
		// in this case we must have child->highestWeight > child->secondCoveringWeight
		// otherwise child->secondCoveringBox would be valid and become first covering box
		if (child->secondCoveringWeight > 0) {
			node->dominatedA += child->dominatedA + 
				child->secondCoveringWeight * (child->fullVolume - child->secondA);
			node->secondA += child->fullVolume;
			node->secondLowestWeight = min(node->secondLowestWeight, child->secondCoveringWeight);
		} else {
			node->secondA += child->secondA;
			node->dominatedA += child->dominatedA;
			node->secondLowestWeight = min(node->secondLowestWeight, child->secondLowestWeight);
		}
	}
}

// firstly, weight > node->lowestWeight/secondLowestWeight, otherwise the recursive call would have ended by updating lazy at fathers
// secondly, weight is not dominated by first covering box of this node and fathers, otherwise we would use fullyCoverUpdate2
// so weight > max(node->highestWeight, node->firstCoveringWeight, node->secondCoveringWeight)
// as a result, some child's contribution needs to be updated and thus lazy is needed
// all boxes of children with weight between node->highestWeight and population[node->firstCoveringID][dimension-2] will be removed
// all boxes of children with weight between population[node->firstCoveringID][dimension-2] and population[id][dimension-2] will be dominated once

inline void fullyCoverUpdate(YildizTreeNode *node, double weight, double coverWeight, double lastCoordinate) {

	if (node->leftChild == NULL && node->rightChild == NULL) {
		// no first covering box exists or first covering box's contribution has been updated outside function call
		// we only need to update gradients' contributions and node's variables
		int i;
		double temp;
		// secondLowestWeight here is determined by gradients dominated by other gradients and covering box
		// secondLowestWeight for leaf node should also consider second covering box, but here we do not consider it
		// it will be considered out of the function call after second covering box is updated
		// fullyCoverUpdate2() also uses the same strategy

		// remove gradients dominated by covering box
		// gradients/subgradients dominated by weight will be useless in first part of gradientStructure and be removed
		// other parameters in function call are used to update contributions of old lightest gradient and new lightest gradient
		// lazy affects lightest gradients in both Gradient[0] and Gradient[1]
		if (node->firstCoveringID != -1) {	
			node->gradientStructure.Remove(weight, population[node->firstCoveringID][dimension-2], lastCoordinate, node->lazy, 
				node->lowestWeight, node->secondLowestWeight);
		} else {
			node->gradientStructure.Remove(weight, max(node->secondCoveringWeight, coverWeight), lastCoordinate, node->lazy, 
				node->lowestWeight, node->secondLowestWeight);
		} 
		// remove gradients dominated twice by covering box, update second part of gradientStructure
		double secondLowestWeight2;
		if (node->firstCoveringID == -1) {
			secondLowestWeight2 = node->gradientStructure.Remove2(coverWeight);
		} else {
			secondLowestWeight2 = node->gradientStructure.Remove2(population[node->firstCoveringID][dimension-2]);
		}
		if (secondLowestWeight2 != node->lowestWeight) {
			node->secondLowestWeight = secondLowestWeight2;
		}
		temp = 1.0;
		for (i=0; i<dimension-2; i++) {
			if (node->gradientStructure.Gradient[i].crbegin() != node->gradientStructure.Gradient[i].crend()) {
				temp *= (node->gradientStructure.L[i] - node->gradientStructure.Gradient[i].crbegin()->first);
			} else {
				temp *= node->gradientStructure.L[i];
			}
		}
		node->A = node->fullVolume - temp;
		temp = 1.0;
		for (i=0; i<dimension-2; i++) {
			if (node->gradientStructure.secondGradient[i].crbegin() != node->gradientStructure.secondGradient[i].crend()) {
				temp *= (node->gradientStructure.L[i] - node->gradientStructure.secondGradient[i].crbegin()->first);
			} else {
				temp *= node->gradientStructure.L[i];
			}
		}
		node->secondA = node->fullVolume - temp;
		node->dominatedA = node->gradientStructure.secondV[0] + node->gradientStructure.secondV[1] - 
			node->gradientStructure.V[0] - node->gradientStructure.V[1];
	} else {
		// only update if weight > child->secondCoveringWeight
		if (weight > node->leftChild->secondCoveringWeight) {
			if (node->leftChild->firstCoveringID != -1) {
				// new lazy is useless if child has first covering box
				if (node->leftChild->secondCoveringWeight == 0.) {
					contributions[node->leftChild->firstCoveringID] -= node->lazy * (node->leftChild->fullVolume - node->leftChild->secondA) + 
						lastCoordinate * 
						( population[node->leftChild->firstCoveringID][dimension-2] * (node->leftChild->fullVolume - node->leftChild->A) - 
						coverWeight * (node->leftChild->fullVolume - node->leftChild->secondA) - 
						node->leftChild->dominatedA ); 
				} else {
					contributions[node->leftChild->firstCoveringID] -= lastCoordinate * 
						( population[node->leftChild->firstCoveringID][dimension-2] * (node->leftChild->fullVolume - node->leftChild->A) -
						node->leftChild->secondCoveringWeight * (node->leftChild->fullVolume - node->leftChild->secondA) - 
						node->leftChild->dominatedA ); 
				}
				if (weight > population[node->leftChild->firstCoveringID][dimension-2]) {
					if (weight > node->leftChild->lowestWeight || weight > node->leftChild->secondLowestWeight) { 
						fullyCoverUpdate(node->leftChild, weight, population[node->leftChild->firstCoveringID][dimension-2], lastCoordinate);	
						node->leftChild->lazy = 0.;
					} else {
						node->leftChild->lazy += (weight - population[node->leftChild->firstCoveringID][dimension-2]) * lastCoordinate;
					}
					node->leftChild->secondCoveringWeight = population[node->leftChild->firstCoveringID][dimension-2];
					if (node->leftChild->leftChild == NULL && node->leftChild->rightChild == NULL) {
						node->leftChild->secondLowestWeight = node->leftChild->secondCoveringWeight;
					}
					node->leftChild->firstCoveringID = -1;
				} else {
					if (weight > node->leftChild->lowestWeight || weight > node->leftChild->secondLowestWeight) { 
						// since weight < child->firstCoveringWeight
						// this is similar to updating second covering box of a node whose first covering box is valid
						// so we call fullyCoverUpdate2
						fullyCoverUpdate2(node->leftChild, weight);
					} 
					// here we do not need to code for leaf node's secondLowestWeight as above
					// leaf node's secondLowestWeight contains the information of secondCoveringWeight
					// therefore if it should be updated, it would have been updated in fullyCoverUpdate2()
					// note that fullyCoverUpdate2() do not consider secondCoveringWeight by default
					// which is equivalent to considering secondCoveringWeight == 0
					node->leftChild->secondCoveringWeight = 0.;
					contributions[node->leftChild->firstCoveringID] += lastCoordinate * 
						( population[node->leftChild->firstCoveringID][dimension-2] * (node->leftChild->fullVolume - node->leftChild->A) -
						weight * (node->leftChild->fullVolume - node->leftChild->secondA) - 
						node->leftChild->dominatedA ); 
				} 
			} else {
				// pass down lazy
				node->leftChild->lazy += node->lazy;
				if (weight > node->leftChild->lowestWeight || weight > node->leftChild->secondLowestWeight) { 
					fullyCoverUpdate(node->leftChild, weight, coverWeight, lastCoordinate);
					node->leftChild->lazy = 0.;
				} else {
					// new lazy is useful if the recursive call ended
					node->leftChild->lazy += (weight - coverWeight) * lastCoordinate;
				}
				node->leftChild->secondCoveringWeight = 0.;
			}
		} 
		node->A = 0.;
		node->lowestWeight = maxWeight;
		node->secondA = 0.;
		node->dominatedA = 0.;
		node->secondLowestWeight = maxWeight;
		node->leftChild->highestWeight = weight;
		updateA(node, node->leftChild);

		if (node->rightChild != NULL) {
			if (node->rightChild->firstCoveringID != -1) {
				if (node->rightChild->secondCoveringWeight == 0.) {
					contributions[node->rightChild->firstCoveringID] -= node->lazy * (node->rightChild->fullVolume - node->rightChild->secondA) + 
						lastCoordinate * 
						( population[node->rightChild->firstCoveringID][dimension-2] * (node->rightChild->fullVolume - node->rightChild->A) - 
						coverWeight * (node->rightChild->fullVolume - node->rightChild->secondA) - 
						node->rightChild->dominatedA ); 
				} else {
					contributions[node->rightChild->firstCoveringID] -= lastCoordinate * 
						( population[node->rightChild->firstCoveringID][dimension-2] * (node->rightChild->fullVolume - node->rightChild->A) -
						node->rightChild->secondCoveringWeight * (node->rightChild->fullVolume - node->rightChild->secondA) - 
						node->rightChild->dominatedA ); 
				}
				if (weight > population[node->rightChild->firstCoveringID][dimension-2]) {
					if (weight > node->rightChild->lowestWeight || weight > node->rightChild->secondLowestWeight) { 
						fullyCoverUpdate(node->rightChild, weight, population[node->rightChild->firstCoveringID][dimension-2], lastCoordinate);	
						node->rightChild->lazy = 0.;
					} else {
						node->rightChild->lazy += (weight - population[node->rightChild->firstCoveringID][dimension-2]) * lastCoordinate;
					}
					node->rightChild->secondCoveringWeight = population[node->rightChild->firstCoveringID][dimension-2];
					if (node->rightChild->leftChild == NULL && node->rightChild->rightChild == NULL) {
						node->rightChild->secondLowestWeight = node->rightChild->secondCoveringWeight;
					}
					node->rightChild->firstCoveringID = -1;
				} else {
					if (weight > node->rightChild->lowestWeight || weight > node->rightChild->secondLowestWeight) { 
						fullyCoverUpdate2(node->rightChild, weight);
					} 
					node->rightChild->secondCoveringWeight = 0.;
					contributions[node->rightChild->firstCoveringID] += lastCoordinate * 
						( population[node->rightChild->firstCoveringID][dimension-2] * (node->rightChild->fullVolume - node->rightChild->A) -
						weight * (node->rightChild->fullVolume - node->rightChild->secondA) - 
						node->rightChild->dominatedA ); 
				} 
			} else {
				node->rightChild->lazy += node->lazy;
				if (weight > node->rightChild->lowestWeight || weight > node->rightChild->secondLowestWeight) { 
					fullyCoverUpdate(node->rightChild, weight, coverWeight, lastCoordinate);
					node->rightChild->lazy = 0.;
				} else {
					node->rightChild->lazy += (weight - coverWeight) * lastCoordinate;
				}
				node->rightChild->secondCoveringWeight = 0.;
			}
			node->rightChild->highestWeight = weight;
			updateA(node, node->rightChild);
		}
	}
}

// we have node->lowestWeight > max(node->firstCoveringWeight, node->highestWeight) > weight > node->secondLowestWeight
// the recursive call will end when 
// 1) at a child whose first covering box is valid
// weight < node->highestWeight, so weight will not update this child, because node->highestWeight has updated it
// weight will be also smaller than child->secondLowestWeight, because child->secondLowestWeight must be no smaller than node->highestWeight
// otherwise it would have been dominated by node->highestWeight and child's first covering box
// as a result, weight will not affect any box of and under the child, nor affect any contributions of first covering boxes of children
// 2) at a leaf whose first covering box is -1
// 
// therefore only subgradients in gradientStructure2 need to be updated and no contribution of covering box/gradient will change
//
// in both cases, lazy is not needed
// node->secondA and node->dominatedA will change because weight > node->secondLowestWeight
inline void fullyCoverUpdate2(YildizTreeNode *node, double weight) {

	if (node->leftChild == NULL && node->rightChild == NULL) {
		// no first covering box exists or first covering box's contribution has been updated outside function call
		// we only need to update gradients' contributions and node's variables
		int i;
		double temp;
		// remove gradients dominated twice by covering box
		node->secondLowestWeight = node->gradientStructure.Remove2(weight);
		if (node->secondLowestWeight == node->lowestWeight) {
			node->secondLowestWeight = maxWeight;
		}
		temp = 1.0;
		for (i=0; i<dimension-2; i++) {
			if (node->gradientStructure.secondGradient[i].crbegin() != node->gradientStructure.secondGradient[i].crend()) {
				temp *= (node->gradientStructure.L[i] - node->gradientStructure.secondGradient[i].crbegin()->first);
			} else {
				temp *= node->gradientStructure.L[i];
			}
			if (!node->gradientStructure.SubWeightFinder[i].empty() &&
				node->gradientStructure.SubWeightFinder[i].begin()->first < node->secondLowestWeight) 
			{
				node->secondLowestWeight = node->gradientStructure.SubWeightFinder[i].begin()->first;
			}
		}
		node->secondA = node->fullVolume - temp;
		node->dominatedA = node->gradientStructure.secondV[0] + node->gradientStructure.secondV[1] - 
			node->gradientStructure.V[0] - node->gradientStructure.V[1];
	} else {
		// internal node
		if (weight > node->leftChild->secondCoveringWeight) {
			node->leftChild->secondCoveringWeight = 0.;
			if (node->leftChild->firstCoveringID == -1) {
				if (weight > node->leftChild->secondLowestWeight) {
					fullyCoverUpdate2(node->leftChild, weight);
				} 
				node->secondA = node->leftChild->secondA;
				node->secondLowestWeight = node->leftChild->secondLowestWeight;
				node->dominatedA = node->leftChild->dominatedA;
			} else {
				node->secondA = node->leftChild->fullVolume;
				node->secondLowestWeight = node->leftChild->secondLowestWeight;
				node->dominatedA = 0.;
			}
		} else {
			node->secondA = node->leftChild->fullVolume;
			node->secondLowestWeight = node->leftChild->secondCoveringWeight;
			if (node->leftChild->firstCoveringID == -1) {
				node->dominatedA = node->leftChild->dominatedA + 
					node->leftChild->secondCoveringWeight * (node->leftChild->fullVolume - node->leftChild->secondA);
			} else {
				node->dominatedA = 0.;
			}
		}
		if (node->rightChild != NULL) {
			if (weight > node->rightChild->secondCoveringWeight) {
				node->rightChild->secondCoveringWeight = 0.;
				if (node->rightChild->firstCoveringID == -1) {
					if (weight > node->rightChild->secondLowestWeight) {
						fullyCoverUpdate2(node->rightChild, weight);
					} 
					node->secondA += node->rightChild->secondA;
					node->secondLowestWeight = min(node->secondLowestWeight, node->rightChild->secondLowestWeight);
					node->dominatedA += node->rightChild->dominatedA;
				} else {
					node->secondA += node->rightChild->fullVolume;
					node->secondLowestWeight = min(node->secondLowestWeight, node->rightChild->secondLowestWeight);
					//node->dominatedA += 0.;
				}
			} else {
				node->secondA += node->rightChild->fullVolume;
				node->secondLowestWeight = min(node->secondLowestWeight, node->rightChild->secondCoveringWeight);
				if (node->rightChild->firstCoveringID == -1) {
					node->dominatedA += node->rightChild->dominatedA + 
						node->rightChild->secondCoveringWeight * (node->rightChild->fullVolume - node->rightChild->secondA);
				} else {
					//node->dominatedA += 0.;
				}
			}
		}
	}
}

// each box will be visited at most 4 times = O(1) times (firstCoveringBox/SubGradient and secondCoveringBox/Gradient for insertion/deletion)
// pass down lazy labels only if 
// 1) id is not dominated by any fully covering box of father and this node, and
// 2a) id is a first covering box of this node and weight[id] > lowestWeight/secondLowestWeight, or
// 2b) id is a partially covering box of this node
// 2a) or 2b) also leads to recursive call in orignial Yildiz's algorithm
// so each operation of passing down lazy charges to each insertion/deletion operation, so the complexity does not increase
inline void insertPoint(YildizTreeNode *node, int id, double lazy, double highestWeight) {
	// update if current box is not dominated by second covering box of this node
	// different A's must change
	
	if (population[id][dimension-2] > node->secondCoveringWeight) {
		if (node->firstCoveringID == -1 || population[id][dimension-2] > population[node->firstCoveringID][dimension-2]) {
			// id is not dominated by any fully covering box of fathers and this node
			if ( covers(population[id], node->upperbound) ) {
				// new first covering box
				if (node->firstCoveringID != -1) {
					// update contribution of current first covering box
					// in this case, node->highestWeight < population[node->firstCoveringID][dimension-2] < population[id][dimension-2]
					if (node->secondCoveringWeight == 0.) {
						// use lazy to update contributions
						// lazy is useless if node->firstCoveringID <> -1 and node->secondCoveringWeight > 0 
						// otherwise second covering box would have been dominated by two boxes before id is inserted
						// the second term is currently dominating area of node->firstCoveringID excluding lazy
						// it is easy to find that lazy must be defined by node->highestWeight
						contributions[node->firstCoveringID] -= lazy * (node->fullVolume - node->secondA) + population[id][dimension-1] * 
							( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
							//node->highestWeight * (node->fullVolume - node->secondA) - 
							highestWeight * (node->fullVolume - node->secondA) - 
							node->dominatedA ); 
					} else {
						contributions[node->firstCoveringID] -= population[id][dimension-1] * 
							( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
							node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
							node->dominatedA ); 
					}
					//getchar();
					// update children
					// node->secondLowestWeight can be larger than node->lowestWeight only when node->secondLowestWeight == maxWeight
					// or secondLowestWeigh is in subgradients
					if (population[id][dimension-2] > node->lowestWeight || population[id][dimension-2] > node->secondLowestWeight) { 
						// lazy must be passed down as long as node->lowestWeight/secondLowestWeight is dominated:
						// 1) if id update a child whose first covering box is valid and secondCoveringWeight=0, lazy will be useful
						// 2) otherwise the recursive call must stop before such child, or child->secondCoveringWeight > 0, so lazy will be useless
						fullyCoverUpdate(node, population[id][dimension-2], population[node->firstCoveringID][dimension-2], population[id][dimension-1]);
						node->lazy = 0.;
					} else {
						// no update for children's covering box/gradients, update lazy
						node->lazy += population[id][dimension-1] * (population[id][dimension-2] - population[node->firstCoveringID][dimension-2]);
					}
					node->secondCoveringWeight = population[node->firstCoveringID][dimension-2];
					if (node->leftChild == NULL && node->rightChild == NULL) {
						node->secondLowestWeight = node->secondCoveringWeight;
					}
					contributions[id] += population[id][dimension-1] * 
						( population[id][dimension-2] * (node->fullVolume - node->A) -
						node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
						node->dominatedA ); 
				} else {
					node->lazy += lazy;
					if (population[id][dimension-2] > node->lowestWeight || population[id][dimension-2] > node->secondLowestWeight) { 
						fullyCoverUpdate(node, population[id][dimension-2], max(node->secondCoveringWeight, highestWeight), population[id][dimension-1]);
						node->lazy = 0.;
					} else {
						node->lazy += population[id][dimension-1] * (population[id][dimension-2] - highestWeight);
					}
					node->secondCoveringWeight = 0.;
					contributions[id] += population[id][dimension-1] * 
						( population[id][dimension-2] * (node->fullVolume - node->A) -
						highestWeight * (node->fullVolume - node->secondA) - 
						node->dominatedA ); 
				}
				node->firstCoveringID = id;
			} else if ( partCovers(population[id], node->lowerbound) ) {
				if (node->leftChild == NULL && node->rightChild == NULL) {
					// insert non-dominated gradient
					// leaf node, update gradient and compute the contribution of gradinents and the covering box
					if (node->firstCoveringID != -1) {							
						// clear lazy and re-calculate dominating area of current first covering box
						if (node->secondCoveringWeight == 0.) {								
							contributions[node->firstCoveringID] -= lazy * (node->fullVolume - node->secondA) + population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
								highestWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						} else {
							contributions[node->firstCoveringID] -= population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
								node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						}
					} else {
						node->lazy += lazy;
					}
					double temp;
					// find the index of current box in the gradient using binary search
					// TODO: think of better way to find id
					bool isUpdateA;
					int idd = findID(node->gradientStructure.weight, population[id][dimension-2]);
					if (node->firstCoveringID != -1) {
						node->gradientStructure.Insert(idd, population[id][dimension-1], 
							population[node->firstCoveringID][dimension-2], node->lazy);
					} else {
						node->gradientStructure.Insert(idd, population[id][dimension-1], 
							max(node->secondCoveringWeight, highestWeight), node->lazy);
					}
					node->lazy = 0.;

					temp = 1.0;
					node->lowestWeight = maxWeight;
					node->secondLowestWeight = maxWeight;
					for (int i=0; i<dimension-2; i++) {
						if (node->gradientStructure.Gradient[i].crbegin() != node->gradientStructure.Gradient[i].crend()) {
							temp *= (node->gradientStructure.L[i] - node->gradientStructure.Gradient[i].crbegin()->first);
							if (node->gradientStructure.weight[node->gradientStructure.Gradient[i].crbegin()->second] < node->lowestWeight) {
								node->lowestWeight = node->gradientStructure.weight[node->gradientStructure.Gradient[i].crbegin()->second];
							}
						} else {
							temp *= node->gradientStructure.L[i];
						}
						if (node->gradientStructure.secondGradient[i].crbegin() != node->gradientStructure.secondGradient[i].crend()
							&& node->gradientStructure.weight[node->gradientStructure.secondGradient[i].crbegin()->second] < node->secondLowestWeight) {
								node->secondLowestWeight = node->gradientStructure.weight[node->gradientStructure.secondGradient[i].crbegin()->second];
						}
					}
					if (node->secondLowestWeight == node->lowestWeight) {
						node->secondLowestWeight = maxWeight;
					}
					node->A = node->fullVolume - temp;
					temp = 1.0;
					for (int i=0; i<dimension-2; i++) {
						if (!node->gradientStructure.SubWeightFinder[i].empty() &&
							node->gradientStructure.SubWeightFinder[i].begin()->first < node->secondLowestWeight) 
						{
							node->secondLowestWeight = node->gradientStructure.SubWeightFinder[i].begin()->first;
						}
						if (node->gradientStructure.secondGradient[i].crbegin() != node->gradientStructure.secondGradient[i].crend()) {
							temp *= (node->gradientStructure.L[i] - node->gradientStructure.secondGradient[i].crbegin()->first);
						} else {
							temp *= node->gradientStructure.L[i];
						}
					}
					if (node->secondCoveringWeight > 0) {
						node->secondLowestWeight = node->secondCoveringWeight;
					}
					node->secondA = node->fullVolume - temp;
					node->dominatedA = node->gradientStructure.secondV[0] + node->gradientStructure.secondV[1] - 
						node->gradientStructure.V[0] - node->gradientStructure.V[1];
					if (node->firstCoveringID != -1) {
						if (node->secondCoveringWeight == 0.) {								
							contributions[node->firstCoveringID] += population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
								highestWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						} else {
							contributions[node->firstCoveringID] += population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
								node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						}
					}
				} else {
					if (node->firstCoveringID != -1) {
						// clear lazy and re-calculate dominating area of current first covering box
						if (node->secondCoveringWeight == 0.) {								
							contributions[node->firstCoveringID] -= lazy * (node->fullVolume - node->secondA) + population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
								highestWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						} else {
							contributions[node->firstCoveringID] -= population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
								node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						}
					} else {
						node->lazy += lazy;
					}
					node->A = 0.;
					node->lowestWeight = maxWeight;
					node->secondA = 0.;
					node->dominatedA = 0.;
					node->secondLowestWeight = maxWeight;
					if (node->firstCoveringID != -1) {
						insertPoint(node->leftChild, id, node->lazy, population[node->firstCoveringID][dimension-2]);
					} else {
						insertPoint(node->leftChild, id, node->lazy, highestWeight);
					}
					updateA(node, node->leftChild);
					if (node->rightChild != NULL) {
						if (node->firstCoveringID != -1) {
							insertPoint(node->rightChild, id, node->lazy, population[node->firstCoveringID][dimension-2]);
						} else {
							insertPoint(node->rightChild, id, node->lazy, highestWeight);
						}
						updateA(node, node->rightChild);
					}
					node->lazy = 0.;
					if (node->firstCoveringID != -1) {
						if (node->secondCoveringWeight == 0.) {								
							contributions[node->firstCoveringID] += population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
								highestWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						} else {
							contributions[node->firstCoveringID] += population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
								node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						}
					}
				}
			} else {
				if (node->firstCoveringID != -1) {
					if (node->secondCoveringWeight == 0.) {
						contributions[node->firstCoveringID] -= lazy * (node->fullVolume - node->secondA);
					} 
				} else {
					node->lazy += lazy;
				}
			}
		} else {
			// note that call of insertPoint(node, id) implies that id is not dominated by covering box of fathers
			// so it is only dominated by first covering box of this node
			if ( covers(population[id], node->upperbound) ) {
				// id becomes new second covering box, and its weight must be smaller than node->lowestWeight, 
				// otherwise node->firstCoveringWeight would be larger than node->lowestWeight
				if (node->firstCoveringID != -1) {
					// clear lazy and re-calculate dominating area of current first covering box
					if (node->secondCoveringWeight == 0.) {								
						contributions[node->firstCoveringID] -= lazy * (node->fullVolume - node->secondA) + population[id][dimension-1] * 
							( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
							highestWeight * (node->fullVolume - node->secondA) - 
							node->dominatedA ); 
					} else {
						contributions[node->firstCoveringID] -= population[id][dimension-1] * 
							( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
							node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
							node->dominatedA ); 
					}
				} else {
					node->lazy += lazy;
				}
				if (population[id][dimension-2] > node->secondLowestWeight) {	
					fullyCoverUpdate2(node, population[id][dimension-2]);
				} 
				node->secondCoveringWeight = population[id][dimension-2];						
				if (node->leftChild == NULL && node->rightChild == NULL) {
					node->secondLowestWeight = node->secondCoveringWeight;
				}
				if (node->firstCoveringID != -1) {
					contributions[node->firstCoveringID] += population[id][dimension-1] * 
						( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
						node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
						node->dominatedA ); 
				}
			} else if ( partCovers(population[id], node->lowerbound) ) {
				if (node->leftChild == NULL && node->rightChild == NULL) {
					// insert subgradient that is only dominated by first covering box of this node
					// it will only affect gradientStructure2
					if (node->firstCoveringID != -1) {							
						if (node->secondCoveringWeight == 0.) {								
							contributions[node->firstCoveringID] -= lazy * (node->fullVolume - node->secondA) + population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
								highestWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						} else {
							contributions[node->firstCoveringID] -= population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
								node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						}
					} else {
						node->lazy += lazy;
					}
					double temp;
					bool isUpdateA;
					int idd = findID(node->gradientStructure.weight, population[id][dimension-2]);
					node->gradientStructure.Insert2(idd);

					temp = 1.0;
					node->secondLowestWeight = maxWeight;
					for (int i=0; i<dimension-2; i++) {
						if (node->gradientStructure.secondGradient[i].crbegin() != node->gradientStructure.secondGradient[i].crend()) {
							temp *= (node->gradientStructure.L[i] - node->gradientStructure.secondGradient[i].crbegin()->first);
							if (node->gradientStructure.weight[node->gradientStructure.secondGradient[i].crbegin()->second] < node->secondLowestWeight) {
								node->secondLowestWeight = node->gradientStructure.weight[node->gradientStructure.secondGradient[i].crbegin()->second];
							}
						} else {
							temp *= node->gradientStructure.L[i];
						}
					}
					if (node->secondLowestWeight == node->lowestWeight) {
						// this can happen if the inserted gradient is dominated by other gradients and
						// other gradients are all non-dominated ... OMG
						node->secondLowestWeight = maxWeight;	
						for (int i=0; i<dimension-2; i++) {
							if (!node->gradientStructure.SubWeightFinder[i].empty() &&
								node->gradientStructure.SubWeightFinder[i].begin()->first < node->secondLowestWeight) 
							{
								node->secondLowestWeight = node->gradientStructure.SubWeightFinder[i].begin()->first;
							}
						}
					}
					if (node->secondCoveringWeight > 0) {
						node->secondLowestWeight = node->secondCoveringWeight;
					}
					node->secondA = node->fullVolume - temp;
					node->dominatedA = node->gradientStructure.secondV[0] + node->gradientStructure.secondV[1] - 
						node->gradientStructure.V[0] - node->gradientStructure.V[1];

					if (node->firstCoveringID != -1) {
						if (node->secondCoveringWeight == 0.) {								
							contributions[node->firstCoveringID] += population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
								highestWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						} else {
							contributions[node->firstCoveringID] += population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
								node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						}
					}
				} else {
					// update contribution of first covering box of this node (may use lazy)
					// chilren's contribution will not change and lazy need not to pass down
					if (node->firstCoveringID != -1) {
						// clear lazy and re-calculate dominating area of current first covering box
						if (node->secondCoveringWeight == 0.) {								
							contributions[node->firstCoveringID] -= lazy * (node->fullVolume - node->secondA) + population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
								highestWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						} else {
							contributions[node->firstCoveringID] -= population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
								node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						}
					} else {
						node->lazy += lazy;
					}
					node->A = 0.;
					node->lowestWeight = maxWeight;
					node->secondA = 0.;
					node->dominatedA = 0.;
					node->secondLowestWeight = maxWeight;
					insertPoint2(node->leftChild, id);
					updateA(node, node->leftChild);
					if (node->rightChild != NULL) {
						insertPoint2(node->rightChild, id);
						updateA(node, node->rightChild);
					}
					if (node->firstCoveringID != -1) {
						if (node->secondCoveringWeight == 0.) {								
							contributions[node->firstCoveringID] += population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) - 
								highestWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						} else {
							contributions[node->firstCoveringID] += population[id][dimension-1] * 
								( population[node->firstCoveringID][dimension-2] * (node->fullVolume - node->A) -
								node->secondCoveringWeight * (node->fullVolume - node->secondA) - 
								node->dominatedA ); 
						}
					}
				}
			} else {
				if (node->firstCoveringID != -1) {
					if (node->secondCoveringWeight == 0.) {
						contributions[node->firstCoveringID] -= lazy * (node->fullVolume - node->secondA);
					} 
				} else {
					node->lazy += lazy;
				}
			}
		}
	} else {
		if (node->firstCoveringID != -1) {
			if (node->secondCoveringWeight == 0.) {
				contributions[node->firstCoveringID] -= lazy * (node->fullVolume - node->secondA);
			} 
		} else {
			node->lazy += lazy;
		}
	}
	node->highestWeight = highestWeight;
}

// id has been dominated by covering box of this node or fathers, so it is dominated by all first covering boxes of this node and children	
// besides, id is partially covering box of fathers otherwise we will call fullyCoverUpdate2
// recursive call will end when
// 1) some child has first covering box or weight[id] < child->secondCoveringWeight
// id will not affect children of this child, and id will not change contribution of the first covering box
// before meeting this child, id may dominate some second covering box 
// since weight[id] < max(node->firstCoveringWeight, node->highestWeight), id will not affect node->lazy
// in summary, we need not to pass down lazy
// 2) this node is a leaf and it has no first covering box and weight[id] > node->secondCoveringWeight
// if id is a fully covering box, it will only dominate subGradients and second covering box because id is dominated by fathers
// so contributions of gradients will not change 
// actually the change of contributions have been updated by or recorded as lazy of fathers, so we need not to update at this moment
// otherwise, we only update subGradient and no contribution needs to be updated due to the same reason above
// only secondCoveringWeight or secondLowestWeight will change, leading to secondA and dominatedA need to be updated

// as a result, we need not to pass down lazy, 
// only contribution of the father whose starts the recursive call may change due to secondA, dominatedA's update
// no contribution will be changed in this function call
inline void insertPoint2(YildizTreeNode *node, int id) {
	
	if (node->firstCoveringID == -1 && 
		population[id][dimension-2] > node->secondCoveringWeight) 
	{
		if ( covers(population[id], node->upperbound) ) {
			if (population[id][dimension-2] > node->secondLowestWeight) {
				// this can only happen when child's first covering box does not exist
				// and second covering box or subgradient exists
				// so the update is unavoidable
				fullyCoverUpdate2(node, population[id][dimension-2]);
			} 
			node->secondCoveringWeight = population[id][dimension-2];
			if (node->leftChild == NULL && node->rightChild == NULL) {
				node->secondLowestWeight = node->secondCoveringWeight;
			}
		} else if ( partCovers(population[id], node->lowerbound) ) {
			if (node->leftChild == NULL && node->rightChild == NULL) {
				
				// insert a dominated gradient, only secondA and dominatedA may changedouble temp;
				int idd = findID(node->gradientStructure.weight, population[id][dimension-2]);
				bool isUpdateA;
				node->gradientStructure.Insert2(idd);

				double temp = 1.0;
				node->secondLowestWeight = maxWeight;
				for (int i=0; i<dimension-2; i++) {
					if (node->gradientStructure.secondGradient[i].crbegin() != node->gradientStructure.secondGradient[i].crend()) {
						temp *= (node->gradientStructure.L[i] - node->gradientStructure.secondGradient[i].crbegin()->first);
						if (node->gradientStructure.weight[node->gradientStructure.secondGradient[i].crbegin()->second] < node->secondLowestWeight) {
							node->secondLowestWeight = node->gradientStructure.weight[node->gradientStructure.secondGradient[i].crbegin()->second];
						}
					} else {
						temp *= node->gradientStructure.L[i];
					}
				}
				if (node->secondLowestWeight == node->lowestWeight) {
					node->secondLowestWeight = maxWeight;
					for (int i=0; i<dimension-2; i++) {
						if (!node->gradientStructure.SubWeightFinder[i].empty() &&
							node->gradientStructure.SubWeightFinder[i].begin()->first < node->secondLowestWeight) 
						{
							node->secondLowestWeight = node->gradientStructure.SubWeightFinder[i].begin()->first;
						}
					}
				}
				if (node->secondCoveringWeight > 0) {
					node->secondLowestWeight = node->secondCoveringWeight;
				}
				node->secondA = node->fullVolume - temp;
				node->dominatedA = node->gradientStructure.secondV[0] + node->gradientStructure.secondV[1] - 
					node->gradientStructure.V[0] - node->gradientStructure.V[1];
			} else {
				node->A = 0.;
				node->lowestWeight = maxWeight;
				node->secondA = 0.;
				node->dominatedA = 0.;
				node->secondLowestWeight = maxWeight;
				insertPoint2(node->leftChild, id);
				updateA(node, node->leftChild);
				if (node->rightChild != NULL) {
					insertPoint2(node->rightChild, id);
					updateA(node, node->rightChild);
				}
			}
		}
	}
}

inline void updateContribution(YildizTreeNode *node) {
	if (node->leftChild == NULL && node->rightChild == NULL) {
		// clear lazy for lightest gradient
		map<double, int>::iterator k, k2;
		double otherSweep;
		for (int i=0; i<2; i++) {
			if (!node->gradientStructure.Gradient[i].empty()) {
				k = node->gradientStructure.Gradient[i].end();
				--k;
				if (k != node->gradientStructure.Gradient[i].begin()) {
					k2 = k;
					--k2;
					otherSweep = k2->first;
				} else {
					otherSweep = 0.;
				}
				if (!node->gradientStructure.SubGradient[i].empty()) {
					k2 = node->gradientStructure.SubGradient[i].end();
					--k2;
					if (otherSweep < k2->first) {
						otherSweep = k2->first;
					}
				}
				bool type = node->gradientStructure.type[k->second];
				if (!node->gradientStructure.Gradient[1-type].empty()) {
					node->gradientStructure.C[k->second] -= node->lazy * (k->first - otherSweep) * 
						(node->gradientStructure.L[1-type] - node->gradientStructure.Gradient[1-type].crbegin()->first);
				} else {
					node->gradientStructure.C[k->second] -= node->lazy * (k->first - otherSweep) * node->gradientStructure.L[1-type];
				}
			}
		}
		// parse contributions
		for (int i=1; i<=node->partialCoverNumber; i++) {
			contributions[node->partialCoverIndex[i-1]] += node->gradientStructure.C[i] +
				node->gradientStructure.dominated[node->gradientStructure.type[i]].Sum(i) * 
				node->gradientStructure.projection[node->gradientStructure.type[i]][i];
		}
	} else {
		// pass down lazy or clear lazy
		if (node->leftChild->firstCoveringID != -1) {
			if (node->leftChild->secondCoveringWeight == 0.) {
				contributions[node->leftChild->firstCoveringID] -= node->lazy * (node->leftChild->fullVolume - node->leftChild->secondA);
			}
		} else {
			node->leftChild->lazy += node->lazy;
		}
		updateContribution(node->leftChild);
		if (node->rightChild != NULL) {
			if (node->rightChild->firstCoveringID != -1) {
				if (node->rightChild->secondCoveringWeight == 0.) {
					contributions[node->rightChild->firstCoveringID] -= node->lazy * (node->rightChild->fullVolume - node->rightChild->secondA);
				}
			} else {
				node->rightChild->lazy += node->lazy;
			}
			updateContribution(node->rightChild);
		}
		node->lazy = 0.;
	}
}

int main(int  argc, char  *argv[]) {

	int i, j;

	/* check parameters */
	if (argc < 4)  {
		fprintf(stderr, "usage: HVC4DGS <number of points> <input file> <reference point file> <outputfile(optional)>\n");
		exit(1);
	}
	sscanf(argv[1], "%d", &popsize);
	char *filenameData = argv[2];
	char *filenameRef = argv[3];

	/* read in data */
	char word[30];

	// read in reference point
	static double* ref = new double[dimension];
	ifstream fileRef;
	fileRef.open(filenameRef, ios::in);
	if (!fileRef.good()){
		printf("reference point file not found \n");
		exit(0);
	}
	for (i=0; i<dimension; i++) {
		fileRef >> word;
		ref[i] = atof(word);
	}
	fileRef.close();

	// read in data file
	ifstream fileData;
	fileData.open(filenameData, ios::in);
	if (!fileData.good()){
		printf("data file not found \n");
		exit(0);
	}
	population = vector<double*>(popsize);
	for (i=0; i<popsize; i++) {
		population[i] = new double[dimension];
		for (j=0; j<dimension; j++) {
			fileData >> word;
			population[i][j] = ref[j] - atof(word);
		}
	}
	fileData.close();

	// timing codes in Linux
	struct timeval tv1, tv2;
	struct rusage ru_before, ru_after;

	double *hvc;

	getrusage (RUSAGE_SELF, &ru_before);
	
	// initialize region
	maxWeight = 0.;
	double* regionLow = new double[dimension];
	double* regionUp = new double[dimension];
	for (j=0; j<dimension; j++)  {
		// determine maximal j coordinate
		regionUp[j] = 0.0;
		for (i=0; i<popsize; i++) {
			if (population[i][j] > regionUp[j]) {
				regionUp[j] = population[i][j];
			}
		}
		regionLow[j] = 0.;
	}
	// sqrt of popsize
	dSqrtDataNumber = sqrt((double)popsize);

	int *sorted_height = new int[popsize];
	for (i=0; i<popsize; i++) {
		sorted_height[i] = i;
		if (maxWeight < population[i][dimension-2]) {
			maxWeight = population[i][dimension-2];
		}
	}
	maxWeight += 1.0;
	Index_Descend_Sort(population, sorted_height, popsize, dimension-1);
	sort(population.begin(), population.end(), Yildiz_cmp);

	treeProjection = new int[popsize];
	piles = new int[popsize];
	boundaries = new int[popsize];
	noBoundaries = new int[popsize];
	contributions = new double[popsize];
	for (i=0; i<popsize; i++) {
		contributions[i] = 0.;
	}

	// root node for the first coordinate
	YildizTreeNode *root = new YildizTreeNode();
	root->partialCoverNumber = popsize;
	root->partialCoverIndex = new int[popsize];
	root->boxType = new bool[popsize];
	root->dims = vector<int>(dimension-2);
	root->lowerbound = new double[dimension-2];
	root->upperbound = new double[dimension-2];
	for (i=0; i<popsize; i++) {
		root->partialCoverIndex[i] = i;
		root->boxType[i] = true;
	}
	for (i=0; i<dimension-2; i++) {
		root->dims[i] = i;
		root->lowerbound[i] = 0.;
		root->upperbound[i] = regionUp[i];
	}
	root->isNewLinkedList = true;
	root->LinkedList = vector<int*>(dimension-1);
	for (i=0; i<dimension-1; i++) {
		root->LinkedList[i] = new int[popsize];
		for (j=0; j<popsize; j++) {
			root->LinkedList[i][j] = j;
		}
		Index_Ascend_Sort(population, root->LinkedList[i], popsize, i);
	}
	alter = 0;

	buildTree(root);
	
	for (i=0; i<popsize; i++) {
		insertPoint(root, i, 0., 0.);
	}
	updateContribution(root);	
	
	hvc = new double[popsize];
	for (i=0; i<popsize; i++) {
		hvc[sorted_height[i]] = contributions[i];
	}

	getrusage (RUSAGE_SELF, &ru_after);
	
	tv1 = ru_before.ru_utime;
	tv2 = ru_after.ru_utime;
	
	if (argc == 5) {
		ofstream myoutput(argv[4]);					
		if (myoutput.fail()) {
			printf("output data file open failed \n");
			exit(0);
		}	
		for (i=0; i<popsize; i++) {
			myoutput << setprecision(16) << hvc[i] << endl;
		}
		myoutput << setprecision(8) << tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6;
		myoutput.close();
	} else {
		for (i=0; i<popsize; i++) {
			printf("%.16g\n", hvc[i]);
		}
		printf("Time(s): %.10g\n", tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6);
	}

	return 0;
}
