#ifndef SUM_REDUCTION_H
#define SUM_REDCUTION_H


template <typename T> void sum(T* input, int N, T initial);
template <typename T> void sum_atomic(T* input, int N, T* initial);
template <typename T> void sum_seg_reduction(T*, int);

#endif
