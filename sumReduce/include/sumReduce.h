#ifndef SUM_REDUCTION_H
#define SUM_REDCUTION_H

enum sumType {
    SUM,
    ATOMIC,
    SEG,
    COAL,
    SHARED,
    COARSE
};

// Use strategy class?

template <typename T> void sum(T* input, int N, T* initial);
template <typename T> void sum_atomic(T* input, int N, T* initial);
template <typename T> void sum_seg_reduction(T*, int, T*);
template <typename T> void sum_coalecsing(T*, int, T*);
template <typename T> void sum_shared_mem(T*, int, T*);
template <typename T> void sum_coarsened(T*, int, T*);
#endif
