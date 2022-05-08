#pragma once

#include "utils.h"

using namespace std;

struct PDD {
    int dim;
    vector<double*> H;
    vector<double*> V;
    vector<set_t> incidence; // V to H incidence.
};

/* Function transfers the memory ownership of H_new to pdd. */
void PDD_batch_intersect(PDD& pdd, vector<double*>& H_new);

/* Function takes ownership of memory of both pdd1 and pdd2 - they cannot be used afterwards. */
PDD PDD_intersect_two_PDDs(PDD& pdd1, PDD& pdd2);

/* Function takes ownership of memory of the input. */
PDD PDD_intersect_all(vector<PDD>& pdds);

void PDD_adjust_H_for_soundness_finite_polytope(const int dim,
                                                vector<double*>& H,
                                                const vector<double*>& V);

void PDD_debug_consistency_check(const PDD& pdd);
