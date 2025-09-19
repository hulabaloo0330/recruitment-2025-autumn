/*

   Do not modify this file.

*/

#ifndef GMRES_H
#define GMRES_H
#include <tuple>
#include "sparseMatrix.hpp"

using RESULT = std::tuple<int, float, double>;

void initialize(SpM<double> *A, double *x, double *b);
RESULT gmres(SpM<double> *A_d, double *_x, double *_b);

#endif
