
#include <Rcpp.h>
using namespace Rcpp;

#include <cmath>
#include <algorithm>

//---------------------------------//
// SERIAL FUNCTIONS
//---------------------------------//

// [[Rcpp::export]]
NumericMatrix matrixSqrt(NumericMatrix orig) {
  NumericMatrix mat(orig.nrow(), orig.ncol());
  std::transform(orig.begin(), orig.end(), mat.begin(), ::sqrt);
  return mat;
}

// [[Rcpp::export]]
double vectorSum(NumericVector x) {
  return std::accumulate(x.begin(), x.end(), 0.0);  
}

//---------------------------------//
// PARALLEL FUNCTIONS
//---------------------------------//

// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
using namespace RcppParallel;

struct SquareRoot : public Worker
{
  // source matrix
  const RMatrix<double> input;
  
  // destination matrix
  RMatrix<double> output;
  
  // initialize with source and destination
  SquareRoot(const NumericMatrix input, NumericMatrix output) 
    : input(input), output(output) {}
  
  // take the square root of the range of elements requested
  void operator()(std::size_t begin, std::size_t end) {
    // std::cout << "begin " << begin << " end " << end << std::endl;
    std::transform(input.begin() + begin, 
                   input.begin() + end, 
                   output.begin() + begin, 
                   ::sqrt);
  }
};

// [[Rcpp::export]]
NumericMatrix parallelMatrixSqrt(NumericMatrix orig) {
  NumericMatrix mat(orig.nrow(), orig.ncol());
  SquareRoot squareRoot(orig, mat);
  parallelFor(0, orig.length(), squareRoot);
  return mat;
}

struct Sum : public Worker
{   
  // source vector
  const RVector<double> input;
  
  // accumulated value
  double value;
  
  // constructors
  Sum(const NumericVector input) : input(input), value(0) {}
  Sum(const Sum& sum, RcppParallel::Split) : input(sum.input), value(0) {}
  
  // accumulate just the element of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    value += std::accumulate(input.begin() + begin, input.begin() + end, 0.0);
  }
  
  // join my value with that of another Sum
  void join(const Sum& rhs) { 
    value += rhs.value; 
  }
};

// [[Rcpp::export]]
double parallelVectorSum(NumericVector x) {
  
  // declare the SumBody instance 
  Sum sum(x);
  
  // call parallel_reduce to start the work
  parallelReduce(0, x.length(), sum);
  
  // return the computed sum
  return sum.value;
}
