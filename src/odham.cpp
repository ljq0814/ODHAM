#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat fun_maxmin(const arma::mat& X, const arma::mat& Y, const int n_subset, const int d){
    // call the function bs in R
    Environment pkg = Environment::namespace_env("splines");
    Function bs = pkg["bs"];

    int n = X.n_rows;
    int p = X.n_cols;
    int q = Y.n_cols;

    arma::vec Tmax(n, fill::zeros);
    arma::vec Tmin(n, fill::zeros);
    arma::vec TT(n_subset, fill::zeros);

    // main recursion over i
    for(int i = 0; i < n; ++i) {
        // generation the index dropping i
        uvec S_i = arma::regspace<uvec>(0, n - 1);
        S_i.shed_row(i);

        for(int m = 0; m < n_subset; ++m) {
            // random subsampling
            uvec isamp = Rcpp::RcppArmadillo::sample(S_i, std::floor(n * 0.5), false);

            // construct sampling index set (include i)
            uvec idx1(isamp.n_elem + 1);
            idx1(0) = i;
            for(uword k = 0; k < isamp.n_elem; ++k) 
                idx1(k + 1) = isamp(k);

            // subset of Y
            arma::mat Y1 = Y.rows(idx1);
            arma::mat Y2 = Y.rows(isamp);

            arma::vec d_i(p, fill::zeros);

            for(int j = 0; j < p; ++j) {
                arma::mat X1_j_tmp = X.rows(idx1);
                arma::mat X1_j_tmp2 = X1_j_tmp.col(j);
                NumericVector X1_j = wrap(X1_j_tmp2);
                arma::mat X2_j_tmp = X.rows(isamp);
                arma::mat X2_j_tmp2 = X2_j_tmp.col(j);
                NumericVector X2_j = wrap(X2_j_tmp2);

                // call the bs() in R
                NumericMatrix B1_R = bs(X1_j, Named("degree") = d);
                NumericMatrix B2_R = bs(X2_j, Named("degree") = d);

                // trasform it into Armadillo
                arma::mat B1(B1_R.begin(), B1_R.nrow(), B1_R.ncol(), false);
                arma::mat B2(B2_R.begin(), B2_R.nrow(), B2_R.ncol(), false);

                // ls solution
                arma::mat lhs1 = B1.t() * B1;
                arma::mat rhs1 = B1.t() * Y1;
                arma::mat rho1 = arma::solve(lhs1, rhs1);

                arma::mat lhs2 = B2.t() * B2;
                arma::mat rhs2 = B2.t() * Y2;
                arma::mat rho2 = arma::solve(lhs2, rhs2);

                // calculate square diff
                d_i(j) = arma::accu(arma::square(rho1 - rho2));
            }
            TT(m) = arma::mean(d_i);
        }
        Tmax(i) = TT.max();
        Tmin(i) = TT.min();
    }
    arma::mat res(n, 2);
    res.col(0) = Tmax;
    res.col(1) = Tmin;
    return res;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec fun_threshold(const arma::mat& X_clean, const arma::mat& Y_clean, const int d, const int B) {
    //step 2: find the threshold according to the clean set
    int n_clean = X_clean.n_rows;
    int p = X_clean.n_cols;
    int id = 0;
            
    Environment pkg = Environment::namespace_env("splines");
    Function bs = pkg["bs"];

    arma::vec TT(B, fill::zeros);

    for (int m = 0; m < B; ++m) {
        arma::uvec sequence = arma::regspace<arma::uvec>(1, n_clean-1);
        arma::uvec isamp = Rcpp::RcppArmadillo::sample(sequence, floor(n_clean * 0.8), true);
        
        arma::uvec idx1(isamp.n_elem + 1);
        idx1(0) = id;
        for(uword k = 0; k < isamp.n_elem; ++k) 
            idx1(k + 1) = isamp(k);

        arma::mat X1 = X_clean.rows(idx1);
        arma::mat Y1 = Y_clean.rows(idx1);
        arma::mat X2 = X_clean.rows(isamp);
        arma::mat Y2 = Y_clean.rows(isamp);
    
        arma::mat rho1(p, d, fill::zeros);
        arma::mat rho2(p, d, fill::zeros);
        arma::vec d_i(p, fill::zeros);

        for (int j = 0; j < p; ++j) {
            NumericVector X1_j = wrap(X1.col(j));
            NumericVector X2_j = wrap(X2.col(j));

            NumericMatrix B1_R = bs(X1_j, Named("degree") = d);
            NumericMatrix B2_R = bs(X2_j, Named("degree") = d);
      
            arma::mat B1(B1_R.begin(), B1_R.nrow(), B1_R.ncol(), false);
            arma::mat B2(B2_R.begin(), B2_R.nrow(), B2_R.ncol(), false);
    
            arma::mat lhs1 = B1.t() * B1;
            arma::mat rhs1 = B1.t() * Y1;
            arma::mat lhs2 = B2.t() * B2;
            arma::mat rhs2 = B2.t() * Y2;
            rho1.row(j) = arma::solve(lhs1, rhs1).t();
            rho2.row(j) = arma::solve(lhs2, rhs2).t();

            d_i(j) = arma::accu(arma::square(rho1.row(j) - rho2.row(j)));
        }
        TT(m) = arma::mean(d_i);
    }
  return TT; 
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec fun_T_test(const arma::mat& X2, const arma::mat& Y2, const arma::mat& X_clean, const arma::mat& Y_clean,
                    const arma::vec& TT, const int p, const int d){
    int n2 = X2.n_rows;
    int n1 = X_clean.n_rows;
    arma::vec T_test2(n2, fill::zeros);

    Environment pkg = Environment::namespace_env("splines");
    Function bs = pkg["bs"];
    
    for (int i = 0; i < n2; ++i) {
        arma::mat rho1(p, d, fill::zeros);
        arma::mat rho2(p, d, fill::zeros);
        arma::vec d_i(p, fill::zeros);

        for (int j = 0; j < p; ++j) {
            arma::mat B2tmp(n1+1, 1, fill::zeros);
            B2tmp.submat(span(0,n1-1),span(0,0)) = X_clean.col(j);
            B2tmp(n1,0) = X2(i,j);
            arma::mat Y2tmp(n1+1,1,fill::zeros);
            Y2tmp.rows(0,n1-1) = Y_clean;
            Y2tmp.row(n1) = Y2.row(i);

            NumericVector X_clean_j = wrap(X_clean.col(j));
            NumericVector X2_j = wrap(B2tmp);
            
            NumericMatrix B1_R = bs(X_clean_j, Named("degree") = d);
            NumericMatrix B2_R = bs(X2_j, Named("degree") = d);
            
            arma::mat B1(B1_R.begin(), B1_R.nrow(), B1_R.ncol(), false);
            arma::mat B2(B2_R.begin(), B2_R.nrow(), B2_R.ncol(), false);

            arma::mat lhs1 = B1.t() * B1;
            arma::mat rhs1 = B1.t() * Y_clean;
            arma::mat lhs2 = B2.t() * B2;
            arma::mat rhs2 = B2.t() * Y2tmp;
            rho1.row(j) = arma::solve(lhs1, rhs1).t();
            rho2.row(j) = arma::solve(lhs2, rhs2).t();

            d_i(j) = arma::accu(arma::square(rho1.row(j) - rho2.row(j)));
        }
        T_test2(i) = arma::mean(d_i);
    }
    T_test2 = T_test2 * arma::median(TT) / arma::median(T_test2);
    return T_test2;
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat fun_maxmin_new(const arma::cube& Xbs, const arma::mat& Y,const int n_subset){
    int n = Xbs.n_rows;
    int p = Xbs.n_slices;

    arma::vec Tmax(n, fill::zeros);
    arma::vec Tmin(n, fill::zeros);
    arma::vec TT(n_subset, fill::zeros);

    // main recursion over i
    for(int i = 0; i < n; ++i) {
        // generation the index dropping i
        uvec S_i = arma::regspace<uvec>(0, n - 1);
        S_i.shed_row(i);

        for(int m = 0; m < n_subset; ++m) {
            // random subsampling
            uvec isamp = Rcpp::RcppArmadillo::sample(S_i, std::floor(n * 0.5), false);

            // construct sampling index set (include i)
            uvec idx1(isamp.n_elem + 1);
            idx1(0) = i;
            for(uword k = 0; k < isamp.n_elem; ++k) 
                idx1(k + 1) = isamp(k);
            
            std::cout << idx1 << std::endl;
            // subset of Y
            arma::mat Y1 = Y.rows(idx1);
            arma::mat Y2 = Y.rows(isamp);

            arma::vec d_i(p, fill::zeros);

            for(int j = 0; j < p; ++j) {
                arma::mat X1_j_tmp = Xbs.slice(j);
                arma::mat B1 = X1_j_tmp.rows(idx1);
                arma::mat X2_j_tmp = Xbs.slice(j);
                arma::mat B2 = X2_j_tmp.rows(isamp);

                // ls solution
                arma::mat lhs1 = B1.t() * B1;
                arma::mat rhs1 = B1.t() * Y1;
                arma::mat rho1 = arma::solve(lhs1, rhs1);

                arma::mat lhs2 = B2.t() * B2;
                arma::mat rhs2 = B2.t() * Y2;
                arma::mat rho2 = arma::solve(lhs2, rhs2);

                // calculate square diff
                d_i(j) = arma::accu(arma::square(rho1 - rho2));
            }
            TT(m) = arma::mean(d_i);
        }
        Tmax(i) = TT.max();
        Tmin(i) = TT.min();
    }
    arma::mat res(n, 2);
    res.col(0) = Tmax;
    res.col(1) = Tmin;
    return res;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec fun_threshold_new(const arma::cube& Xbs_clean, const arma::mat& Y_clean, const int B) {
    //step 2: find the threshold according to the clean set
    int n_clean = Xbs_clean.n_rows;
    int d = Xbs_clean.n_cols;
    int p = Xbs_clean.n_slices;
    int id = 0;

    arma::vec TT(B, fill::zeros);

    for (int m = 0; m < B; ++m) {
        arma::uvec sequence = arma::regspace<arma::uvec>(1, n_clean-1);
        arma::uvec isamp = Rcpp::RcppArmadillo::sample(sequence, floor(n_clean * 0.8), true);
        
        arma::uvec idx1(isamp.n_elem + 1);
        idx1(0) = id;
        for(uword k = 0; k < isamp.n_elem; ++k) 
            idx1(k + 1) = isamp(k);

        arma::mat Y1 = Y_clean.rows(idx1);
        arma::mat Y2 = Y_clean.rows(isamp);
    
        arma::mat rho1(p, d, fill::zeros);
        arma::mat rho2(p, d, fill::zeros);
        arma::vec d_i(p, fill::zeros);

        for (int j = 0; j < p; ++j) {
            arma::mat X1_j_tmp = Xbs_clean.slice(j);
            arma::mat X2_j_tmp = Xbs_clean.slice(j);

            arma::mat B1 = X1_j_tmp.rows(idx1);
            arma::mat B2 = X2_j_tmp.rows(isamp);
    
            arma::mat lhs1 = B1.t() * B1;
            arma::mat rhs1 = B1.t() * Y1;
            arma::mat lhs2 = B2.t() * B2;
            arma::mat rhs2 = B2.t() * Y2;
            rho1.row(j) = arma::solve(lhs1, rhs1).t();
            rho2.row(j) = arma::solve(lhs2, rhs2).t();

            d_i(j) = arma::accu(arma::square(rho1.row(j) - rho2.row(j)));
        }
        TT(m) = arma::mean(d_i);
    }
    return TT; 
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec fun_T_test_new(const arma::cube& X2bs, const arma::mat& Y2,
                const arma::cube& Xbs_clean, const arma::mat& Y_clean, const arma::vec& TT) {
    int n2 = X2bs.n_rows;
    int n1 = Xbs_clean.n_rows;
    int d = X2bs.n_cols;
    int p = X2bs.n_slices;
    arma::vec T_test2(n2, fill::zeros);
    
    for (int i = 0; i < n2; ++i) {
        arma::mat rho1(p, d, fill::zeros);
        arma::mat rho2(p, d, fill::zeros);
        arma::vec d_i(p, fill::zeros);
    
        for (int j = 0; j < p; ++j) {
            arma::mat B2(n1+1, d, fill::zeros);
            arma::mat B1 = Xbs_clean.slice(j);
            B2.submat(span(0,n1-1),span(0,d-1)) = B1;
            arma::mat X2bs_tmp = X2bs.slice(j);
            B2.row(n1) = X2bs_tmp.row(i);
            
            arma::mat Y2tmp(n1+1,1,fill::zeros);
            Y2tmp.rows(0,n1-1) = Y_clean;
            Y2tmp.row(n1) = Y2.row(i);
    
            arma::mat lhs1 = B1.t() * B1;
            arma::mat rhs1 = B1.t() * Y_clean;
            arma::mat lhs2 = B2.t() * B2;
            arma::mat rhs2 = B2.t() * Y2tmp;
            rho1.row(j) = arma::solve(lhs1, rhs1).t();
            rho2.row(j) = arma::solve(lhs2, rhs2).t();

            d_i(j) = arma::accu(arma::square(rho1.row(j) - rho2.row(j)));
        }
        T_test2(i) = arma::mean(d_i);
    }
    T_test2 = T_test2 * arma::median(TT) / arma::median(T_test2);
    return T_test2;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
NumericVector test(const arma::mat& X2,const arma::mat &X1) {
    int n2 = X2.n_rows;
    int p = X2.n_cols;
    int n1 = X1.n_rows;

    int i = 0;
    int j = 0;

    arma::mat B2(n1+1, 1, fill::zeros);
    B2.submat(span(0,n1-1),span(0,0)) = X1.col(j);
    B2(n1,0) = X2(i,j);

    NumericVector X2_j = wrap(B2);
    return(X2_j);
}

// [[Rcpp::export]]
arma::uvec sample_index(const int &size, const arma::mat &Y){
    int i = 9;
    arma::uvec sequence = arma::linspace<arma::uvec>(1, size-1, size-1);
    return sequence;
}
