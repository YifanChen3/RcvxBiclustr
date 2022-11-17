#include <iostream>
#include <RcppArmadillo.h>
#include <math.h>

using namespace Rcpp;
using namespace arma;

//[[Rcpp::depends(RcppArmadillo)]]


double f1(const double x, const arma::vec& resSq, const int n, const double rhs) {
    return mean(min(resSq / x, ones(n))) - rhs;
}


double rootf1(const arma::vec& resSq, const int n, const double rhs, double low, double up,
              const double tol = 0.001, const int maxIte = 500) {
    int ite = 1;
    while (ite <= maxIte && up - low > tol) {
        double mid = 0.5 * (up + low);
        double val = f1(mid, resSq, n, rhs);
        if (val < 0) {
            up = mid;
        } else {
            low = mid;
        }
        ite++;
    }
    return 0.5 * (low + up);
}

double mad(const arma::vec& x) {
    return 1.482602 * median(abs(x - median(x)));
}

double quantile(arma::vec& x, double q){
    int n = x.size();
    double i = (n-1)*q;
    int low = floor(i);
    int high = ceil(i);
    double qs = x(low);
    double h = (i-low);
    return (1.0 - h) * qs + h * x(high);
}

//[[Rcpp::export]]
NumericVector robustweights(NumericMatrix X, double delta=15.0, double zeta=0.1) {
    int n = X.nrow(), p=X.ncol();
    arma::mat X1(X.begin(),n,p,false);

    arma::vec d(p,fill::zeros);
    arma::mat w(n,n,fill::zeros);

    for (int i=0;i<n-1;i++){
        for (int j=i+1;j<n;j++){
            double num=0.;
            for (int k=0;k<p;k++){
                d(k) = fabs(X1(i,k)-X1(j,k));

                if (d(k) < delta){
                    num = num + pow(d(k),2);
                }
                else {
                    num = num + pow(delta,2);
                }
            }
            w(i,j) = exp(-zeta*num);
            w(j,i) = exp(-zeta*num);
        }
    }
    return wrap(w);
}


arma::vec prox_L2(arma::vec& v, double sigma){
    int n = v.size();
    int i;
    arma::vec prox(n,fill::zeros);
    double sum=0.0;
    for (i=0; i<n; i++){
        sum = sum + pow(v(i),2.0);
    }
    double l2norm=sqrt(sum);
    if (sum==0.0){
        for (i=0; i<n; i++)
            prox(i)=v(i);
    }
    else{
        for(i=0; i<n; i++)
            prox(i)=fmax(0.0, 1.0-(sigma/l2norm))*v(i);
    }
    return prox;
}


int Sign(double a){
    if(a>0){return 1;}
    else if (a<0){return -1;}
    return 0;
}


double soft_thresholding(double a, double b){
    if (b<0){
        return 0.0;
    }else{
        return Sign(a)*fmax(0.0,fabs(a)-b);
    }
}


arma::mat update_U(arma::mat& V, arma::mat& Y, arma::mat& Z, arma::mat& W, arma::mat E){
    int n = W.n_rows;
    arma::mat I(n,n,fill::eye);
    arma::mat Et = trans(E);
    arma::mat sumEtEI = Et*E+I;
    arma::mat invEtEI = inv(sumEtEI);
    // get the final update for U
    arma::mat U = invEtEI*(Et*(V+Y)+W+Z);
    // update U
    return U;
}

// update W

arma::mat update_W(arma::mat& X, arma::mat& U, arma::mat& Z, double tau, double rho){
    int n = X.n_rows, p=X.n_cols;
    int i,j;
    arma::mat W(n,p,fill::zeros);
    double w1, w2;
    for (i=0; i<n; i++){
        for (j=0; j<p; j++){
            if (fabs(rho*(X(i,j)-U(i,j)+Z(i,j))/(1+rho))<=tau){//fix bug
                w1=(X(i,j)+rho*(U(i,j)-Z(i,j)))/(1+rho);
                W(i,j)=w1;
            }
            else{
                w2=X(i,j)-soft_thresholding(X(i,j)-U(i,j)+Z(i,j),tau/rho);
                W(i,j)=w2;
            }
        }
    }
    return W;
}

// update V

arma::mat update_V(arma::mat& U, arma::mat& Y, arma::mat E, arma::vec wt, double lambda, double rho){
    int p=U.n_cols;
    int nK = E.n_rows;
    int j, k;
    arma::mat EU = E*U;
    arma::mat V(nK,p,fill::zeros);
    arma::vec a(p,fill::zeros);
    arma::vec aa(p,fill::zeros);
    for (k=0;k<nK;k++){
        for (j=0;j<p;j++){
            a(j)= EU(k,j)-Y(k,j);
        }
        aa=prox_L2(a,wt(k)*lambda/rho);
        for (j=0;j<p;j++){
            V(k,j)=aa(j);
        }
    }
    return V;
}

//update Z

arma::mat update_Z(arma::mat& Z_old,arma::mat& U, arma::mat& W, double rho){
    int n = U.n_rows, p=U.n_cols;
    int i,j;
    double Z_temp;
    arma::mat Z(n,p,fill::zeros);
    for (i=0; i<n; i++){
        for (j=0;j<p;j++){
            Z_temp = Z_old(i,j)-rho*(U(i,j)-W(i,j));
            Z(i,j) = Z_temp;
        }
    }
    return Z;
}

// update Y

arma::mat update_Y(arma::mat& Y_old, arma::mat& U, arma::mat& V, arma::mat E, double rho){
    int p=U.n_cols, nK = E.n_rows;
    int i, j;
    double y;
    arma::mat EU = E*U;
    arma::mat Y(nK,p,fill::zeros);
    for (i=0; i<nK; i++){
        for (j=0; j<p; j++){
            y=Y_old(i,j);
            y=y-rho*(EU(i,j)-V(i,j));
            Y(i,j)=y;
        }
    }
    return Y;
}

// update s

int update_s(arma::mat V){
    int V_row = V.n_rows, V_col = V.n_cols;
    int i,j;
    int count;
    int s = V_row;
    for (i=0;i<V_row;i++){
        count = 0;
        for (j=0;j<V_col;j++){
            if (V(i,j)==0){
                count++;
            }
        }
        if (count==V_col){
            s = s - 1;
        }
    }
    return s;
}


double tolerance(arma::mat W, arma::mat W_old){
    int n = W.n_rows, p=W.n_cols;
    int i,j;
    double temp;
    double sum=0.;
    for (i=0;i<n;i++){
        for (j=0;j<p;j++){
            temp = fabs(W(i,j)-W_old(i,j));
            sum += pow(temp,2.0); //Changed a little bit
        }
    }
    return sqrt(sum);
}

//Might have issue on adding a U as parameter

arma::mat robust_convex_cluster(arma::mat& X, arma::mat& W, arma::mat& V, arma::mat& Y,
                                arma::mat& Z, arma::mat E, int max_iter, double tol_abs,
                                double lambda, double rho, double tau, arma::vec wt){
    int n = X.n_rows, p=X.n_cols;
    int it;
    //int iter;
    double tol;
    //arma::vec wt1 = as<arma::arma::vec>(wt);
    arma::mat U(n,p,fill::zeros);
    //U = as<arma::arma::mat>(U_temp);

    for (it=0; it<max_iter; it++){
        arma::mat W_old=W;
        arma::mat Y_old=Y;
        arma::mat Z_old=Z;

        U=update_U(V,Y,Z,W,E);
        W=update_W(X,U,Z,tau,rho);
        V=update_V(U,Y,E,wt,lambda,rho);
        Y=update_Y(Y_old,U,V,E,rho);
        Z=update_Z(Z_old,U,W,rho);

        tol = tolerance(W,W_old);
        if (tol<tol_abs*n*p){
            break;
        }
    }
    //if (it > max_iter){
    //    iter = max_iter;
    //}
    //else{
    //    iter = it;
    //}
    return W; //Same as return U
}

//[[Rcpp::export]]
List robust_convex_bicluster(arma::mat& X, arma::mat& W1, arma::mat& W2, arma::mat& V1,
                             arma::mat& V2, arma::mat& Y1, arma::mat& Y2, arma::mat& Z1,
                             arma::mat& Z2, arma::mat E1, arma::mat E2,
                             int max_iter, double tol_abs,
                             double lambda, double rho, arma::vec wt_row, arma::vec wt_col){
    int n = X.n_rows, p=X.n_cols;
    int its,iter;
    double tol;
    int np = n*p;
    double rhs = log(np*np)/np;
    double tau = 1.345 * mad(arma::vectorise(X));

    arma::mat P = zeros(n,p);
    arma::mat Q = zeros(p,n);
    arma::mat U = X;
    arma::mat R = trans(U);
    int max_iter_row = 1000;
    int max_iter_col = 1000;
    int s1, s2, s;

    //main loop
    for (its=0;its<=max_iter;its++){

        arma::mat UTPT = trans(U)+trans(P);
        R = robust_convex_cluster(UTPT,W2,V2,Y2,Z2,E2,max_iter_row,tol_abs,lambda,rho,tau,wt_col);
        P = P + U - trans(R);

        arma::mat RTQT = trans(R)+trans(Q);
        U = robust_convex_cluster(RTQT,W1,V1,Y1,Z1,E1,max_iter_col,tol_abs,lambda,rho,tau,wt_row);
        Q = Q + R - trans(U);

        s1 = update_s(V1);
        s2 = update_s(V2);
        s = std::min(s1,s2);
        rhs = rhs * ((np-s)/np);

        arma::vec res = arma::vectorise(X)-arma::vectorise(U);
        arma::vec resSq = arma::square(res);
        tau = sqrt((long double)rootf1(resSq, np, rhs, min(resSq), quantile(resSq,0.95)));

        tol = tolerance(U,trans(R));

        if (tol < tol_abs*n*p) {
            break;
        }
    }

    if (its >= max_iter) {
        iter = max_iter;
    }
    else {
        iter = its;
    }

    List a = List::create(_["U"]=wrap(U),
                          _["V_row"]=wrap(V1),
                          _["V_col"]=wrap(V2),
                          _["iter"]=wrap(iter),
                          _["tol"]=wrap(tol),
                          _["tau"]=wrap(tau));
    return a;
}


// Non-tuning-free Method

//[[Rcpp::export]]
List robust_convex_bicluster_naive(arma::mat& X, arma::mat& W1, arma::mat& W2, arma::mat& V1,
                                   arma::mat& V2, arma::mat& Y1, arma::mat& Y2, arma::mat& Z1,
                                   arma::mat& Z2, arma::mat E1, arma::mat E2,
                                   int max_iter, double tol_abs,
                                   double lambda, double rho, double tau, arma::vec wt_row, arma::vec wt_col){
    int n = X.n_rows, p=X.n_cols;
    int its,iter;
    double tol;
    arma::mat P = zeros(n,p);
    arma::mat Q = zeros(p,n);
    arma::mat U = X;
    arma::mat R = trans(U);
    int max_iter_row = 1000;
    int max_iter_col = 1000;


    //main loop
    for (its=0;its<=max_iter;its++){

        arma::mat UTPT = trans(U)+trans(P);
        R = robust_convex_cluster(UTPT,W2,V2,Y2,Z2,E2,max_iter_row,tol_abs,lambda,rho,tau,wt_col);
        P = P + U - trans(R);

        arma::mat RTQT = trans(R)+trans(Q);
        U = robust_convex_cluster(RTQT,W1,V1,Y1,Z1,E1,max_iter_col,tol_abs,lambda,rho,tau,wt_row);
        Q = Q + R - trans(U);

        tol = tolerance(U,trans(R));
        if (tol < tol_abs*n*p) {
            break;
        }
    }

    if (its >= max_iter) {
        iter = max_iter;
    }
    else {
        iter = its;
    }

    List a = List::create(_["U"]=wrap(U),
                          _["V_row"]=wrap(V1),
                          _["V_col"]=wrap(V2),
                          _["iter"]=wrap(iter),
                          _["tol"]=wrap(tol));
    return a;
}
