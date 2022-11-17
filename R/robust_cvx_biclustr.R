#' Robust Convex Clustering
#'
#' \code{robust_cvx_biclustr} performs robust convex biclustering using ADMM. This is an R wrapper function around C code.
#' Dimensions of various arguments are as follows:
#' \itemize{
#' \item{n is the number of data observations}
#' \item{p is the number of features}
#' \item{nK is the number non-zero weights.}
#' }
#'
#' @param X The n-by-p data matrix whose rows are being clustered.
#' @param W The n-by-p matrix of Lagrange multipliers.
#' @param V The centroid difference matrix.
#' @param Y The nK-by-p matrix of Lagrange multipliers.
#' @param Z The n-by-p matrix of Lagrange multipliers.
#' @param max_iter The maximum number of iterations. The default value is 1e5.
#' @param rho Augmented Lagrangian penalty parameter.
#' @param tau The robustification parameter in huber loss.
#' @param lambda The regularization parameter controlling the amount of shrinkage.
#' @param wt A vector of nK positive weights.
#' @param tol_abs The convergence tolerance. The default value is 1e-05.
#'
#' @return \code{U} The centroid matrix.
#' @return \code{W} The centroid matrix.
#' @return \code{V} The centroid difference matrix.
#' @return \code{Y} The Lagrange multiplier matrix.
#' @return \code{Z} The Lagrange multiplier matrix.
#' @return \code{iter} The number of iterations taken.
#' @return \code{tol} The residual tolerances.
#' @export
#' @author  Yifan Chen, Chuanquan Li, Chunyin Lei
#' @useDynLib RcvxBiclustr
#' @examples
#' ## Create random problems
#' library(RcvxBiclustr)
#' set.seed(111)
#' random.num <- seq(-5,5,0.5)
#' random.mean <- sample(random.num,16,replace = FALSE)
#' C <- NULL
#' for (i in c(1:4)){
#'   R <- NULL
#'   for (j in c(1:4)){
#'     m <- matrix(rnorm(625,random.mean[4*(i-1)+j],2),25,25)
#'     R <- rbind(R,m)
#'   }
#'   C <- cbind(C,R)
#' }
#' X <- C
#' noise.t <- matrix(rt(10000,2),100,100)
#' X <- X + noise.t
#' p <- 100
#' n <- 100
#' wt_row <- robust_weights(X,8,0.001)
#' wt_col <- robust_weights(t(X),8,0.001)
#' rho <- 0.34
#' a <- robust_cvx_biclustr(X,rho=rho,lambda=1.2,wt_row = wt_row,wt_col = wt_col)
#' image(a$U,zlim=c(-4,4))





robust_cvx_biclustr <- function(X,W1=NULL,W2=NULL,V1=NULL,V2=NULL,Y1=NULL,Y2=NULL,Z1=NULL,Z2=NULL,
                                max_iter=700, rho,lambda,wt_row,wt_col,tol_abs=1e-05){
  n <- as.integer(nrow(X))
  p <- as.integer(ncol(X))
  E1 <- create_E_matrix(n)$E
  E2 <- create_E_matrix(p)$E
  nK1 <- nrow(E1)
  nK2 <- nrow(E2)
  if(is.null(V1)==TRUE){
    V1 <- matrix(1,nrow=nK1,ncol=p)
    #V <- matrix(rnorm(nK*p),nrow=nK,ncol=p)
  }
  if(is.null(W1)==TRUE){
    W1 <- matrix(1,nrow=n,ncol=p)
    #W <- matrix(rnorm(n*p),nrow=n,ncol=p)
  }
  if(is.null(Y1)==TRUE){
    Y1 <- matrix(0,nrow=nK1,ncol=p)
    #Y <- matrix(rnorm(nK*p),nrow=nK,ncol=p)
  }
  if(is.null(Z1)==TRUE){
    Z1 <- matrix(0,nrow=n,ncol=p)
    #Z <- matrix(rnorm(n*p),nrow=n,ncol=p)
  }
  if(is.null(V2)==TRUE){
    V2 <- matrix(1,nrow=nK2,ncol=n)
    #V <- matrix(rnorm(nK*p),nrow=nK,ncol=p)
  }
  if(is.null(W2)==TRUE){
    W2 <- matrix(1,nrow=p,ncol=n)
    #W <- matrix(rnorm(n*p),nrow=n,ncol=p)
  }
  if(is.null(Y2)==TRUE){
    Y2 <- matrix(0,nrow=nK2,ncol=n)
    #Y <- matrix(rnorm(nK*p),nrow=nK,ncol=p)
  }
  if(is.null(Z2)==TRUE){
    Z2 <- matrix(0,nrow=p,ncol=n)
    #Z <- matrix(rnorm(n*p),nrow=n,ncol=p)
  }
  sol = robust_convex_bicluster(X=X,W1=W1,W2=W2,V1=V1,V2=V2,Y1=Y1,Y2=Y2,Z1=Z1,Z2=Z2,
                                E1=E1,E2=E2,max_iter=max_iter,tol_abs=tol_abs,
                                lambda=lambda,rho=rho,wt_row=wt_row,wt_col=wt_col)
  return(list(U=sol$U,V_row=sol$V_row,V_col=sol$V_col,iter=sol$iter,tol=sol$tol,tau=sol$tau))
}



robust_cvx_biclustr_naive <- function(X,W1=NULL,W2=NULL,V1=NULL,V2=NULL,Y1=NULL,Y2=NULL,Z1=NULL,Z2=NULL,
                                      max_iter=700,rho,tau,lambda,wt_row,wt_col,tol_abs=1e-05){
  n <- as.integer(nrow(X))
  p <- as.integer(ncol(X))
  E1 <- create_E_matrix(n)$E
  E2 <- create_E_matrix(p)$E
  nK1 <- nrow(E1)
  nK2 <- nrow(E2)
  if(is.null(V1)==TRUE){
    V1 <- matrix(1,nrow=nK1,ncol=p)
    #V <- matrix(rnorm(nK*p),nrow=nK,ncol=p)
  }
  if(is.null(W1)==TRUE){
    W1 <- matrix(1,nrow=n,ncol=p)
    #W <- matrix(rnorm(n*p),nrow=n,ncol=p)
  }
  if(is.null(Y1)==TRUE){
    Y1 <- matrix(0,nrow=nK1,ncol=p)
    #Y <- matrix(rnorm(nK*p),nrow=nK,ncol=p)
  }
  if(is.null(Z1)==TRUE){
    Z1 <- matrix(0,nrow=n,ncol=p)
    #Z <- matrix(rnorm(n*p),nrow=n,ncol=p)
  }
  if(is.null(V2)==TRUE){
    V2 <- matrix(1,nrow=nK2,ncol=n)
    #V <- matrix(rnorm(nK*p),nrow=nK,ncol=p)
  }
  if(is.null(W2)==TRUE){
    W2 <- matrix(1,nrow=p,ncol=n)
    #W <- matrix(rnorm(n*p),nrow=n,ncol=p)
  }
  if(is.null(Y2)==TRUE){
    Y2 <- matrix(0,nrow=nK2,ncol=n)
    #Y <- matrix(rnorm(nK*p),nrow=nK,ncol=p)
  }
  if(is.null(Z2)==TRUE){
    Z2 <- matrix(0,nrow=p,ncol=n)
    #Z <- matrix(rnorm(n*p),nrow=n,ncol=p)
  }
  sol = robust_convex_bicluster_naive(X=X,W1=W1,W2=W2,V1=V1,V2=V2,Y1=Y1,Y2=Y2,Z1=Z1,Z2=Z2,
                                      E1=E1,E2=E2,max_iter=max_iter,tol_abs=tol_abs,
                                      lambda=lambda,rho=rho,tau=tau,wt_row=wt_row,wt_col=wt_col)
  return(list(U=sol$U,V_row=sol$V_row,V_col=sol$V_col,iter=sol$iter,tol=sol$tol))
}
