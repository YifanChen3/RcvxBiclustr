#' Compute Robust weights
#'
#' \code{robust_weights} computes the robust weights given a data matrix \code{X},
#' a scale parameter \code{zeta} and a parameter that controls the weights \code{delta}.
#' Namely, the lth weight \code{w[l]} is given by
#' \deqn{
#' w[l] = exp(-\zeta{\sum_{j\in D_{1}}(X_{i'j}-X_{ij})^2+\sum_{j\in D_{2}}{\delta}^2})
#' , where the lth pair of nodes is (\code{i},\code{i'})
#' and \code{D1={j:|X_{ij}-X_{i'j}|<delta}}, \code{D2={j:|X_{ij}-X_{i'j}|>delta}}.
#' }
#' @param X The data matrix to be clustered. The rows are observations, and the columns
#' are features.
#' @param delta The nonnegative parameter that controls the scale of robust weights
#' when there is outlier(s) in the data.
#' @param zeta The nonnegative parameter that controls the scale of robust weights.
#' @author Yifan Chen, Chuanquan Li, Chunyin Lei
#' @useDynLib RcvxBiclustr
#' @import gdata
#' @export
#' @return A vector \cite{wt} of weights for robust convex clustering.

tri2vec <- function(i,j,n) {
  return(n*(i-1) - i*(i-1)/2 + j -i)
}

knn_weights <- function(w,k,n) {
  i <- 1
  neighbors <- tri2vec(i,(i+1):n,n)
  keep <- neighbors[sort(w[neighbors],decreasing=TRUE,index.return=TRUE)$ix[1:k]]
  for (i in 2:(n-1)) {
    group_A <- tri2vec(i,(i+1):n,n)
    group_B <- tri2vec(1:(i-1),i,n)
    neighbors <- c(group_A,group_B)
    knn <- neighbors[sort(w[neighbors],decreasing=TRUE,index.return=TRUE)$ix[1:k]]
    keep <- union(knn,keep)
  }
  i <- n
  neighbors <- tri2vec(1:(i-1),i,n)
  knn <- neighbors[sort(w[neighbors],decreasing=TRUE,index.return=TRUE)$ix[1:k]]
  keep <- union(knn,keep)
  if (length(keep) > 0)
    w[-keep] <- 0
  return(Matrix(data=w,ncol=1,sparse=TRUE))
}

robust_weights <- function(X, delta, zeta, k){
  n <- nrow(X)
  sol <- robustweights(X=X,delta=delta,zeta=zeta)
  weights <- lowerTriangle(sol)
  weights <- as.numeric(knn_weights(weights,k,n))
  weights <- weights/sqrt(n)
  return(weights)
}
