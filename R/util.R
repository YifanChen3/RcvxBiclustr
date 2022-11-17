#' Create E matrix
#'
#' \code{create_E_matrix} generates the edges matrix where the edges are denoted
#' as \code{l=(i,j)}.
#'
#' @param n Number of data observations
#' @return E The edge matrix.
#' @return nK The number of non-zero weights.
#' @export
#' @examples
#' n <- 10
#' E <- create_E_matrix(n)$E
#' nK <- create_E_matrix(n)$nK
create_E_matrix <- function(n){
  for(i in seq(from=n-1,to=1,by=-1)){
    if(i>1){temp <- diag(rep(-1,i))}else{temp <- -1}
    temp1 <- cbind(rep(1,i),temp)
    if(n-i-1==0){
      E <- temp1
    }else{E <- rbind(E,cbind(matrix(0,ncol=n-i-1, nrow=i),temp1))}
  }
  k <- dim(E)[1]
  return(list(E=E, nK=k))
}

