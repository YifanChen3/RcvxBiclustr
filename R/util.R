## Clusterpath preprocessing
vec2tri <- function(k,n){
  i <- ceiling(0.5*(2*n-1 - sqrt((2*n-1)^2 - 8*k)))
  j <- k - n*(i-1) + i*(i-1)/2 + i
  return(as.matrix(cbind(i,j)))
}

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

#' Create adjacency matrix from V
#'
#' \code{create_adjacency_matrix} creates an n-by-n adjacency matrix from the matrix of centroid differences.
#'
#' @param V Matrix of centroid differences
#' @param n Number of observations in clustering
#' @import Matrix
#' @export
create_adjacency_matrix <-function(V,n){
  diff <- apply(V,2,FUN=function(x) {norm(as.matrix(x),'f')})
  connected_ix <- which(diff==0)
  ix <- vec2tri(1:(n*(n-1)/2),n)
  i <- ix[connected_ix,1]
  j <- ix[connected_ix,2]
  A <- Matrix(0, nrow = n, ncol = n, sparse = TRUE)
  A[(j-1)*n + i] <-1
  return(A)
}

#' Find clusters
#'
#' \code{find_clusters_from_adjacency} uses breadth-first search to identify the connected components
#' of the corresponding adjacency graph of the centroid differences vectors.
#'
#' @param A adjacency matrix
#' @import igraph
#' @export
find_clusters_from_adjacency <- function(A){
  G <- graph.adjacency(A, mode = 'upper')
  n <- nrow(A)
  node_seen <- logical(n)
  cluster <- integer(n)
  k <- 1
  for(i in 1:n){
    if (!node_seen[i]){
      connected_set <- graph.bfs(G, root = i, unreachable = FALSE)$order
      node_seen[connected_set] <- TRUE
      cluster[connected_set] <- k
      k <- k+1
    }
  }
  nClusters <- k-1
  size <- integer(nClusters)
  for (j in 1:nClusters) {
    size[j] <- length(which(cluster==j))
  }
  return(list(cluster=cluster, size=size))
}
