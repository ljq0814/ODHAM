####### step 0: computing the robust mean and covariance
robust_esti <- function(X, d){
    p <- ncol(X)
    arr <- array(0,c(d,d,p))
    meanmat <- matrix(0,d,p)
    for (j in 1:p){
        bsm <- bs(X[,j],degree=d)
        ls <- robust::covRob(bsm)
        arr[,,j] <- ls$cov
        meanmat[,j] <- ls$center
    }
    return(list(meanmat = meanmat, covarr = arr))
}

compute_bs <- function(X,d){
    n <- nrow(X); p <- ncol(X)
    res <- array(0,c(n,d,p))
    for (j in 1:p){
        nout = 0.05*n
        bsm <- bs(X[,j], degree = d, Boundary.knots = range(X[(nout+1):n,j]))
        res[,,j] <- bsm
    }
    return(res)
}

#### step 1: find the clean set based on the intersect of max and min statitics
fun_cleanset=function(X,Y,d,n_subset,h_sub){
    n <- nrow(X)
    p <- ncol(X)
    T1 <- fun_maxmin(X,Y,d,n_subset)
    Tmax <- T1[,1]
    Tmin <- T1[,2]
    clean_max <- order(Tmax,decreasing=FALSE)[1:floor(n*h_sub)]
    clean_min <- order(Tmin,decreasing=FALSE)[1:floor(n*h_sub)]
    cleanset <- intersect(clean_max,clean_min)
    return(cleanset)
}

####### step 3:  testing for non clean set 
fun_test <- function(X,Y,d,n_subset,h_sub,B,alpha){
    n <- nrow(X); p <- ncol(X)

    clean_index1 <- fun_cleanset(X,Y,d,n_subset,h_sub)
    X_clean <- X[clean_index1,]
    Y_clean <- matrix(Y[clean_index1,],ncol=1)
    ### compute the empirical cdf for test statistics
    TT <- fun_threshold(X_clean,Y_clean,d,B)
    Fhat1 <- ecdf(TT)

    ### test for sample index2
    inf_pot <- setdiff(1:n,clean_index1)
    X2 <- X[inf_pot,]
    Y2 <- matrix(Y[inf_pot,],ncol=1)
    n2 <- dim(X2)[1]
    
    T_test2 <- rep(0,n2)
    T_test2 <- fun_T_test(X2,Y2,X_clean,Y_clean,TT,p,d)

    phat2 <- 1 - Fhat1(T_test2)
    phat2_sort <- sort.int(phat2,index.return=TRUE)  
    S_index2 <- phat2_sort$ix
    dp2 <- phat2_sort$x - alpha*c(1:n2)/n2
    In2 <- which(dp2<=0) 
    if (length(In2)==0){
        clean_index2 <- 1:n2
        inf_index2=setdiff(1:n2, clean_index2)
    }
    else{ 
        rin2=max(In2)  
        inf_index2=S_index2[1:rin2]
    }
    inf_final=inf_pot[inf_index2]
    return(list(inf_final=inf_final))
}

#### step 1: find the clean set based on the intersect of max and min statitics
fun_cleanset_new=function(Xbs,Y,n_subset,h_sub){
    n <- dim(Xbs)[1]
    p <- dim(Xbs)[3]
    T1 <- fun_maxmin_new(Xbs,Y,n_subset)
    Tmax = T1[,1]
    Tmin = T1[,2]
    clean_max = order(Tmax,decreasing=FALSE)[1:floor(n*h_sub)]
    clean_min = order(Tmin,decreasing=FALSE)[1:floor(n*h_sub)]
    cleanset <- intersect(clean_max,clean_min)
    return(cleanset)
}
####### step 3:  testing for  non clean set 
fun_test_new <- function(X,Y,d,n_subset,h_sub,B,alpha){
    n <- nrow(X); p <- ncol(X)   
    Xbs <- compute_bs(X,d)
    
    clean_index1 <- fun_cleanset_new(Xbs,Y,n_subset,h_sub)
    Xbs_clean <- Xbs[clean_index1,,]
    Y_clean <- matrix(Y[clean_index1,],ncol=1)
    ### compute the empirical cdf for test statistics
    TT <- fun_threshold_new(Xbs_clean,Y_clean,B)
    Fhat1 <- ecdf(TT)
    
    ### test for sample index2
    inf_pot <- setdiff(1:n,clean_index1)
    Xbs2 <- Xbs[inf_pot,,]
    Y2 <- matrix(Y[inf_pot,],ncol=1)
    n2 <- dim(Xbs2)[1]
    
    T_test2 <- rep(0,n2)
    T_test2 <- fun_T_test_new(Xbs2,Y2,Xbs_clean,Y_clean,TT)

    phat2 <- 1 - Fhat1(T_test2)
    phat2_sort <- sort.int(phat2,index.return=TRUE)  
    S_index2 <- phat2_sort$ix
    dp2 <- phat2_sort$x - alpha*c(1:n2)/n2
    In2 <- which(dp2 <= 0) 
    if (length(In2) == 0){
        clean_index2 <- 1:n2
        inf_index2 = setdiff(1:n2, clean_index2)
    }
    else{ 
        rin2 = max(In2)  
        inf_index2 = S_index2[1:rin2]
    }
    inf_final = inf_pot[inf_index2]
    return(list(inf_final = inf_final))
}
