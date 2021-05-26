args = commandArgs(trailingOnly=TRUE)

library(pgmm)
run_pgmm<-function(x,z,bic,cls,q8,p,G8,N,model,cluster,lambda,psi,TOL=tol){
  p4<-.C("pgmm_c",as.double(x),as.double(z),as.double(bic),as.integer(cls),as.integer(q8),as.integer(p),as.integer(G8),as.integer(N),
         as.integer(model),as.integer(cluster),as.double(lambda),as.double(psi),as.double(TOL),PACKAGE="pgmm")
  list(p4[[2]],p4[[3]],p4[[11]],p4[[12]])
}

x1_temp <- read.table(paste0(args[1],".txt"))
x1_temp <- as.matrix(x1_temp)
x1 <- as.vector(t(x1_temp))

z1_temp <- read.table(paste0(args[2],".txt"))
x1_temp <- as.matrix(z1_temp)
z1 <- as.vector(t(z1_temp))

bic <- as.numeric(args[3])

class_temp <- read.table(paste0(args[4],".txt"))
class_temp <- as.matrix(class_temp)
class <- as.vector(t(class_temp))

q1 <- as.numeric(args[5])

p <- as.numeric(args[6])

g1 <- as.numeric(args[7])

N <- as.numeric(args[8])

modelsubset <- as.numeric(args[9])

class_ind <- as.numeric(args[10])

lam_temp <- read.table(paste0(args[11],".txt"))
lam_temp <- as.matrix(lam_temp)
lam <- as.vector(t(lam_temp))

psi_temp <- read.table(paste0(args[12],".txt"))
psi_temp <- as.matrix(psi_temp)
psi <- as.vector(t(psi_temp))

tol<- as.double(args[13])

output <- run_pgmm(x1,z1,bic,class,q1,p,g1,N,modelsubset,class_ind,lam,psi,tol)

write.table(output[[1]], file=paste0("zbest",g1,q1,modelsubset,".txt"), row.names=FALSE, col.names=FALSE, sep=" ")

write.table(output[[2]], file=paste0("bicbest",g1,q1,modelsubset,".txt"), row.names=FALSE, col.names=FALSE, sep=" ")

write.table(output[[3]], file=paste0("lambdabest",g1,q1,modelsubset,".txt"), row.names=FALSE, col.names=FALSE, sep=" ")

write.table(output[[4]], file=paste0("psibest",g1,q1,modelsubset,".txt"), row.names=FALSE, col.names=FALSE, sep=" ")
