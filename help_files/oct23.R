

# task 1 


library(bnlearn)
library(gRain)
# probabilities known

net <- model2network('[U][C|U][A|C][B|C][D|A:B][Ch|U][Ah|Ch][Bh|Ch][Dh|Ah:Bh]')
graphviz.plot(net)


Cpt_u = matrix(c(0.5,0.5), dimnames = list( c('0','1')))
  
Cpt_c = matrix(c(0.9,0.1,0.1,0.9),ncol=2,nrow=2, dimnames = list( 'C'=c('0','1'),'U'=c('0','1')))

Cpt_a = matrix(c(1,0,0.2,0.8),ncol=2,nrow=2, dimnames = list( 'A'=c('0','1'), 'C'=c('0','1')))

Cpt_b = matrix(c(1,0,0.2,0.8),ncol=2,nrow=2, dimnames = list( 'B'=c('0','1'),'C'=c('0','1')))
  
Cpt_d = matrix(c(0.9,0.1,0,1,0,1,0,1),ncol=4,nrow=2 )
dim(Cpt_d)=c(2,2,2)
dimnames(Cpt_d) = list('D'=c('0','1'), 'A'=c('0','1'),'B'=c('0','1'))

Cpt_ch = matrix(c(0.9,0.1,0.1,0.9),ncol=2,nrow=2, dimnames = list( 'Ch'=c('0','1'),'U'=c('0','1')))
  
Cpt_ah = matrix(c(1,0,0.2,0.8),ncol=2,nrow=2, dimnames = list( 'Ah'=c('0','1'), 'Ch'=c('0','1')))
  
Cpt_bh =  matrix(c(1,0,0.2,0.8),ncol=2,nrow=2, dimnames = list( 'Bh'=c('0','1'),'Ch'=c('0','1')))
  
Cpt_dh =  matrix(c(0.9,0.1,0,1,0,1,0,1),ncol=4,nrow=2 )
dim(Cpt_dh)=c(2,2,2)
dimnames(Cpt_dh) = list('Dh'=c('0','1'), 'Ah'=c('0','1'),'Bh'=c('0','1'))
  
bn <- custom.fit(net, dist=list(U=Cpt_u, C=Cpt_c,A=Cpt_a,B=Cpt_b, D=Cpt_d,Ch=Cpt_ch,Ah=Cpt_ah, Bh=Cpt_bh, Dh=Cpt_dh))
bn <- compile(as.grain(bn))                
                 
querygrain(setEvidence(bn,nodes = c("D","Ah"), states=c("1","0")),c("Dh"))             

# the probability of being dead in the hypothetical world given dead in real and A did not shot is 0.379.




# task 2

library(HMM)

# Vector with the names of the states.
states <- as.character(c(1.1,1.2,2.1,2.2,2.3,3.1,3.2,4.1,5.1,5.2))

# Vector with the names of the symbols.
symbols <- 1:5

# Vector with the starting probabilities of the states.
startProbs <- rep(0.1, 10)

a=1/2

transProbs <-  matrix(c(0,1,0,0,0,0,0,0,0,0,
                        0,a,a,0,0,0,0,0,0,0,
                        0,0,0,1,0,0,0,0,0,0,
                        0,0,0,0,1,0,0,0,0,0,
                        0,0,0,0,a,a,0,0,0,0,
                        0,0,0,0,0,0,1,0,0,0,
                        0,0,0,0,0,0,a,a,0,0,
                        0,0,0,0,0,0,0,a,a,0,
                        0,0,0,0,0,0,0,0,0,1,
                        a,0,0,0,0,0,0,0,0,a),nrow=10, ncol=10, byrow = TRUE) # empty matrix

# Stochastic matrix containing the emission probabilities of the states.
# Output of the robot at each state
emissionProbs <- matrix(c(1/3,1/3,0,0,1/3,
                          1/3,1/3,0,0,1/3,
                          1/3,1/3,1/3,0,0,
                          1/3,1/3,1/3,0,0,
                          1/3,1/3,1/3,0,0,
                          0,1/3,1/3,1/3,0,
                          0,1/3,1/3,1/3,0,
                          0,1/3,1/3,1/3,1/3,
                          1/3,0,0,1/3,1/3,
                          1/3,0,0,1/3,1/3
                          ),nrow=10, ncol=5, byrow = TRUE) # empty matrix


hmm <- initHMM(States = states,
               Symbols = symbols,
               startProbs = startProbs,
               transProbs = transProbs,
               emissionProbs = emissionProbs)


sims <- simHMM(hmm, 100)

sims

# Task 3


V <- rep(0,10)
pi <- rep(0,10)
gamma = 0.95
theta=0.1
S <- V[-10]
V[10] <- 1

repeat{
  delta = 0
  for(s in 1:9){
    
    v <- V[s]
    
    V[s] <- max(gamma*V[s],  gamma*V[s+1])
    
    delta = max(delta, abs(v-V[s]))
  }
  if (delta < theta){
    break
  }

}


for(s in 1:9){
  
  pi[s] <- which.max(c(gamma*V[s], gamma*V[s+1]) )
  
}
  
V
pi



