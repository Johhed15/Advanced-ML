# Advanced-ML
Labs from Advanced Machine Learning course


## LAB 1

Graphical models(Bayesian networks)

## LAB 2

Hidden Markov Models(HMM)

## LAB 3

Reinforcement learning(Q-learning)

## LAB 4 


# Example of how to set up a Robot that have 50% probability to stay or 50% to move one step forward, with 20 % uncertanty of the current state[i-2,i,i+2].


```R
# States 
states <- as.character(1:10)
# Movements for the robot
sym <- paste0('State_',states)
# starting probabilities, equal for all states
startprobs <- rep(1/10,10)

# Transitions probabilities 1/2 probability to stay or change

t_prob <- matrix(0,nrow=10,ncol=10) # empty matrix
diag(t_prob) <- 0.5  # 0.5 to stay
diag(t_prob[,-1]) <- 0.5 # 0.5 to move to NEXT sector
t_prob[10,1] <- 0.5 # left bottom corner(step from 10 to 1)

# emissionprobs
e_prob <- matrix(0,nrow=10, ncol=10) # empty matrix

for (i  in 1:10) { # filling the matrix with probabilities, each row sums to 1
  j <- (i-2):(i+2)
  
  # if we are going over 10 then go back to 1 and upwards
    j[j>10] <- j[j>10] - 10    
  
  # if we are getting a value under 1 then go from 10 and downwards
    j[j<1] <- j[j<1] + 10  
    
  e_prob[i,j] <- 1/5
}

hmm <-initHMM(states,sym, startprobs,transProbs=t_prob,e_prob)

rownames(t_prob) <- sym
colnames(t_prob) <- states
rownames(e_prob) <- sym
colnames(e_prob) <- states
knitr::kable(t_prob, caption='Transitions probabilities')


``` 


<br>
<br>
<div align="center">
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExY3F5N3dub2N2NWRtYml6MnZwMnZ5ZmNnZzZ2YXE2eGxtZXJrMjRydyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/VGC01roisedLNzwyO0/giphy.webp" width="600" height="600"/>
</div>
