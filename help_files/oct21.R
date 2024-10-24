# Oct 21




# task 1

data('asia')
set.seed(12345)
dag = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]") # true dag
plot(dag)
hc <- bn.fit(dag,asia,method = 'bayes')
samp <- matrix(0,ncol=8,nrow = 1000)

for (i in 1:1000){
  
  A <- sample(c('no', 'yes'),1,p=hc$A$prob)
  S <- sample(c('no', 'yes'),1,p=hc$S$prob)
  
  T_ <- sample(c('no', 'yes'),1,p=hc$T$prob[,A])
  B <- sample(c('no', 'yes'),1,p=hc$B$prob[,S])
  L <- sample(c('no', 'yes'),1,p=hc$L$prob[,S])
  
  E <- sample(c('no', 'yes'),1,p=hc$E$prob[,L,T_])
  D  <- sample(c('no', 'yes'),1,p=hc$D$prob[,B,E])
  
  X<- sample(c('no', 'yes'),1,p=hc$X$prob[,E])
  
  samp[i,] <- c(A,S,T_,L,B,E,X,D)
  
}

table(samp[samp[,8]=='yes',2])/sum(table(samp[samp[,8]=='yes',2]))

prop.table(table(samp[samp[,8]=='yes',2]))

hc <- bn.fit(dag,asia,method = 'bayes')

fit <- as.grain(hc)
fit <- compile(fit)
# all nodes except 'S' and setting the current row to the state
ev <- setEvidence(fit , nodes = 'D', states='yes')

# marginal prob for S[r]
qg <- querygrain(ev,node='S', type='marginal') # posterior distribution (remove type = marginal for that)
qg



# task 2

library(HMM)

# Other ways of doing the prob matrices
a <- 1/2
tpb <-  matrix(c(0.9,0.1,0,
                 0,0,1,
                 0.2,0,0.8
                 ),nrow=3, ncol=3, byrow = TRUE) # empty matrix
states=c('Healty', 'Sick_day1', 'Sick_day2+')
sym <- c('Healty', 'Sick')
# Stochastic matrix containing the emission probabilities of the states.
# Output of the robot at each state
epb <- matrix(c(0.6,0.4,
                0.3,0.7,
                0.3,0.7
),nrow=3, ncol=2, byrow = TRUE) # empty matrix

startProbs <- c(0.5,0.5,0)

hmm <- initHMM(States = states,
               Symbols = sym,
               startProbs = startProbs,
               transProbs = tpb,
               emissionProbs = epb)


simHMM(hmm, 100)





# 3

# given code 
arrows <- c("^", ">", "v", "<")
action_deltas <- list(c(1,0), # up
                      c(0,1), # right
                      c(-1,0), # down
                      c(0,-1)) # left


vis_environment <- function(iterations=0, epsilon = 0.5, alpha = 0.1, gamma = 0.95, beta = 0){
  
  # Visualize an environment with rewards. 
  # Q-values for all actions are displayed on the edges of each tile.
  # The (greedy) policy for each state is also displayed.
  # 
  # Args:
  #   iterations, epsilon, alpha, gamma, beta (optional): for the figure title.
  #   reward_map (global variable): a HxW array containing the reward given at each state.
  #   q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
  #   H, W (global variables): environment dimensions.
  
  df <- expand.grid(x=1:H,y=1:W)
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,1],NA),df$x,df$y)
  df$val1 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,2],NA),df$x,df$y)
  df$val2 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,3],NA),df$x,df$y)
  df$val3 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,4],NA),df$x,df$y)
  df$val4 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) 
    ifelse(reward_map[x,y] == 0,arrows[GreedyPolicy(x,y)],reward_map[x,y]),df$x,df$y)
  df$val5 <- as.vector(foo)
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,max(q_table[x,y,]),
                                     ifelse(reward_map[x,y]<0,NA,reward_map[x,y])),df$x,df$y)
  df$val6 <- as.vector(foo)
  
  print(ggplot(df,aes(x = y,y = x)) +
          scale_fill_gradient(low = "white", high = "green", na.value = "red", name = "") +
          geom_tile(aes(fill=val6)) +
          geom_text(aes(label = val1),size = 2.5,nudge_y = .35,na.rm = TRUE) +
          geom_text(aes(label = val2),size = 2.5,nudge_x = .35,na.rm = TRUE) +
          geom_text(aes(label = val3),size = 2.5,nudge_y = -.35,na.rm = TRUE) +
          geom_text(aes(label = val4),size = 2.5,nudge_x = -.35,na.rm = TRUE) +
          geom_text(aes(label = val5),size = 7.5) +
          geom_tile(fill = 'transparent', colour = 'black') + 
          ggtitle(paste("Q-table after ",iterations," iterations\n",
                        "(epsilon = ",epsilon,", alpha = ",alpha,"gamma = ",gamma,", beta = ",beta,")")) +
          theme(plot.title = element_text(hjust = 0.5)) +
          scale_x_continuous(breaks = c(1:W),labels = c(1:W)) +
          scale_y_continuous(breaks = c(1:H),labels = c(1:H)))
  
}


transition_model <- function(x, y, action, beta){
  
  # Computes the new state after given action is taken. The agent will follow the action 
  # with probability (1-beta) and slip to the right or left with probability beta/2 each.
  # 
  # Args:
  #   x, y: state coordinates.
  #   action: which action the agent takes (in {1,2,3,4}).
  #   beta: probability of the agent slipping to the side when trying to move.
  #   H, W (global variables): environment dimensions.
  # 
  # Returns:
  #   The new state after the action has been taken.
  
  delta <- sample(-1:1, size = 1, prob = c(0.5*beta,1-beta,0.5*beta))
  final_action <- ((action + delta + 3) %% 4) + 1
  foo <- c(x,y) + unlist(action_deltas[final_action])
  foo <- pmax(c(1,1),pmin(foo,c(H,W)))
  
  return (foo)
}
# greedy function
GreedyPolicy <- function(x, y){
  
  # Get a greedy action for state (x,y) from q_table.
  #
  # Args:
  #   x, y: state coordinates.
  #   q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
  # 
  # Returns:
  #   An action, i.e. integer in {1,2,3,4}.
  
  # Your code here.
  max_actions <- which(q_table[x,y,]==max(q_table[x,y,])) # picking out all max values
  
  max_actions[sample(seq_along(max_actions),1)]# Random if two actions are equally good in the table
  #sample(max_actions,1) Why does this choose 1 more often?!
}

# Epsilon greedy function
EpsilonGreedyPolicy <- function(x, y, epsilon){
  
  # Get an epsilon-greedy action for state (x,y) from q_table.
  #
  # Args:
  #   x, y: state coordinates.
  #   epsilon: probability of acting randomly.
  # 
  # Returns:
  #   An action, i.e. integer in {1,2,3,4}.
  
  # Your code here.
  if (runif(1)< 1-epsilon){
    GreedyPolicy(x,y) # best known action
  }
  else{
    sample(c(1,2,3,4),1) # random action
  }
  
}



q_learning <- function(start_state, epsilon = 0.5, alpha = 0.1, gamma = 0.95, 
                       beta = 0){
  
  # Perform one episode of Q-learning. The agent should move around in the 
  # environment using the given transition model and update the Q-table.
  # The episode ends when the agent reaches a terminal state.
  # 
  # Args:
  #   start_state: array with two entries, describing the starting position of the agent.
  #   epsilon (optional): probability of acting randomly.
  #   alpha (optional): learning rate.
  #   gamma (optional): discount factor.
  #   beta (optional): slipping factor.
  #   reward_map (global variable): a HxW array containing the reward given at each state.
  #   q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
  # 
  # Returns:
  #   reward: reward received in the episode.
  #   correction: sum of the temporal difference correction terms over the episode.
  #   q_table (global variable): Recall that R passes arguments by value. So, q_table being
  #   a global variable can be modified with the superassigment operator <<-.
  
  # Your code here.
  
  x <- start_state[1]
  y <- start_state[2]
  episode_correction <- c()
  repeat{
    # Follow policy, execute action, get reward.
    
    action <-  EpsilonGreedyPolicy(x,y,epsilon)
    new_pos <- transition_model(x,y,action,beta)
    reward <- reward_map[new_pos[1], new_pos[2]]
    
    
    
    # correction
    corr <-(reward + gamma *max(q_table[new_pos[1],new_pos[2],])-q_table[x,y,action] )
    # Q-table update.
    q_table[x,y,action] <<- q_table[x,y,action] + alpha*corr
    # sum of corrections
    episode_correction <- c(episode_correction , corr)
    x <- new_pos[1]
    y <- new_pos[2]
    
    if(reward!=0)
      # End episode.
      return (c(reward,sum(episode_correction)))
  }
  
}




H <- 7
W <- 8

reward_map <- matrix(0, nrow = H, ncol = W)
reward_map[1,] <- -1
reward_map[7,] <- -1
reward_map[4,5] <- 5
reward_map[4,8] <- 10

q_table <- array(0,dim = c(H,W,4))

vis_environment()

res <- matrix(0,ncol=3, nrow=3)

for(j in c(0.5,0.75,0.95)){
  for(e in c(0.1,0.25,0.5)){
    q_table <- array(0,dim = c(H,W,4))
    reward <- NULL
  
  for(i in 1:30000){
     q_learning(epsilon=e ,gamma = j,start_state = c(4,1))
  }
  for (grek in 1:1000) {
    foo <- q_learning(epsilon=e ,gamma = j, start_state = c(4,1), alpha=0)
    reward <- c(reward,foo[1])
    }
   print(paste('gamma',j,'epsilon', e,'reward',mean(reward)))
   vis_environment(i, gamma = j, epsilon=e)
  }
}
# epsilon 0.5 and gamma = 0.95 gave the highest average reward for the validation data

mean(4,5)



# task 4


library(ggplot2)
library(vctrs)
library(knitr)
library(pracma)
library(kernlab)
x <- c(-1,-0.6,-0.2,0.4,0.8)
y <- c(0.768, -0.044, -0.940,0.719, -0.664)



SquaredExpKernel <- function(x1,x2,sigmaF=1,l=3){
  n1 <- length(x1)
  n2 <- length(x2)
  K <- matrix(NA,n1,n2)
  for (i in 1:n2){
    K[,i] <- sigmaF^2*exp(-0.5*( (x1-x2[i])/l)^2 )
  }
  return(K)
}


posteriorGP <- function(X, y, XStar, sigmaNoise, k, sigmaF=1, l=3) {
  
  # 2
  n <- length(y)
  K <- k(X, X, sigmaF = sigmaF, l = l)
  L <- t(chol(K + sigmaNoise^2 * diag(nrow(K))))
  alpha <- solve(t(L), solve(L, y))
  
  # 4
  k_star <-  k(X, XStar, sigmaF = sigmaF, l = l)
  f_star <- t(k_star) %*% alpha
  v <- solve(L, k_star)
  
  # 6
  var_f_star <-  k(XStar, XStar, sigmaF = sigmaF, l = l) - (t(v) %*% v)
  #logmar <- -0.5*(t(y)%*%a)-sum(log(diag(L)))-(n/2)*log(2*pi)
  # validation instead of logmar can be problematic as they are not iid
  # last values do not come from same pdf as training data. Can predict n+1 and train again to pred n+2
  return(list("mean" = f_star, "var" = diag(var_f_star)))
  
} 


result <- posteriorGP(X = c(-1,-0.6,-0.2,0.4,0.8),
                      y = c(0.768, -0.044, -0.940,0.719, -0.664),
                      XStar = seq(-1,1,0.01), sigmaNoise = 1,
                      k = SquaredExpKernel, sigmaF = 1, l = 0.3)

points_df <- data.frame(
  x = c(-1, -0.6, -0.2, 0.4, 0.8),
  y = c(0.768, -0.044, -0.940, 0.719, -0.664)
)

# Plot data with upper and lower bounds
data <- data.frame(x =  seq(-1, 1, 0.01),
                   mean = result$mean,
                   lower = result$mean - 1.96 * sqrt(result$var), # change to
                   # sqrt(results$var + sigma2^2) for prob band for Y
                   upper =  result$mean + 1.96 * sqrt(result$var))

ggplot(data, aes(x = x)) +
  geom_line(aes(y = mean), color = "black") +
  geom_point(data = points_df, aes(x = x, y = y)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey", alpha = 0.4) +
  labs(title = "Posterior Mean and 95% Probability Bands",
       x = "x",
       y = "Posterior Mean of f") +
  theme_bw()

result2 <- posteriorGP(X = c(-1,-0.6,-0.2,0.4,0.8),
                      y = c(0.768, -0.044, -0.940,0.719, -0.664),
                      XStar = seq(-1,1,0.01), sigmaNoise = 0.1,
                      k = SquaredExpKernel, sigmaF = 1, l = 0.3)

points_df <- data.frame(
  x = c(-1, -0.6, -0.2, 0.4, 0.8),
  y = c(0.768, -0.044, -0.940, 0.719, -0.664)
)

# Plot data with upper and lower bounds
data <- data.frame(x =  seq(-1, 1, 0.01),
                   mean = result$mean,
                   lower = result$mean - 1.96 * sqrt(result$var), # change to
                   # sqrt(results$var + sigma2^2) for prob band for Y
                   upper =  result$mean + 1.96 * sqrt(result$var)
                   ,
                   mean2 = result2$mean,
                   lower2 = result2$mean - 1.96 * sqrt(result2$var), # change to
                   # sqrt(results$var + sigma2^2) for prob band for Y
                   upper2 =  result2$mean + 1.96 * sqrt(result2$var))

ggplot(data, aes(x = x)) +
  geom_line(aes(y = mean), color = "black") +
  geom_line(aes(y = mean2), color = "blue") +
  geom_point(data = points_df, aes(x = x, y = y)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey", alpha = 0.4) +
  geom_ribbon(aes(ymin = lower2, ymax = upper2), fill = "red", alpha = 0.4) +
  labs(title = "Posterior Mean and 95% Probability Bands",
       x = "x",
       y = "Posterior Mean of f") +
  theme_bw()





result <- posteriorGP(X = c(-1,-0.6,-0.2,0.4,0.8),
                      y = c(0.768, -0.044, -0.940,0.719, -0.664),
                      XStar = seq(-1,1,0.01), sigmaNoise = 1,
                      k = SquaredExpKernel, sigmaF = 1, l = 1)

points_df <- data.frame(
  x = c(-1, -0.6, -0.2, 0.4, 0.8),
  y = c(0.768, -0.044, -0.940, 0.719, -0.664)
)

# Plot data with upper and lower bounds
data <- data.frame(x =  seq(-1, 1, 0.01),
                   mean = result$mean,
                   lower = result$mean - 1.96 * sqrt(result$var), # change to
                   # sqrt(results$var + sigma2^2) for prob band for Y
                   upper =  result$mean + 1.96 * sqrt(result$var))

ggplot(data, aes(x = x)) +
  geom_line(aes(y = mean), color = "black") +
  geom_point(data = points_df, aes(x = x, y = y)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey", alpha = 0.4) +
  labs(title = "Posterior Mean and 95% Probability Bands",
       x = "x",
       y = "Posterior Mean of f") +
  theme_bw()

result2 <- posteriorGP(X = c(-1,-0.6,-0.2,0.4,0.8),
                       y = c(0.768, -0.044, -0.940,0.719, -0.664),
                       XStar = seq(-1,1,0.01), sigmaNoise = 0.1,
                       k = SquaredExpKernel, sigmaF = 1, l = 1)

points_df <- data.frame(
  x = c(-1, -0.6, -0.2, 0.4, 0.8),
  y = c(0.768, -0.044, -0.940, 0.719, -0.664)
)

# Plot data with upper and lower bounds
data <- data.frame(x =  seq(-1, 1, 0.01),
                   mean = result$mean,
                   lower = result$mean - 1.96 * sqrt(result$var), # change to
                   # sqrt(results$var + sigma2^2) for prob band for Y
                   upper =  result$mean + 1.96 * sqrt(result$var)
                   ,
                   mean2 = result2$mean,
                   lower2 = result2$mean - 1.96 * sqrt(result2$var), # change to
                   # sqrt(results$var + sigma2^2) for prob band for Y
                   upper2 =  result2$mean + 1.96 * sqrt(result2$var))

ggplot(data, aes(x = x)) +
  geom_line(aes(y = mean), color = "black") +
  geom_line(aes(y = mean2), color = "blue") +
  geom_point(data = points_df, aes(x = x, y = y)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey", alpha = 0.4) +
  geom_ribbon(aes(ymin = lower2, ymax = upper2), fill = "red", alpha = 0.4) +
  labs(title = "Posterior Mean and 95% Probability Bands",
       x = "x",
       y = "Posterior Mean of f") +
  theme_bw()
