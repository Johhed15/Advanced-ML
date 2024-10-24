# exam oct 2022

# task 1


library(bnlearn)
library(gRain)
library(knitr)


data = data('asia')
dag = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")


# est first ten
train <- asia
fit <- bn.fit(dag, train[1:10,], method='bayes')
D10 <- fit$D # D depends on B and E
fitg <- as.grain(fit)
fitg <- compile(fitg)

for (i in 11:5000){
  
  # all nodes except 'S' and setting the current row to the state
  ev <- setEvidence(fitg , nodes = colnames(train[-c(5,6)]), states=as.vector(unlist(train[i,-c(5,6)])))
  
  # marginal prob for S[r]
  qb <- querygrain(ev,node=c('B'))
  # predicting S
  train[i,"B"] <- sample(c("no","yes"),size=1, p=qb$B)
  
  # marginal prob for S[r]
  qe <- querygrain(ev,node=c('E'))

  train[i,"E"] <- sample(c("no","yes"),size=1, p=qe$E) 
  
}

fit <- bn.fit(dag, train, method='bayes')
Dest <- fit$D # D depends on B and E


fit <- bn.fit(dag, asia, method='bayes')
Dtrue <- fit$D # D depends on B and E
Dest
Dtrue
D10




# task 2


a <- 1/2
tpb <-  matrix(c(0,1,0,0,0,0,0,0,0,0,
                 0,a,a,0,0,0,0,0,0,0,
                 0,0,0,1,0,0,0,0,0,0,
                 0,0,0,0,1,0,0,0,0,0,
                 0,0,0,0,a,a,0,0,0,0,
                 0,0,0,0,0,0,1,0,0,0,
                 0,0,0,0,0,0,a,a,0,0,
                 0,0,0,0,0,0,0,a,a,0,
                 0,0,0,0,0,0,0,0,0,1,
                 a,0,0,0,0,0,0,0,0,a),nrow=10, ncol=10, byrow = TRUE) # empty matrix

# Output of the robot at each state
epb <- matrix(c(1/3,1/3,0,0,1/3,
                1/3,1/3,0,0,1/3,
                1/3,1/3,1/3,0,0,
                1/3,1/3,1/3,0,0,
                1/3,1/3,1/3,0,0,
                0,1/3,1/3,1/3,0,
                0,1/3,1/3,1/3,0,
                0,0,1/3,1/3,1/3,
                1/3,0,0,1/3,1/3,
                1/3,0,0,1/3,1/3
),nrow=10, ncol=5, byrow = TRUE) # empty matrix

states= as.character(c(1.1,1.2,2.1,2.2,2.3,3.1,3.2,4.1,5.1,5.2))

symbols=c(1:5)
library(HMM)

startProbs <- rep(1/3,5)
hmm <- initHMM(States = states,
               Symbols = symbols,
               startProbs = startProbs,
               transProbs = tpb,
               emissionProbs = epb)



simHMM(hmm, 100)


# task 3 
library(knitr)
library(ggplot2)
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


# Environment A (learning)
set.seed(12345)
H <- 5
W <- 7

reward_map <- matrix(0, nrow = H, ncol = W)
reward_map[3,6] <- 10
reward_map[2:4,3] <- -1

q_table <- array(0,dim = c(H,W,4))

vis_environment()

for (a in c(0.001,0.01,0.1)){ # looping over alphas
for(i in 1:500){
  foo <- q_learning(start_state = c(3,1), alpha=a,gamma=1)
  
  if(any(i==c(10,500)))
    vis_environment(i,alpha=a,gamma=1)
}}


# as gamma =1 we can see that for alpha =0.1 we get the value 10 for many actions
# as we go for the long term reward. We get the best policy for alpha =0.1 as it 
# has learned much more than 0.01 and 0.001, as they dont update the values that much 
# for each episode, a low learning rate requires more episodes to converge. Optimal path can be found for 0.01 too but the values are not converged.



# task 4


X<-seq(0,10,.1)
Yfun<-function(x){
  return (x*(sin(x)+sin(3*x))+rnorm(length(x),0,2))
}
plot(X,Yfun(X),xlim=c(0,10),ylim=c(-15,15))

SquaredExpKernel <- function(x1,x2,sigmaF=0.5,l=2){
  n1 <- length(x1)
  n2 <- length(x2)
  K <- matrix(NA,n1,n2)
  for (i in 1:n2){
    K[,i] <- sigmaF^2*exp(-0.5*( (x1-x2[i])/l)^2 )
  }
  return(K)
}

# the pattern seems to change within 0.5 so i chose l of 0.5 as they should correlate less 
# of they are further than this distance.






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



result <- posteriorGP(X = X,
                      y = Yfun(X),
                      XStar = seq(0,10,0.01),
                      sigmaNoise = 2,
                      k = SquaredExpKernel,
                      sigmaF = 0.5,
                      l = 0.5)


points_df <- data.frame(
  x =X,
  y = Yfun(X)
)

# Plot data with upper and lower bounds
data <- data.frame(x =  seq(0, 10, 0.01),
                   mean = result$mean,
                   lower = result$mean - 1.96 * sqrt(result$var),
                   upper =  result$mean + 1.96 * sqrt(result$var))

ggplot(data, aes(x = x)) +
  geom_line(aes(y = mean), color = "black") +
  geom_point(data = points_df, aes(x = x, y = y)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey", alpha = 0.4) +
  labs(title = "Posterior Mean and 95% Probability Bands",
       x = "x",
       y = "Posterior Mean of f") +
  theme_bw()

# the variance seems to low for the model as it looks like its not using the data
# that much, could also try to lower l but starts with raising sigmaf
result <- posteriorGP(X = X,
                      y = Yfun(X),
                      XStar = seq(0,10,0.01),
                      sigmaNoise = 2,
                      k = SquaredExpKernel,
                      sigmaF = 1.5,
                      l = 0.5)


points_df <- data.frame(
  x =X,
  y = Yfun(X)
)

# Plot data with upper and lower bounds
data <- data.frame(x =  seq(0, 10, 0.01),
                   mean = result$mean,
                   lower = result$mean - 1.96 * sqrt(result$var+2**2),
                   upper =  result$mean + 1.96 * sqrt(result$var+4))

ggplot(data, aes(x = x)) +
  geom_line(aes(y = mean), color = "black") +
  geom_point(data = points_df, aes(x = x, y = y)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey", alpha = 0.4) +
  labs(title = "Posterior Mean and 95% Probability Bands",
       x = "x",
       y = "Posterior Mean of f") +
  theme_bw()

# covers around 95% of the y values in the data set so looks good. 

# b


X<-seq(0,10,2)
Yfun<-function(x){
  return (x*(sin(x)+sin(3*x))+rnorm(length(x),0,.2))
}
plot(X,Yfun(X),xlim=c(0,10),ylim=c(-15,15))



for (i in 1:9){
result <- posteriorGP(X = X,
                      y = Yfun(X),
                      XStar = seq(0,10,0.01),
                      sigmaNoise = 2,
                      k = SquaredExpKernel,
                      sigmaF = 1.5,
                      l = 0.5)




X <- c(X,0.01*which.max(sqrt(result$var+2**2))) # checking max variance and add a point close to that
# 0.01 is scaling from index to between 0-10

}


points_df <- data.frame(
  x =X,
  y = Yfun(X)
)

# Plot data with upper and lower bounds
data <- data.frame(x =  seq(0, 10, 0.01),
                   mean = result$mean,
                   lower = result$mean - 1.96 * sqrt(result$var+2**2),
                   upper =  result$mean + 1.96 * sqrt(result$var+4))

ggplot(data, aes(x = x)) +
  geom_line(aes(y = mean), color = "black") +
  geom_point(data = points_df, aes(x = x, y = y)) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey", alpha = 0.4) +
  labs(title = "Posterior Mean and 95% Probability Bands",
       x = "x",
       y = "Posterior Mean of f") +
  theme_bw()

# added 9 more points each where we were the most uncertain for that iteration