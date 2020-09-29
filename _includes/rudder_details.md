To estimate the final return the LSTM uses return decomposition, which relies on pattern recognition.
On the left side of the figure shown below, we illustrate a simple environment where an agent needs to collect a key to unlock a door and receive the treasure.

![](assets/rudder_details.png)

Such a step-function pattern could be detected by a feed-forward neural network (FNN) as shown in the middle section on the left side.
The issue is that with a FNN the context information of <i>having a key</i> must be re-extracted in every state and training becomes much more difficult.
In contrast, RUDDER uses an LSTM, which is able to store the information of <i>having a key</i> in its hidden state, such that it needs to extract this information only once in the state where the distinct event occurred.
Afterwards this context information is only updated when the lock is opened as shown on the bottom left side. 

In the right section in the figure above we illustrate how the reward is redistributed according to the step-function.
An inherent property of RUDDER is, that by identifying key events and redistributing the reward to them it also pushes the expected future reward towards zero.
This considerably reduces the variance of the remaining return and, hence the estimations become more efficient.
Zeroing the expected future reward and redistributing the reward is mathematically endorsed by introducing the concepts of return-equivalent decision processes, reward redistribution, return decomposition and optimal reward redistribution.
One important contribution established by RUDDER was the introduction of the theoretically founded sequence-Markov decision processes (SDPs), which extend beyond the scope of Markov decision processes (MDPs).
The definition of SDPs was thoroughly explained in the appendix of the RUDDER paper including proofs that show that the optimal policy remains unchanged.
