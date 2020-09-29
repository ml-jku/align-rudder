An alignment algorithm maximizes the total score $$S$$ and, thereby, aligns events $$i$$ and $$j$$ with probability $$q_{ij}$$.  
High values of $$q_{ij}$$ means that the algorithm often aligns events $$i$$ and $$j$$ using the scoring matrix $$\unicode{x1D54A}$$ with entries $$\unicode{x1D564}_{i,j}$$.  
According to Theorem 2 and Equation \[3\] in [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC53667), asymptotically with the sequence length, we have  
$$\unicode{x1D564}_{i,j} = \frac{\ln(\frac{q_{ij}}{(p_i \ p_j)})}{\lambda^*}$$  
where $$\lambda^*$$ is the single unique positive root of $$\sum_{i=1,j=1}^{n,n} p_i p_j \exp(\lambda \unicode{x1D564}_{i,j}) =1$$ (Equation \[4\] in [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC53667)).  
We now can choose a desired probability $$q_{ij}$$ and then
compute the scoring matrix $$\unicode{x1D54A}$$ with entries $$\unicode{x1D564}_{i,j}$$. High values of $$q_{ij}$$ should indicate relevant events for the strategy.  
A priori, we only know that a relevant event should
be aligned to itself, while we do now know which events are
relevant.  
Therefore we set $$q_{ij}$$ to large values for every $$i=j$$ 
and to low values for $$i\not=j$$.
Concretely, we set
$$q_{ij}=p_i-\epsilon$$ for $$i=j$$ and $$q_{ij}=\epsilon/(n-1)$$ for $$i\not=j$$,
where $$n$$ is the number of different possible events.
Events with smaller $$p_i$$
receive a higher score $$\unicode{x1D564}_{i,i}$$ when aligned to themselves since
this is seldom observed randomly.