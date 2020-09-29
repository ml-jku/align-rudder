If $$\tau_t=e_{0:t}$$ is the prefix sequence of $$\tau$$ of length $$t+1$$,
then the reward redistribution is  
$$R_{t+1} = \left( S(\tau_t) - S(\tau_{t-1}) \right) \ C
    \ = \ g((s,a)_{0:t}) - g((s,a)_{0:t-1})$$,  
$$R_{T+2} \ = \ \tilde{G}_0 \ - \ \sum_{t=0}^T R_{t+1}$$,  
$$C = \frac{E_{\mathrm{demo}} \left[\tilde{G}_0 \right]}{E_{\mathrm{demo}} \left[ \sum_{t=0}^T S(\tau_t) \ - \ S(\tau_{t-1}) \right] }$$  
where $$\tilde{G}_0=\sum_{t=0}^T\tilde{R}_{t+1}$$ 
is the original return of the sequence $$\tau$$ and $$S(\tau_{-1})=0$$. 
$$E_{\mathrm{demo}}$$ is the expectation over demonstrations,
and $$C$$ scales $$R_{t+1}$$ to the range of $$\tilde{G}_0$$. 
$$R_{T+2}$$ is the correction reward (see RUDDER paper), 
with zero expectation for demonstrations:  
$$E_{\mathrm{demo}} \left[ R_{T+2}\right] = 0$$.  
Since $$\tau_t=e_{0:t}$$ and $$e_t=f(s_t,a_t)$$, we can set
$$g((s,a)_{0:t})=S(\tau_t) C $$.  
We ensured strict return equivalence, since
$$G_0=\sum_{t=0}^{T+1} R_{t+1} = \tilde{G}_0$$.  
The redistributed reward depends only on the past,
that is, $$R_{t+1}=h((s,a)_{0:t})$$.  
For computational efficiency, the profile alignment of $$\tau_{t-1}$$
can be extended to a profile alignment for $$\tau_t$$ like exact matches are 
extended to high-scoring sequence pairs with the [BLAST algorithm](http://cs.brown.edu/courses/csci1810/resources/ch1_readings/Basic_local_alignment_search_tool.pdf).