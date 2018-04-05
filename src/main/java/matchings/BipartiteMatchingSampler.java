package matchings;

import java.util.Collections;
import java.util.List;

import bayonet.distributions.Multinomial;
import bayonet.distributions.Random;
import blang.core.LogScaleFactor;
import blang.distributions.Generators;
import blang.mcmc.ConnectedFactor;
import blang.mcmc.SampledVariable;
import blang.mcmc.Sampler;

/**
 * Each time a Permutation is encountered in a Blang model, 
 * this sampler will be instantiated. 
 */
public class BipartiteMatchingSampler implements Sampler {
  /**
   * This field will be populated automatically with the 
   * permutation being sampled. 
   */
  @SampledVariable BipartiteMatching matching;
  /**
   * This will contain all the elements of the prior or likelihood 
   * (collectively, factors), that depend on the permutation being 
   * resampled. 
   */
  @ConnectedFactor List<LogScaleFactor> numericFactors;

  @Override
  public void execute(Random rand) {
	  int n = this.matching.componentSize();
	  int i = rand.nextInt(n);
	  int m = this.matching.free2().size();
	  double log_pi_current = logDensity();
	  int connection_i = this.matching.getConnections().get(i);
	  
	  if (connection_i == BipartiteMatching.FREE) {
		  int j = rand.nextInt(m);
		  this.matching.getConnections().set(i, matching.free2().get(j));				  
	  }
	  else {
		  int j = rand.nextInt(m+1);
		  if (j == m) {
			  this.matching.getConnections().set(i, BipartiteMatching.FREE);
		  }
		  else {
			  this.matching.getConnections().set(i, matching.free2().get(j));
		  }
	  }
	  
	  double log_pi_new = logDensity();
	  boolean accept_proposal = Generators.bernoulli(rand,Math.exp(log_pi_new-log_pi_current));
	  
	  if (!accept_proposal) {
		  this.matching.getConnections().set(i, connection_i);
		  }
	  
	  
  }
  
  private double logDensity() {
    double sum = 0.0;
    for (LogScaleFactor f : numericFactors)
      sum += f.logDensity();
    return sum;
  }
}
