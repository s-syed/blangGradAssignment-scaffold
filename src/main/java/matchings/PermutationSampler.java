package matchings;

import java.util.Collections;
import java.util.List;

import bayonet.distributions.Random;
import blang.core.LogScaleFactor;
import blang.distributions.Generators;
import blang.mcmc.ConnectedFactor;
import blang.mcmc.SampledVariable;
import blang.mcmc.Sampler;
import briefj.collections.UnorderedPair;

/**
 * Each time a Permutation is encountered in a Blang model, 
 * this sampler will be instantiated. 
 */
public class PermutationSampler implements Sampler {
  /**
   * This field will be populated automatically with the 
   * permutation being sampled. 
   */
  @SampledVariable Permutation permutation;
  /**
   * This will contain all the elements of the prior or likelihood 
   * (collectively, factors), that depend on the permutation being 
   * resampled. 
   */
  @ConnectedFactor List<LogScaleFactor> numericFactors;

  @Override
  public void execute(Random rand) {
    // Fill this. 
	  int n = this.permutation.componentSize();
	  int i = rand.nextInt(n);
	  int j = rand.nextInt(n);
	  double log_pi_current = logDensity();
	  Collections.swap(this.permutation.getConnections(), i, j);
	  double log_pi_new = logDensity();
	  
	  boolean accept_proposal = Generators.bernoulli(rand,Math.exp(log_pi_new-log_pi_current));
	  
	  if (!accept_proposal) {
		  Collections.swap(this.permutation.getConnections(), i, j);
	  }
  }
  
  private double logDensity() {
    double sum = 0.0;
    for (LogScaleFactor f : numericFactors)
      sum += f.logDensity();
    return sum;
  }
}
