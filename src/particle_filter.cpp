/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 700;
	double i_weight = 1.0/num_particles;
	
	//standard deviations
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];
	//generate distribution
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	//sample from the distribution
	for(int i = 0; i < num_particles; ++i){
		Particle ptc;
		ptc.id = i+1;
		//sample values using generator
		ptc.x = dist_x(gen);
		ptc.y = dist_y(gen);
		ptc.theta = dist_theta(gen);
		//add common weight
		ptc.weight = i_weight;
		//add to particles vector
		particles.push_back(ptc);
	}
	
	//set initialized flag to true
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	//common values
	double vel_yaw_ratio = yaw_rate!=0 ? ((double)velocity/yaw_rate) : -1.0;
	//predict
	for(int i=0; i < num_particles; ++i){
		double cos_theta = cos(particles[i].theta);
		double sin_theta = sin(particles[i].theta);
		double d_x, d_y;
		
		if(vel_yaw_ratio > -0.5){
			d_x = vel_yaw_ratio * ( sin(particles[i].theta + yaw_rate*delta_t) - sin_theta );
			d_y = vel_yaw_ratio * ( -cos(particles[i].theta + yaw_rate*delta_t) + cos_theta );
		}else{
			d_x = velocity * cos_theta * delta_t;
			d_y = velocity * sin_theta * delta_t;
		}
		particles[i].theta += yaw_rate * delta_t + std_pos[2];
		particles[i].x += d_x + std_pos[0];
		particles[i].y += d_y + std_pos[1];
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	//create local copy of observations, so that we can pop used observations
	std::vector<double> obs;
	for(int i=0; i < observations.size(); i++){
		obs.push_back(observations[i]);
	}
	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	double w_max = *std::max_element(weights.begin(), weights.end());
	int index = 0;
	double beta = 0;
	std::vector<Particle> p_smp;
	for(int i=0; i < num_particles; i++){
		beta += w_max;
		while(true){
			if(index >= num_particles)
				index = 0;
			if(weights[index] < beta){
				beta -= weights[index];
				index++;
			}else{
				p_smp.push_back(particles[index]);
				break;
			}
		}
	}
	particles = p_smp;
	// update weight vector
	for(int i=0; i< num_particles; i++)
		weights[i] = particles[i].weight;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
