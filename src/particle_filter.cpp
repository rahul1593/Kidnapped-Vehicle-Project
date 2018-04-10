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
	num_particles = 50;
	const double i_weight = 1.0;
	
	//standard deviations
	const double std_x = std[0];
	const double std_y = std[1];
	const double std_theta = std[2];
	//generate distribution
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	//sample from the distribution
	for(int i = 0; i < num_particles; ++i){
		Particle ptc;
		ptc.id = i;
		//sample values using generator
		ptc.x = dist_x(gen);
		ptc.y = dist_y(gen);
		ptc.theta = dist_theta(gen);
		//add common weight
		ptc.weight = i_weight;
		//add particle and weight to respective vector
		particles.push_back(ptc);
        weights.push_back(i_weight);
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
    const bool turning = (fabs(yaw_rate) != 0.0001);
	const double vel_yaw_ratio =  (turning ? ((double)velocity/yaw_rate) : 0.0);
    
    default_random_engine gen;
    normal_distribution<double> noise_x(0.0, std_pos[0]);
    normal_distribution<double> noise_y(0.0, std_pos[1]);
    normal_distribution<double> noise_theta(0.0, std_pos[2]);
    
    //predict
	for(int i=0; i < num_particles; ++i){
		double d_x, d_y;
		const double cos_theta = cos(particles[i].theta);
		const double sin_theta = sin(particles[i].theta);
		const double n_x = noise_x(gen);
		const double n_y = noise_y(gen);
		const double n_t = noise_theta(gen);
        
		if(turning){//if yaw rate is not zero
			d_x = vel_yaw_ratio * ( sin(particles[i].theta + yaw_rate*delta_t) - sin_theta );
			d_y = vel_yaw_ratio * ( -cos(particles[i].theta + yaw_rate*delta_t) + cos_theta );
		}else{
			d_x = velocity * cos_theta * delta_t;
			d_y = velocity * sin_theta * delta_t;
		}
		particles[i].theta += yaw_rate * delta_t + n_t;
		particles[i].x += d_x + n_x;
		particles[i].y += d_y + n_y;
	}
}

/*
 * Transform the points from car's coordinate system to map's coordinate system.
 */
void transform2MapCoordinates(struct Particle p, std::vector<LandmarkObs> observations, std::vector<LandmarkObs> &map_observations){
    const double cos_theta = cos(p.theta);
    const double sin_theta = sin(p.theta);
    
    for(int i=0; i < observations.size(); i++){
        const double tx = p.x + (cos_theta * observations[i].x) - (sin_theta * observations[i].y);
        const double ty = p.y + (cos_theta * observations[i].y) + (sin_theta * observations[i].x);
        const int tid = observations[i].id;
        map_observations.push_back((LandmarkObs){
            tid,
            tx,
            ty
        });
    }
}

/*
 * Find Observations which are in range
 */
void observationsInRange(double sensor_range, struct Particle p, Map map_landmarks, std::vector<LandmarkObs> &observations_in_range){
    
    for(int i=0; i < map_landmarks.landmark_list.size(); i++){
        const double cdif = dist(p.x, p.y, map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f);
        //store observation if in range
        if(cdif <= sensor_range){
            observations_in_range.push_back((LandmarkObs){
                map_landmarks.landmark_list[i].id_i,
                map_landmarks.landmark_list[i].x_f,
                map_landmarks.landmark_list[i].y_f
            });
        }
	}
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for(int i=0; i < observations.size(); i++){
        double min_dif = 10000000.0;
        int min_id = -1;
		for(int j=0; j < predicted.size(); j++){
            const double cdif = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
            if(cdif < min_dif){
                min_id = predicted[j].id;
                min_dif = cdif;
            }
        }
        //store prediction id for observation
        observations[i].id = min_id;
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
	//   and the following is a good resource for the actual equation to implement (look at equation 3.33)
	//   http://planning.cs.uiuc.edu/node99.html
	// values for calculating weights
    const double dx = 0.5/(std_landmark[0] * std_landmark[0]);
    const double dy = 0.5/(std_landmark[1] * std_landmark[1]);
    const double norm_f = sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
    //std::cout << weights[0] << std::endl;
    // update weights for all particles
    for(int i=0; i < num_particles; i++){
        std::vector<LandmarkObs> map_observations;
        std::vector<LandmarkObs> observations_in_range;
        //get observations in map coordinates
        transform2MapCoordinates(particles[i], observations, map_observations);
        //get landmarks in sensor range
        observationsInRange(sensor_range, particles[i], map_landmarks, observations_in_range);
        //associate observations in range to landmarks
        dataAssociation(observations_in_range, map_observations);
        // update weights
        double n_w = 1.0;
        for(int j=0; j < map_observations.size(); j++){
            const int id = map_observations[j].id;
            const double dfx = map_observations[j].x - observations_in_range[id].x;
            const double dfy = map_observations[j].y - observations_in_range[id].y;
            //std::cout << dfx << " " << dfy << std::endl;
            n_w *= (exp(-( (dfx*dfx*dx) + (dfy*dfy*dy) )) / norm_f);
        }
        particles[i].weight = n_w;
        weights[i] = n_w;
    }
    //std::cout << weights[0] << std::endl;
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
