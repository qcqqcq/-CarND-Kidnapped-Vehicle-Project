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

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;
	double init_weight = 1;

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	double sample_x = dist_x(gen);
	double sample_y = dist_y(gen);
	double sample_psi = dist_theta(gen);

	for (int i = 0; i < num_particles; i++) {
		Particle part;
		part.id = i;
		part.x = dist_x(gen);
		part.y = dist_y(gen);
		part.theta = dist_theta(gen);
		part.weight = init_weight;

		particles.push_back(part);
		weights.push_back(init_weight);
	}


	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // Set up noise
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);


	for (int i = 0; i < num_particles; i++) {
		Particle part = particles[i];
		if (yaw_rate != 0) {
			part.x = part.x + (velocity / yaw_rate)*(sin(part.theta + yaw_rate*delta_t) - sin(part.theta));

			part.y = part.y + (velocity / yaw_rate)*(cos(part.theta) - cos(part.theta + yaw_rate*delta_t));

			part.theta = part.theta + yaw_rate*delta_t;
		}
		else {
			part.x = part.x + velocity*delta_t*cos(part.theta);
			part.y = part.y + velocity*delta_t*sin(part.theta);
		}

        part.x = part.x + dist_x(gen);
        part.y = part.y + dist_y(gen);
        part.theta = part.theta + dist_theta(gen);

		particles[i] = part;
	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html


	for (int i = 0; i < num_particles; i++) {
		Particle part = particles[i];
		part.weight = 1.0;

		// Map from particle coordinates to map coords
		// rotate

		for (int j = 0; j < observations.size(); j++){
			LandmarkObs obs = observations[j];
			double obs_x_rot = obs.x*cos(part.theta) - obs.y*sin(part.theta);
			double obs_y_rot = obs.x*sin(part.theta) + obs.y*cos(part.theta);

			double obs_x_map = part.x + obs_x_rot;
			double obs_y_map = part.y + obs_y_rot; 

			
			// For each sensor measurement, associate closest landmark
			double closest_dist = sensor_range;

			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				Map::single_landmark_s lm = map_landmarks.landmark_list[k];				
				double lm_dist = dist(lm.x_f, lm.y_f, obs_x_map, obs_y_map);
				// Update closest landmark
				if (lm_dist < closest_dist) {
					obs.id = lm.id_i;
					closest_dist = lm_dist;					
				}
			}		

			// Calculate probability for each landmark
			double delta_x = obs_x_map - map_landmarks.landmark_list[obs.id-1].x_f;
			double delta_y = obs_y_map - map_landmarks.landmark_list[obs.id-1].y_f;
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];

			double p_obs = exp(-delta_x*delta_x / (2.0*sig_x*sig_x) - delta_y*delta_y / (2.0*sig_y*sig_y));
			
			// Combine p(obs) to get new final weight
			part.weight = part.weight*p_obs;

		}
	
		particles[i] = part;
		weights[i] = part.weight;
	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	std::default_random_engine gen;
	std::discrete_distribution<int> dist(weights.begin(),weights.end());

	std::vector<Particle> new_particles;
	
	for (int i = 0; i < particles.size(); i++) {
		int chosen_part_id = dist(gen);
		new_particles.push_back(particles[chosen_part_id]);
	}

	particles = new_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
