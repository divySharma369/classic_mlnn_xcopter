/**
 * @file RLHF.h
 * @author classic_mlnn_xopter
 * @brief Reinforcement Learning from Human Feedback (RLHF) scaffolding 
 *        to allow the drone to auto-learn weights as it flies along in the sky.
 */

#ifndef RLHF_H
#define RLHF_H

#include <vector>

class RLHFController {
public:
    RLHFController(float learningRate = 0.01) : lr(learningRate) {}

    /**
     * @brief Computes the reward based on human pilot overrides during flight.
     *        If the drone is stabilizing well but the pilot has to aggressively 
     *        correct it, the reward is negative.
     * 
     * @param desired_state The desired pilot input (stick position).
     * @param actual_state The actual drone state (gyroscope/accelerometer).
     * @param human_override_intensity How much the human had to manually fight the drone.
     * @return float The calculated reward.
     */
    float computeReward(float desired_state, float actual_state, float human_override_intensity) {
        // Simple reward heuristic:
        // Reward is high if desired state matches actual state, and human override is low.
        float error = desired_state - actual_state;
        float penalty = human_override_intensity * 0.5f;
        float reward = - (error * error) - penalty;
        
        return reward;
    }

    /**
     * @brief Automatically adjusts the NN weights while the drone is flying in the sky.
     *        This function bridges the BPNN architecture to dynamically adjust 
     *        based on the human feedback reward signal.
     * 
     * @param current_weights The current weights of the BPNN layer.
     * @param reward The computed RLHF reward signal.
     * @param state_features The state features observed during the time step.
     */
    void updateWeights(std::vector<std::vector<float>>& current_weights, 
                       float reward, 
                       const std::vector<float>& state_features) {
        // Placeholder RL algorithm (e.g., Policy Gradient or simple Actor-Critic update)
        // W_new = W_old + LR * Reward * Gradient
        
        for (size_t i = 0; i < current_weights.size(); i++) {
            for (size_t j = 0; j < current_weights[i].size(); j++) {
                // Simplified weight update mapping gradient to state features
                float feature_val = (j < state_features.size()) ? state_features[j] : 1.0f;
                current_weights[i][j] += lr * reward * feature_val;
            }
        }
    }

private:
    float lr; // Learning rate for the RLHF module
};

#endif /* RLHF_H */
