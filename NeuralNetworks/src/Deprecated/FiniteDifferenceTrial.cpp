// Precompiled headers
#include "Pch.h"

#include "FiniteDifferenceTrial.h"
#include "Utility/Math.h"

namespace FiniteDifferenceTrial
{
	namespace
	{
		// Training data for the model: y = x * 4
		constexpr float GTrainingData[][2] =
		{
			{ 0.0f, 0.0f },  // 0 * 4 = 0
			{ 1.0f, 4.0f },  // 1 * 4 = 4
			{ 2.0f, 8.0f },  // 2 * 4 = 8
			{ 3.0f, 12.0f }  // 3 * 4 = 12
		};

		float CalculateCost(const float weight)
		{
			float result = 0.0f;

			constexpr size_t trainingDataSize = std::size(GTrainingData);
			for (const auto& trainingSample : GTrainingData)
			{
				// Get the current training sample
				const float trainingSampleX = trainingSample[0];
				const float trainingSampleY = trainingSample[1];

				// Use the model to evaluate an output
				const float modelPrediction = trainingSampleX * weight;

				// Subtract the correct output from the model's predicted output to get the amount it was 'off' by
				const float predictionError = modelPrediction - trainingSampleY;

				// Square the error to get a positive number, and add it to the result
				// Using x * x is more efficient than powf(x, 2.0f)
				result += predictionError * predictionError;
			}

			// Average out all the accumulated error (Mean Squared Error)
			result /= trainingDataSize;

			return result;
		}
	}

	// Actual formula:
	// y = x * 4
	//
	// Model formula:
	// y = x * w

	void Run()
	{
		// Set our starting weight
		float weight = Math::RandomFloat() * 10.0f;

		printf("Initial Cost = %f, Weight = %f\n", CalculateCost(weight), weight);
		printf("Initial prediction: 2 * %f = %f (expected: 8)\n", weight, 2.0f * weight);

		// Finite difference hyperparameters
		constexpr float learningRate = 1e-1f;   // Learning rate for gradient descent
		constexpr float epsilon = 1e-3f;        // Small value for numerical derivative approximation
		constexpr float convergenceThreshold = 1e-6f;  // Stop when cost is below this threshold
		constexpr size_t maxIterations = 10000;

		for (size_t iteration = 0; iteration < maxIterations; ++iteration)
		{
			// Calculate the current error/cost
			const float cost = CalculateCost(weight);

			// Check for convergence
			if (cost < convergenceThreshold)
			{
				printf("Converged at iteration %llu with cost %f\n", iteration, cost);
				break;
			}

			// Calculate the derivative using Finite Difference: [f(a + h) - f(a)] / h
			const float derivative = (CalculateCost(weight + epsilon) - cost) / epsilon;

			// Adjust the weight using gradient descent
			weight -= learningRate * derivative;

			// Print results of this iteration (every 1000 iterations to reduce output)
			if (iteration % 1000 == 0 || iteration < 10)
			{
				printf("Iteration %llu: Cost = %f, Weight = %f, Derivative = %f\n", 
					iteration, cost, weight, derivative);
			}
		}

		printf("------------------------------\n");
		printf("Final Weight = %f (expected: ~4.0)\n", weight);
		printf("------------------------------\n");
		printf("Results (Model Predictions):\n");

		for (const auto& trainingSample : GTrainingData)
		{
			const float x = trainingSample[0];
			const float expectedY = trainingSample[1];
			const float predictedY = x * weight;

			printf("x = %.1f: predicted = %.2f, expected = %.2f, error = %.2f\n", 
				x, predictedY, expectedY, std::abs(predictedY - expectedY));
		}
	}
}
