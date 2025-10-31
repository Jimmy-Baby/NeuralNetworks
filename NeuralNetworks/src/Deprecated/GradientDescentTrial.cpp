// Precompiled headers
#include "Pch.h"

#include "GradientDescentTrial.h"
#include "Utility/Math.h"

namespace GradientDescentTrial
{
	namespace
	{
		// Training data for the model: y = x * 50
		constexpr double GTrainingData[][2] =
		{
			{ 0.0, 0.0 },  // 0 * 50 = 0
			{ 1.0, 50.0 },     // 1 * 50 = 50
			{ 2.0, 100.0 },    // 2 * 50 = 100
			{ 3.0, 150.0 },    // 3 * 50 = 150
			{ 4.0, 200.0 },    // 4 * 50 = 200
		};

		double CalculateCost(const double weight)
		{
			double result = 0.0;
			constexpr size_t trainingDataSize = std::size(GTrainingData);

			for (const auto& trainingSample : GTrainingData)
			{
				// Get the current training sample
				const double trainingSampleX = trainingSample[0];
				const double trainingSampleY = trainingSample[1];

				// Use the model to evaluate an output
				const double modelPrediction = trainingSampleX * weight;

				// Subtract the correct output from the model's predicted output to get the amount it was 'off' by
				const double predictionError = modelPrediction - trainingSampleY;

				// Square the error to get a positive number, and add it to the result
				result += predictionError * predictionError;
			}

			// Average out all the accumulated error (Mean Squared Error)
			result /= trainingDataSize;

			return result;
		}

		// Calculate the derivative of the cost function with respect to weight
		// For MSE cost: dC/dw = (2/n) * Σ(predicted - actual) * x
		double CalculateGradient(const double weight)
		{
			double result = 0.0;
			constexpr size_t trainingDataSize = std::size(GTrainingData);

			for (const auto& trainingSample : GTrainingData)
			{
				const double trainingInput = trainingSample[0];
				const double expectedResult = trainingSample[1];

				const double modelPrediction = trainingInput * weight;
				const double modelError = modelPrediction - expectedResult;

				// Gradient of MSE: 2 * error * input
				result += 2.0 * modelError * trainingInput;
			}

			// Average out all of the gradients
			result /= trainingDataSize;

			return result;
		}
	}

	// Actual formula:
	// y = x * 50
	//
	// Model formula:
	// y = x * w

	void Run()
	{
		// Calculate our starting weight
		const double randomDouble = Math::RandomFloat();
		double weight = randomDouble * 10.0;

		printf("Initial Cost = %.15f, Weight = %.15f\n", CalculateCost(weight), weight);

		// Gradient descent hyperparameters
		constexpr double learningRate = 1e-2;
		constexpr double convergenceThreshold = 1e-10;
		constexpr size_t maxIterations = 10000;

		for (size_t iteration = 0; iteration < maxIterations; ++iteration)
		{
			// Calculate the current cost
			const double cost = CalculateCost(weight);

			// Check for convergence
			if (cost < convergenceThreshold)
			{
				printf("Converged at iteration %llu with cost %.15f\n", iteration, cost);
				break;
			}

			// Calculate the gradient of the cost function
			const double gradient = CalculateGradient(weight);

			// Update weight using gradient descent
			weight -= learningRate * gradient;

			// Print results of this iteration (every 100 iterations to reduce output)
			if (iteration % 100 == 0 || iteration < 10)
			{
				printf("Iteration %llu: Cost = %.15f, Weight = %.15f, Gradient = %.15f\n", 
					iteration, cost, weight, gradient);
			}
		}

		printf("------------------------------\n");
		printf("Final weight = %.15f (expected: ~50.0)\n", weight);
		printf("------------------------------\n");
		printf("Results (Model Predictions):\n");

		for (const auto& trainingSample : GTrainingData)
		{
			const double x = trainingSample[0];
			const double expectedY = trainingSample[1];
			const double predictedY = x * weight;

			printf("x = %.1f: predicted = %.2f, expected = %.2f, error = %.2f\n", 
				x, predictedY, expectedY, std::abs(predictedY - expectedY));
		}

		printf("\nAdditional predictions:\n");
		for (int index = 5; index < 10; ++index)
		{
			const double x = index;
			const double predictedY = x * weight;
			const double expectedY = x * 50.0;

			printf("x = %.1f: predicted = %.2f, expected = %.2f, error = %.2f\n", 
				x, predictedY, expectedY, std::abs(predictedY - expectedY));
		}
	}
}
