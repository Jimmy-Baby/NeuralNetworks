#include "PatternFiniteDiff.h"
#include "../Math.h"

#include <cstdio>
#include <cstdlib>
#include <format>

namespace PatternFiniteDiff
{
	float g_TrainingData[][2] =
	{
		{ 0.000000, 00.000000 },
		{ 1.000000, 04.000000 },
		{ 2.000000, 08.000000 },
		{ 3.000000, 12.000000 },
		{ 4.000000, 16.000000 },
	};

	float CalculateCost(const float Weight, const float Bias)
	{
		float Result = 0.0f;

		constexpr size_t TrainingDataSize = std::size(g_TrainingData);

		for (size_t Index = 0; Index < TrainingDataSize; ++Index)
		{
			// Get the current training sample
			const float TrainingSampleX = g_TrainingData[Index][0];

			// Use the model to evaluate an output
			const float ModelPrediction = TrainingSampleX * Weight + Bias;

			// Subtract the correct output from the model's predicted output to get the amount it was 'off' by
			const float PredictionError = ModelPrediction - g_TrainingData[Index][1];

			// Square the error to get a positive number, and add it to the result
			Result += powf(PredictionError, 2.0f);
		}

		// Average out all the accumulated error
		Result /= TrainingDataSize;

		return Result;
	}

	// Actual formula:
	// y = x * 4
	//
	// Model formula:
	// y = x * w

	void Run()
	{
		// Set our starting weight
		float Weight = Math::RandomFloat() * 10.0f;

		// Set our starting bias
		float Bias = Math::RandomFloat() * 10.0f;

		printf("Cost = %f, Weight = %f, Bias = %f\n", CalculateCost(Weight, Bias), Weight, Bias);

		for (size_t Index = 0; Index < 100000; ++Index)
		{
			constexpr float Rate = 1e-4f;

			// Finite diff
			constexpr float Epsilon = 1e-3f;

			// Calculate the current error/cost
			const float Cost = CalculateCost(Weight, Bias);

			// Calculate the weight wiggle using Finite Difference -> f(a + h) - f(a)
			const float WiggleWeight = (CalculateCost(Weight + Epsilon, Bias) - Cost) / Epsilon;

			// Calculate the weight wiggle using Finite Difference -> f(a + h) - f(a)
			const float WiggleBias = (CalculateCost(Weight, Bias + Epsilon) - Cost) / Epsilon;

			// Adjust the weight
			Weight -= Rate * WiggleWeight;

			// Adjust the bias
			Bias -= Rate * WiggleBias;

			// Calculate the new cost after adjusting
			const float NewCost = CalculateCost(Weight, Bias);

			// Print results of this iteration
			printf("Cost = %f, Weight = %f, Bias = %f\n", CalculateCost(Weight, Bias), Weight, Bias);

			if (std::format("{:f}", NewCost) == "0.000000")
			{
				printf("Breaking!\n");
				break;
			}
		}

		printf("------------------------------\n");
		printf("Weight = %f, Bias = %f\n", Weight, Bias);
		printf("------------------------------\n");
		printf("Results:\n");

		for (int Index = 0; Index < 10; ++Index)
		{
			const float X = Index + 1.0f;
			const float Y = X * Weight + Bias;

			printf("{ %04.1f, %04.1f }\n", X, Y);
		}
	}
}
