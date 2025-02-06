#include "PatternFiniteDiff.h"
#include "../CustomMath.h"

#include <cstdio>
#include <cstdlib>
#include <format>

namespace PatternFiniteDiff
{
	float g_TrainingData[][2] =
	{
		{ 0, 1 },
		{ 1, 0 }
	};

	float CalculateCost(const float Weight)
	{
		float Result = 0.0f;

		constexpr size_t TrainingDataSize = std::size(g_TrainingData);

		for (size_t Index = 0; Index < TrainingDataSize; ++Index)
		{
			// Get the current training sample
			const float TrainingSampleX = g_TrainingData[Index][0];

			// Use the model to evaluate an output
			const float ModelPrediction = TrainingSampleX * Weight;

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

		printf("Cost = %f, Weight = %f\n", CalculateCost(Weight), Weight);
		printf("2 == 8 == %f\n", 2 * Weight);

		for (size_t Index = 0; Index < 10000; ++Index)
		{
			constexpr float Rate = 1e-3f;

			// Finite diff
			constexpr float Epsilon = 1e-3f;

			// Calculate the current error/cost
			const float Cost = CalculateCost(Weight);

			// Calculate the weight wiggle using Finite Difference -> f(a + h) - f(a)
			const float WiggleWeight = (CalculateCost(Weight + Epsilon) - Cost) / Epsilon;

			// Adjust the weight
			Weight -= Rate * WiggleWeight;
			
			// Calculate the new cost after adjusting
			const float NewCost = CalculateCost(Weight);

			// Print results of this iteration
			printf("Cost = %f, Weight = %f\n", CalculateCost(Weight), Weight);

			if (std::format("{:f}", NewCost) == "0.000000")
			{
				printf("Breaking!\n");
				break;
			}
		}

		printf("------------------------------\n");
		printf("Weight = %f\n", Weight);
		printf("------------------------------\n");
		printf("Results:\n");

		for (int Index = 0; Index < std::size(g_TrainingData); ++Index)
		{
			//const float X = Index + 1.0f;
			const float Y = Index * Weight;

			printf("{ %d, %04.1f }\n", Index, Y);
		}
	}
}
